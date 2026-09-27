"""Blocking V0: multi-pass key blocking inside one country partition.

Passes (each can be switched on/off and capped in the config):
    exact_name      key = name_norm                                 (Block A)
    addr_num_word   key = (numeric address token, useful address word) (Block B)
    addr_rare_word  key = useful address word with pool DF <= max_df (Block B)
    name_token      key = name_core token with pool DF <= max_df    (off in v0; EXP-B-02 D)
    exact_core      key = name_core (legal forms removed)            (EXP-B-02 A)
    sorted_core     key = sorted name_core tokens                     (EXP-B-02 B)
    name_addr       key = (rarest name_core token, rarest useful address word) (EXP-B-04 B)
    addr_pair       key = unordered pair of the record's rarest useful address words (EXP-B-04 C)
    name_num        key = (rarest name-view token, rarest house number)          (EXP-B-09)
  option strip_leading_zeros: "04126" == "4126" for address-number keys (EXP-B-02 C)

Mechanics (all numpy/pyarrow, no Python loops over records):
* tokens are exploded with pyarrow and dictionary-encoded over S1+pool together, so
  identical strings get identical integer ids;
* keys whose pool block exceeds ``max_block_size`` are dropped (no giant blocks);
* candidate pairs of all passes are unioned and de-duplicated, each pair gets a cheap
  score = sum over shared keys of pass_weight * log(1 + N_pool / block_size);
* the top ``top_k`` pairs per S1 by cheap score (ties -> lower pool index) are kept.
  That capped set is exactly what the matcher scores and what candidate_pairs.tsv holds.
* S1 records are processed in chunks sized so that the raw (pre-dedup) expansion stays
  below ``max_raw_pairs_per_chunk``, which bounds peak memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ber.normalization.views import ADDRESS_STOPWORDS

PASS_ORDER = ("exact_name", "addr_num_word", "addr_rare_word", "name_token", "exact_core", "sorted_core", "name_addr", "addr_pair", "name_num")

_LEADING_ZEROS = r"(^|\s)0+(\d)"


def strip_leading_zeros(arr: pa.Array) -> pa.Array:
    """``"04126 007 0"`` -> ``"4126 7 0"``: leading zeros removed from every token that
    starts with zeros followed by a digit (a lone ``0`` is kept). EXP-B-02 (C)."""
    return pc.replace_substring_regex(arr, pattern=_LEADING_ZEROS, replacement=r"\1\2")


def sort_tokens(arr: pa.Array) -> pa.Array:
    """``"parts advanced"`` -> ``"advanced parts"`` (order-insensitive name key). EXP-B-02 (B)."""
    return pa.array([" ".join(sorted(x.split())) for x in arr.to_pylist()], pa.string())


# ----------------------------------------------------------------------------- helpers
def _combine(a: pa.ChunkedArray | pa.Array) -> pa.Array:
    return a.combine_chunks() if isinstance(a, pa.ChunkedArray) else a


def encode_columns(
    columns: list[pa.Array | pa.ChunkedArray], split: bool, chunk_rows: int = 250_000, transform=None
) -> tuple[np.ndarray, np.ndarray, pa.Array]:
    """Encode strings of several columns into one integer space, chunk by chunk.

    The columns are laid out one after another (record index of column j starts at the
    total length of columns < j). ``split=True`` whitespace-splits each string into tokens
    (duplicates inside a record removed); ``split=False`` uses the whole string as one
    token. Empty strings/tokens are skipped. ``transform`` (pa.Array -> pa.Array) is
    applied to every chunk before encoding (e.g. leading-zero stripping, token sorting). Returns (record int32, token id int32,
    vocabulary) with ids consistent across all columns.

    Working in record chunks keeps the transient Arrow/numpy memory small: only one
    chunk's exploded tokens exist at a time; the global vocabulary grows incrementally.
    """
    vocab = pa.array([], pa.string())
    recs: list[np.ndarray] = []
    toks: list[np.ndarray] = []
    offset = 0
    for col in columns:
        n = len(col)
        for a in range(0, n, chunk_rows):
            sub = _combine(col.slice(a, chunk_rows))
            if transform is not None:
                sub = transform(sub)
            if split:
                lists = pc.utf8_split_whitespace(sub)
                rec = pc.list_parent_indices(lists)
                flat = pc.list_flatten(lists)
            else:
                flat = sub
                rec = pa.array(np.arange(len(sub), dtype=np.int32))
            keep = pc.not_equal(flat, "")
            flat = pc.filter(flat, keep)
            rec_np = pc.filter(rec, keep).to_numpy(zero_copy_only=False).astype(np.int64)
            if len(flat) == 0:
                continue
            enc = pc.dictionary_encode(flat)
            local_vocab = enc.dictionary
            local_idx = enc.indices.to_numpy(zero_copy_only=False).astype(np.int64)
            if split:  # de-duplicate tokens within a record (records never span chunks)
                code = np.unique(rec_np * len(local_vocab) + local_idx)
                rec_np, local_idx = code // len(local_vocab), code % len(local_vocab)
            gid = pc.fill_null(pc.index_in(local_vocab, value_set=vocab), -1).to_numpy(zero_copy_only=False).astype(np.int64)
            new = gid < 0
            gid[new] = len(vocab) + np.arange(int(new.sum()))
            if new.any():
                vocab = pa.concat_arrays([vocab, pc.filter(local_vocab, pa.array(new))])
            recs.append((rec_np + offset + a).astype(np.int32))
            toks.append(gid[local_idx].astype(np.int32))
        offset += n
    if not recs:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), vocab
    return np.concatenate(recs), np.concatenate(toks), vocab


def explode_tokens(arr: pa.Array) -> tuple[np.ndarray, np.ndarray, pa.Array]:
    """Whitespace-split each string -> (record index, token id, token dictionary)."""
    return encode_columns([arr], split=True)


def rank_within_groups(groups_sorted: np.ndarray) -> np.ndarray:
    """0-based position of each row inside its (contiguous) group."""
    n = len(groups_sorted)
    if n == 0:
        return np.zeros(0, np.int64)
    new = np.empty(n, bool)
    new[0] = True
    new[1:] = groups_sorted[1:] != groups_sorted[:-1]
    first = np.flatnonzero(new)
    sizes = np.diff(np.append(first, n))
    return np.arange(n) - np.repeat(first, sizes)


def ragged_gather(starts: np.ndarray, counts: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Concatenate values[starts[i] : starts[i]+counts[i]] for all i."""
    total = int(counts.sum())
    if total == 0:
        return values[:0]
    offsets = np.repeat(np.cumsum(counts) - counts, counts)
    idx = np.repeat(starts, counts) + (np.arange(total) - offsets)
    return values[idx]


def cross_within_records(rec_a, tok_a, rec_b, tok_b, vocab_b: int):
    """All (a, b) token pairs that belong to the same record. Inputs sorted by record."""
    lo = np.searchsorted(rec_b, rec_a, "left")
    hi = np.searchsorted(rec_b, rec_a, "right")
    cnt = hi - lo
    out_rec = np.repeat(rec_a, cnt)
    out_a = np.repeat(tok_a, cnt).astype(np.int64)
    out_b = ragged_gather(lo, cnt, tok_b)
    return out_rec, out_a * vocab_b + out_b


def rarest_per_record(rec, tok, df, limit: int):
    """Keep at most ``limit`` tokens per record, preferring low pool DF."""
    order = np.lexsort((tok, df[tok], rec))
    rec, tok = rec[order], tok[order]
    keep = rank_within_groups(rec) < limit
    return rec[keep], tok[keep]


def cross_keys(a_rec, a_tok, a_df, a_lim, b_rec, b_tok, b_df, b_lim, vocab_b, window, same=False):
    """Per record: the ``a_lim`` rarest tokens of list A x the ``b_lim`` rarest of list B.

    ``same=True`` (A and B are the same list): unordered pairs of distinct tokens only,
    key = min_id * vocab + max_id. Computed in record windows (bounded temporaries).
    Inputs must be sorted by record (``encode_columns`` guarantees it)."""
    total = int(max(a_rec[-1] if len(a_rec) else -1, b_rec[-1] if len(b_rec) else -1)) + 1
    out_rec, out_key = [], []
    for lo in range(0, total, window):
        hi = lo + window
        a0, a1 = np.searchsorted(a_rec, [lo, hi])
        b0, b1 = np.searchsorted(b_rec, [lo, hi])
        if a0 == a1 or b0 == b1:
            continue
        ar, at = rarest_per_record(a_rec[a0:a1], a_tok[a0:a1], a_df, a_lim)
        br, bt = (ar, at) if same else rarest_per_record(b_rec[b0:b1], b_tok[b0:b1], b_df, b_lim)
        r, k = cross_within_records(ar, at, br, bt, vocab_b)
        if same:
            x, y = k // vocab_b, k % vocab_b
            keep = x < y
            r, k = r[keep], k[keep]
        out_rec.append(r)
        out_key.append(k)
    if not out_rec:
        return np.zeros(0, np.int32), np.zeros(0, np.int64)
    return np.concatenate(out_rec), np.concatenate(out_key)


def num_word_keys(n_rec, n_tok, n_df, w_rec, w_tok, w_df, word_ok, max_nums, max_words, vocab_w, window):
    """(numeric token, useful word) keys per record, computed in record windows so the
    lexsort / cross-product temporaries never cover the whole partition.
    ``n_rec`` and ``w_rec`` must be ascending (``encode_columns`` guarantees it)."""
    total = int(max(n_rec[-1] if len(n_rec) else -1, w_rec[-1] if len(w_rec) else -1)) + 1
    out_rec, out_key = [], []
    for lo in range(0, total, window):
        hi = lo + window
        a, b = np.searchsorted(n_rec, [lo, hi])
        c, d = np.searchsorted(w_rec, [lo, hi])
        if a == b or c == d:
            continue
        nr, nt = rarest_per_record(n_rec[a:b], n_tok[a:b], n_df, max_nums)
        wr, wt = w_rec[c:d], w_tok[c:d]
        ok = word_ok[wt]
        wr, wt = rarest_per_record(wr[ok], wt[ok], w_df, max_words)
        r, k = cross_within_records(nr, nt, wr, wt, vocab_w)
        out_rec.append(r)
        out_key.append(k)
    if not out_rec:
        return np.zeros(0, np.int32), np.zeros(0, np.int64)
    return np.concatenate(out_rec), np.concatenate(out_key)


def sort_by_key(key: np.ndarray, rec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort (key, rec) rows by key, then rec, returning (sorted keys, int32 recs).

    When key and rec fit together in 62 bits they are packed into one int64 and sorted
    in place (no argsort index array); otherwise falls back to lexsort."""
    rec = np.asarray(rec)
    if len(key) == 0:
        return np.asarray(key, np.int64)[:0], rec.astype(np.int32)[:0]
    rec_bits = max(int(rec.max()).bit_length(), 1)
    key_bits = max(int(key.max()).bit_length(), 1)
    if int(key.min()) >= 0 and rec_bits + key_bits <= 62:
        packed = np.asarray(key, np.int64) << rec_bits
        packed |= rec.astype(np.int64)
        packed.sort()
        mask = (1 << rec_bits) - 1
        out_rec = (packed & mask).astype(np.int32)
        packed >>= rec_bits
        return packed, out_rec
    order = np.lexsort((rec, key))
    return np.asarray(key, np.int64)[order], rec[order].astype(np.int32)


@dataclass
class PassIndex:
    name: str
    bit: int
    s1_rec: np.ndarray  # S1 local index per hitting key row (sorted)
    s1_pos: np.ndarray  # position into keys arrays
    starts: np.ndarray  # per kept key: start into pool_rec
    counts: np.ndarray  # per kept key: block size
    weight: np.ndarray  # per kept key: float32 cheap-score weight
    pool_rec: np.ndarray  # pool local indices sorted by key


# ----------------------------------------------------------------------------- blocker
class CountryBlocker:
    """Builds all pass indices for one country (S1 subset x full country pool)."""

    def __init__(self, s1: pa.Table, pool, cfg: dict):
        """``pool`` is a pyarrow Table or any object with ``num_rows`` and
        ``column(name)`` (e.g. a lazy per-column Parquet reader to save memory)."""
        self.cfg = cfg
        self.n_s1 = s1.num_rows
        self.n_pool = pool.num_rows
        self.stats: dict = {"n_s1": self.n_s1, "n_pool": self.n_pool, "passes": {}}
        self.passes: list[PassIndex] = []
        pcfg = cfg["passes"]
        stop = pa.array(sorted(ADDRESS_STOPWORDS), pa.string())

        chunk_rows = int(cfg.get("encode_chunk_rows", 250_000))

        num_transform = strip_leading_zeros if cfg.get("strip_leading_zeros", False) else None
        # EXP-N-02: the name-token passes (exact_core, sorted_core, name_addr, name_token) may read a
        # train-learned cross-script view instead of name_core; identical to name_core for any name
        # without a mapped token, so the default ("name_core") reproduces every earlier blocker.
        name_view = cfg.get("name_view", "name_core")

        def encode(col, split, transform=None):
            # S1 records first (0..n_s1-1), then pool records (n_s1..)
            return encode_columns([s1.column(col), pool.column(col)], split, chunk_rows, transform)

        # --- Block A: exact normalized name
        if pcfg.get("exact_name", {}).get("enabled", False):
            self._add_pass("exact_name", list(encode("name_norm", split=False)[:2]), pcfg["exact_name"])

        # --- Block A2: exact name_core (legal forms removed) -- EXP-B-02 (A)
        ec = pcfg.get("exact_core", {})
        if ec.get("enabled", False) and not ec.get("with_addr_words", False):
            self._add_pass("exact_core", list(encode(name_view, split=False)[:2]), ec)

        # --- Block A3: sorted name_core tokens (word-order insensitive) -- EXP-B-02 (B)
        if pcfg.get("sorted_core", {}).get("enabled", False):
            self._add_pass("sorted_core", list(encode(name_view, split=False, transform=sort_tokens)[:2]), pcfg["sorted_core"])

        # --- Block B: address tokens
        need_words = any(pcfg.get(p, {}).get("enabled", False) for p in ("addr_num_word", "addr_rare_word", "name_addr", "addr_pair")) or             (ec.get("enabled", False) and ec.get("with_addr_words", False))
        if need_words:
            w_rec, w_tok, w_vocab = encode("addr_words", split=True)
            V = len(w_vocab)
            w_df = np.bincount(w_tok[w_rec >= self.n_s1], minlength=V)
            is_stop = pc.is_in(w_vocab, value_set=stop).to_numpy(zero_copy_only=False) if V else np.zeros(0, bool)
            short = (pc.utf8_length(w_vocab).to_numpy(zero_copy_only=False) < 2) if V else np.zeros(0, bool)
            useful = ~(is_stop | short)

            c = pcfg.get("addr_rare_word", {})
            if c.get("enabled", False):
                sel = useful[w_tok] & (w_df[w_tok] <= c["max_df"])
                self._add_pass("addr_rare_word", [w_rec[sel], w_tok[sel]], c)
                del sel

            # --- EXP-B-04 (C): pairs of the record's rarest useful address words
            c = pcfg.get("addr_pair", {})
            if c.get("enabled", False):
                # only words present in the pool can ever match: never spend a slot on DF-0 words
                sel = useful[w_tok] & (w_df[w_tok] >= 1) & (w_df[w_tok] <= c["word_max_df"])
                keys = list(cross_keys(w_rec[sel], w_tok[sel], w_df, c["max_words"], w_rec[sel], w_tok[sel], w_df,
                                       c["max_words"], max(V, 1), chunk_rows, same=True))
                del sel
                self._add_pass("addr_pair", keys, c)

            # --- EXP-B-04 (B): (rarest name_core token) x (rarest useful address word)
            c = pcfg.get("name_addr", {})
            if c.get("enabled", False):
                t_rec, t_tok, t_vocab = encode(name_view, split=True)
                t_df = np.bincount(t_tok[t_rec >= self.n_s1], minlength=len(t_vocab))
                tsel = (t_df[t_tok] >= 1) & (t_df[t_tok] <= c.get("name_max_df", 10**12))
                wsel = useful[w_tok] & (w_df[w_tok] >= 1) & (w_df[w_tok] <= c["word_max_df"])
                keys = list(cross_keys(t_rec[tsel], t_tok[tsel], t_df, c["max_name_tokens"], w_rec[wsel], w_tok[wsel], w_df,
                                       c["max_words"], max(V, 1), chunk_rows))
                del t_rec, t_tok, tsel, wsel
                self._add_pass("name_addr", keys, c)

            # --- EXP-B-07: (exact word-order-insensitive name) x (useful address word). For GENERIC names
            # whose exact-name block exceeds the cap, the (name, word) block is small again; the shared
            # word may be a common one (a city), so up to `max_words` words per record, rarest first.
            if ec.get("enabled", False) and ec.get("with_addr_words", False):
                n_rec, n_tok, n_vocab = encode(name_view, split=False, transform=sort_tokens)
                n_df = np.bincount(n_tok[n_rec >= self.n_s1], minlength=len(n_vocab))
                empty = pc.equal(n_vocab, "").to_numpy(zero_copy_only=False) if len(n_vocab) else np.zeros(0, bool)
                nsel = (n_df[n_tok] >= 1) & ~empty[n_tok]
                wsel = useful[w_tok] & (w_df[w_tok] >= 1) & (w_df[w_tok] <= ec.get("word_max_df", 10**12))
                keys = list(cross_keys(n_rec[nsel], n_tok[nsel], n_df, 1, w_rec[wsel], w_tok[wsel], w_df,
                                       int(ec.get("max_words", 30)), max(V, 1), chunk_rows))
                del n_rec, n_tok, nsel, wsel
                self._add_pass("exact_core", keys, ec)

            c = pcfg.get("addr_num_word", {})
            if c.get("enabled", False):
                n_rec, n_tok, n_vocab = encode("addr_nums", split=True, transform=num_transform)
                n_df = np.bincount(n_tok[n_rec >= self.n_s1], minlength=len(n_vocab))
                word_ok = useful & (w_df <= c["word_max_df"])
                keys = list(
                    num_word_keys(n_rec, n_tok, n_df, w_rec, w_tok, w_df, word_ok, c["max_nums"], c["max_words"], max(V, 1), chunk_rows)
                )
                del n_rec, n_tok, w_rec, w_tok  # free token arrays before the index build
                self._add_pass("addr_num_word", keys, c)
            else:
                del w_rec, w_tok

        # --- optional: rare name-core tokens (EXP-B-02)
        c = pcfg.get("name_token", {})
        if c.get("enabled", False):
            t_rec, t_tok, t_vocab = encode(name_view, split=True)
            t_df = np.bincount(t_tok[t_rec >= self.n_s1], minlength=len(t_vocab))
            if c.get("pair", False):
                # EXP-B-06: key = unordered pair of the record's `max_tokens` rarest name tokens present
                # in the pool (DF >= 1, DF <= name_max_df) -- a name-only key for records whose address
                # gives no usable key (empty / generic). Same bit as name_token; block cap applies.
                sel = (t_df[t_tok] >= 1) & (t_df[t_tok] <= c.get("name_max_df", 10**12))
                k = int(c.get("max_tokens", 2))
                keys = list(cross_keys(t_rec[sel], t_tok[sel], t_df, k, t_rec[sel], t_tok[sel], t_df, k,
                                       max(len(t_vocab), 1), chunk_rows, same=True))
                del t_rec, t_tok, sel
                self._add_pass("name_token", keys, c)
            else:
                sel = t_df[t_tok] <= c["max_df"]
                self._add_pass("name_token", [t_rec[sel], t_tok[sel]], c)

        # --- EXP-B-09: (rarest name-view token) x (rarest house number). Recovers pairs that share COMMON name
        # tokens plus a house number, which the rarest-token keys miss. Same selection rules as the other
        # combination keys: pool-present tokens / numbers only (DF >= 1), `max_name_tokens` rarest tokens x
        # `max_nums` rarest zero-stripped numbers (the addr_num_word number view); block cap applies.
        c = pcfg.get("name_num", {})
        if c.get("enabled", False):
            t_rec, t_tok, t_vocab = encode(name_view, split=True)
            t_df = np.bincount(t_tok[t_rec >= self.n_s1], minlength=len(t_vocab))
            tsel = t_df[t_tok] >= 1
            n_rec, n_tok, n_vocab = encode("addr_nums", split=True, transform=num_transform)
            n_df = np.bincount(n_tok[n_rec >= self.n_s1], minlength=len(n_vocab))
            nsel = n_df[n_tok] >= 1
            keys = list(cross_keys(t_rec[tsel], t_tok[tsel], t_df, int(c["max_name_tokens"]), n_rec[nsel], n_tok[nsel], n_df,
                                   int(c["max_nums"]), max(len(n_vocab), 1), chunk_rows))
            del t_rec, t_tok, tsel, n_rec, n_tok, nsel
            self._add_pass("name_num", keys, c)

        # pass-flag storage: uint8 while every enabled pass has a bit < 8 (all configs before EXP-B-09, unchanged
        # byte for byte), uint16 once a later pass (bit >= 8) is enabled
        self.flag_dtype = np.uint16 if any(p.bit > 0xFF for p in self.passes) else np.uint8

    # ------------------------------------------------------------------ pass building
    def _add_pass(self, name: str, arrays: list, c: dict) -> None:
        """Build one pass index from ``arrays = [rec, key]``.

        The list is emptied here so the caller holds no references and the (large)
        input arrays can be freed as soon as they are split. Blocks are compressed to
        the ones some S1 actually hits before any other per-key array is built."""
        rec, key = arrays
        arrays.clear()
        is_pool = rec >= self.n_s1
        n_pool_rows = int(is_pool.sum())
        s_key, s_rec = key[~is_pool], rec[~is_pool]
        p_key, p_rec = key[is_pool], rec[is_pool] - self.n_s1
        del rec, key, is_pool
        p_key, p_rec = sort_by_key(p_key, p_rec)
        # block boundaries on the sorted keys (no second sort)
        n = len(p_key)
        if n:
            new = np.empty(n, bool)
            new[0] = True
            np.not_equal(p_key[1:], p_key[:-1], out=new[1:])
            starts = np.flatnonzero(new)
            del new
            uniq = p_key[starts]
            counts = np.diff(np.append(starts, n)).astype(np.int32)
        else:
            starts = np.zeros(0, np.int64)
            counts = np.zeros(0, np.int32)
            uniq = p_key[:0]
        del p_key
        n_keys = len(uniq)
        max_block = int(c.get("max_block_size", c.get("max_df", 10**9)))
        dropped_keys = int((counts > max_block).sum())

        # S1 lookup, then keep only hits on blocks within the size cap
        if n_keys:
            pos = np.searchsorted(uniq, s_key)
            np.minimum(pos, n_keys - 1, out=pos)
            hit = uniq[pos] == s_key
        else:
            pos = np.zeros(len(s_key), np.int64)
            hit = np.zeros(len(s_key), bool)
        n_s1_key_rows = len(s_key)
        del uniq, s_key
        if n_keys:
            hit &= counts[pos] <= max_block
        s_rec, pos = s_rec[hit], pos[hit]
        del hit

        # Compact: keep only the blocks some S1 actually hits.
        used = np.unique(pos)
        u_counts = counts[used]
        p_rec = ragged_gather(starts[used], u_counts, p_rec)
        del starts, counts
        u_starts = (np.cumsum(u_counts, dtype=np.int64) - u_counts).astype(np.int64)
        weight = (float(c.get("weight", 1.0)) * np.log1p(self.n_pool / np.maximum(u_counts, 1))).astype(np.float32)
        pos = np.searchsorted(used, pos)
        order = np.argsort(s_rec, kind="stable")
        s_rec, pos = s_rec[order].astype(np.int64), pos[order]
        del order
        self.passes.append(PassIndex(name, 1 << PASS_ORDER.index(name), s_rec, pos, u_starts, u_counts, weight, p_rec))
        self.stats["passes"][name] = {
            "s1_key_rows": int(n_s1_key_rows),
            "pool_key_rows": n_pool_rows,
            "kept_keys": int(n_keys - dropped_keys),
            "dropped_oversize_keys": dropped_keys,
            "keys_hit_by_s1": int(len(used)),
            "raw_pairs": int(u_counts[pos].sum()),
        }

    # ------------------------------------------------------------------ chunking
    def chunk_bounds(self) -> list[tuple[int, int]]:
        per_s1 = np.zeros(self.n_s1, np.int64)
        for p in self.passes:
            per_s1 += np.bincount(p.s1_rec, weights=p.counts[p.s1_pos], minlength=self.n_s1).astype(np.int64)
        limit = int(self.cfg["max_raw_pairs_per_chunk"])
        bounds, lo, acc = [], 0, 0
        for i, v in enumerate(per_s1):  # O(n_s1) scalar loop, cheap relative to blocking
            if acc + v > limit and i > lo:
                bounds.append((lo, i))
                lo, acc = i, 0
            acc += int(v)
        if lo < self.n_s1:
            bounds.append((lo, self.n_s1))
        self.stats["raw_pairs_total"] = int(per_s1.sum())
        return bounds

    def candidates(self, lo: int, hi: int) -> dict[str, np.ndarray]:
        """Deduplicated, top-k capped candidates for S1 local indices [lo, hi)."""
        s1_parts, pool_parts, w_parts, f_parts = [], [], [], []
        for p in self.passes:
            a, b = np.searchsorted(p.s1_rec, [lo, hi])
            rec, pos = p.s1_rec[a:b], p.s1_pos[a:b]
            cnt = p.counts[pos]
            s1_parts.append(np.repeat(rec, cnt))
            pool_parts.append(ragged_gather(p.starts[pos], cnt, p.pool_rec))
            w_parts.append(np.repeat(p.weight[pos], cnt))
            f_parts.append(np.full(int(cnt.sum()), p.bit, self.flag_dtype))
        s1 = np.concatenate(s1_parts) if s1_parts else np.zeros(0, np.int64)
        if len(s1) == 0:
            return _empty_candidates(self.flag_dtype)
        code = s1.astype(np.int64) * self.n_pool + np.concatenate(pool_parts).astype(np.int64)
        w = np.concatenate(w_parts)
        f = np.concatenate(f_parts)
        del s1, s1_parts, pool_parts, w_parts, f_parts
        order = np.argsort(code, kind="stable")
        code, w, f = code[order], w[order], f[order]
        uniq, first = np.unique(code, return_index=True)
        score = np.add.reduceat(w, first).astype(np.float32)
        flags = np.bitwise_or.reduceat(f, first)
        del code, w, f
        s1 = (uniq // self.n_pool).astype(np.int32)
        pool = (uniq % self.n_pool).astype(np.int32)
        order = np.lexsort((pool, -score, s1))
        s1, pool, score, flags = s1[order], pool[order], score[order], flags[order]
        rank = rank_within_groups(s1)
        keep = rank < int(self.cfg["top_k"])
        return {
            "s1_local": s1[keep],
            "pool_local": pool[keep],
            "cheap_score": score[keep],
            "cheap_rank": rank[keep].astype(np.int16),
            "pass_flags": flags[keep],
            "pairs_before_cap": np.int64(len(s1)),
        }


def _empty_candidates(flag_dtype=np.uint8) -> dict[str, np.ndarray]:
    return {
        "s1_local": np.zeros(0, np.int32),
        "pool_local": np.zeros(0, np.int32),
        "cheap_score": np.zeros(0, np.float32),
        "cheap_rank": np.zeros(0, np.int16),
        "pass_flags": np.zeros(0, flag_dtype),
        "pairs_before_cap": np.int64(0),
    }
