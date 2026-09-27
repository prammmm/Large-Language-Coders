"""FINAL RESEARCH AUDIT helpers (diagnostic only: no production code path imports this module).

* per-pair overlap profile on the SAME views the blocker uses (name view = name_xlit, useful address
  words = addr_words minus stopwords/1-char, numbers = addr_nums with leading zeros stripped);
* hypothetical blocking-key families built with the blocker's own vectorised helpers, and an exact
  emitted-pair estimator (block cap applied, per-S1 union, overlap with existing candidates).
"""

from __future__ import annotations

import re

import numpy as np
import pyarrow as pa

from ber.blocking.blocker import cross_keys, encode_columns, ragged_gather, strip_leading_zeros
from ber.normalization.views import ADDRESS_STOPWORDS

_LZ = re.compile(r"(^|\s)0+(\d)")


# ----------------------------------------------------------------------------- per-pair profile
def script_class(s: str) -> str:
    letters = [ch for ch in (s or "") if ch.isalpha()]
    if not letters:
        return "empty"
    nl = sum(ord(ch) > 0x24F for ch in letters)
    return "latin" if nl == 0 else "nonlatin" if nl == len(letters) else "mixed"


def tokens(s: str | None) -> set[str]:
    return set((s or "").split())


def useful_words(s: str | None) -> set[str]:
    return {w for w in (s or "").split() if w not in ADDRESS_STOPWORDS and len(w) >= 2}


def numbers(s: str | None) -> set[str]:
    return set(_LZ.sub(r"\1\2", s or "").split())


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else float("nan")


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a or "", b or ""):
        if x != y:
            break
        n += 1
    return n


def pair_profile(s1: dict, rec: dict, name_df: dict | None = None, word_df: dict | None = None, rare_name: int = 50, rare_word: int = 20) -> dict:
    """``s1`` / ``rec``: dicts with name_norm, name_core, name_xlit, addr_norm, addr_nums, addr_words. DF dicts: pool DF."""
    name_df, word_df = name_df or {}, word_df or {}
    sn, rn = tokens(s1.get("name_norm")), tokens(rec.get("name_norm"))
    sc, rc = tokens(s1.get("name_core")), tokens(rec.get("name_core"))
    sx, rx = tokens(s1.get("name_xlit")), tokens(rec.get("name_xlit"))
    sw, rw = useful_words(s1.get("addr_words")), useful_words(rec.get("addr_words"))
    snum, rnum = numbers(s1.get("addr_nums")), numbers(rec.get("addr_nums"))
    sa, ra = tokens(s1.get("addr_norm")), tokens(rec.get("addr_norm"))
    shared_x = sx & rx
    shared_w = sw & rw
    sxs, rxs = s1.get("name_xlit") or "", rec.get("name_xlit") or ""
    return {
        "name_norm_shared": len(sn & rn), "name_core_shared": len(sc & rc), "xlit_shared": len(shared_x),
        "name_norm_jacc": jaccard(sn, rn), "name_core_jacc": jaccard(sc, rc), "xlit_jacc": jaccard(sx, rx),
        "xlit_equal": bool(sxs) and sxs == rxs, "xlit_sorted_equal": bool(sx) and sorted(sxs.split()) == sorted(rxs.split()),
        "same_tokens_other_order": bool(sx) and sx == rx and sxs != rxs,
        "rare_xlit_shared": sum(1 for t in shared_x if name_df.get(t, 0) <= rare_name),
        "core_equal_norm_differs(legal form)": bool(sc) and sc == rc and sn != rn,
        "xlit_prefix_len": common_prefix(sxs, rxs),
        "name_dup_tokens": len((s1.get("name_xlit") or "").split()) != len(sx) or len((rec.get("name_xlit") or "").split()) != len(rx),
        "s1_script": script_class(s1.get("name_core")), "rec_script": script_class(rec.get("name_core")),
        "script_mismatch": script_class(s1.get("name_core")) != script_class(rec.get("name_core")),
        "xlit_bridges_script": script_class(s1.get("name_core")) != script_class(rec.get("name_core")) and bool(shared_x),
        "s1_words": len(sw), "rec_words": len(rw), "shared_words": len(shared_w),
        "shared_rare_words": sum(1 for w in shared_w if word_df.get(w, 0) <= rare_word),
        "shared_common_words": sum(1 for w in shared_w if word_df.get(w, 0) > rare_word),
        "addr_jacc": jaccard(sa, ra), "shared_number": bool(snum & rnum), "s1_has_number": bool(snum), "rec_has_number": bool(rnum),
        "s1_addr_empty": not (s1.get("addr_norm") or ""), "rec_addr_empty": not (rec.get("addr_norm") or ""),
        "s1_name_empty": not sx, "rec_name_empty": not rx,
        "name_only": bool(shared_x) and not shared_w and not (snum & rnum),
        "address_only": not shared_x and bool(shared_w or (snum & rnum)),
        "number_only": bool(snum & rnum) and not shared_w and not shared_x,
        "nothing_shared": not shared_x and not shared_w and not (snum & rnum),
    }


# ----------------------------------------------------------------------------- hypothetical key families
FAMILIES = ("tok_x_num", "tok_pair", "tok_x_word", "word_pair", "num_x_word", "single_tok", "single_word")


def encode_views(s1: pa.Table, pool: pa.Table, chunk_rows: int = 250_000) -> dict:
    """Integer-encode the three blocker views over S1 (records 0..n_s1-1) then pool (n_s1..)."""
    out = {"n_s1": s1.num_rows, "n_pool": pool.num_rows}
    for name, col, tf in (("tok", "name_xlit", None), ("word", "addr_words", None), ("num", "addr_nums", strip_leading_zeros)):
        rec, tok, vocab = encode_columns([s1.column(col), pool.column(col)], True, chunk_rows, tf)
        if name == "word":
            stop = pa.array(sorted(ADDRESS_STOPWORDS), pa.string())
            bad = (np.asarray(pa.compute.is_in(vocab, value_set=stop), bool) | (np.asarray(pa.compute.utf8_length(vocab)) < 2)) if len(vocab) else np.zeros(0, bool)
            keep = ~bad[tok] if len(tok) else np.zeros(0, bool)
            rec, tok = rec[keep], tok[keep]
        df = np.bincount(tok[rec >= out["n_s1"]], minlength=len(vocab)) if len(vocab) else np.zeros(0, np.int64)
        out[name] = (rec.astype(np.int64), tok.astype(np.int64), len(vocab), df)
    return out


def family_keys(enc: dict, family: str, k_a: int = 2, k_b: int = 2, max_df: int = 50, window: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """Distinct (record, key) arrays of a key family (keys are family-local int64 codes)."""
    T, W, N = enc["tok"], enc["word"], enc["num"]
    if family == "tok_x_num":
        r, k = cross_keys(T[0], T[1], T[3], k_a, N[0], N[1], N[3], k_b, max(N[2], 1), window)
    elif family == "tok_pair":
        r, k = cross_keys(T[0], T[1], T[3], k_a, T[0], T[1], T[3], k_a, max(T[2], 1), window, same=True)
    elif family == "tok_x_word":
        r, k = cross_keys(T[0], T[1], T[3], k_a, W[0], W[1], W[3], k_b, max(W[2], 1), window)
    elif family == "word_pair":
        r, k = cross_keys(W[0], W[1], W[3], k_a, W[0], W[1], W[3], k_a, max(W[2], 1), window, same=True)
    elif family == "num_x_word":
        r, k = cross_keys(N[0], N[1], N[3], k_a, W[0], W[1], W[3], k_b, max(W[2], 1), window)
    elif family == "single_tok":
        sel = T[3][T[1]] <= max_df
        r, k = T[0][sel], T[1][sel]
    elif family == "single_word":
        sel = W[3][W[1]] <= max_df
        r, k = W[0][sel], W[1][sel]
    else:
        raise ValueError(family)
    code = np.unique(np.asarray(r, np.int64) * (int(k.max()) + 1 if len(k) else 1) + np.asarray(k, np.int64)) if len(r) else np.zeros(0, np.int64)
    base = int(k.max()) + 1 if len(k) else 1
    return code // base, code % base


def emit_pairs(rec: np.ndarray, key: np.ndarray, n_s1: int, cap: int, s1_mask: np.ndarray | None = None):
    """Pairs a key family would emit: S1 record x pool records sharing a key whose POOL block size <= cap
    (the blocker drops larger keys). Returns (s1_rec, pool_rec) unique pairs (pool_rec relative to the pool)
    and per-key stats. ``s1_mask`` (bool over S1 records) restricts the emitting S1."""
    is_pool = rec >= n_s1
    pk, pr = key[is_pool], rec[is_pool] - n_s1
    o = np.lexsort((pr, pk))
    pk, pr = pk[o], pr[o]
    uk, start, cnt = np.unique(pk, return_index=True, return_counts=True)
    sk, sr = key[~is_pool], rec[~is_pool]
    if s1_mask is not None:
        m = s1_mask[sr]
        sk, sr = sk[m], sr[m]
    pos = np.searchsorted(uk, sk)
    pos_c = np.minimum(pos, max(len(uk) - 1, 0))
    hit = (len(uk) > 0) & (uk[pos_c] == sk) if len(sk) else np.zeros(0, bool)
    ok = hit & (cnt[pos_c] <= cap)
    over = hit & (cnt[pos_c] > cap)
    c = cnt[pos_c[ok]]
    s_out = np.repeat(sr[ok], c)
    p_out = ragged_gather(start[pos_c[ok]], c, pr) if ok.any() else np.zeros(0, np.int64)
    code = np.unique(s_out.astype(np.int64) * (1 << 32) + p_out.astype(np.int64)) if len(s_out) else np.zeros(0, np.int64)
    stats = {"s1_key_rows": int(len(sk)), "keys_hit": int(hit.sum()), "keys_over_cap": int(over.sum()),
             "gross_pairs": int(c.sum()), "unique_pairs": int(len(code)),
             "block_size_of_used_keys": {q: float(np.quantile(c, q)) for q in (0.5, 0.9, 0.99)} if len(c) else None}
    return code >> 32, code & ((1 << 32) - 1), stats


# ----------------------------------------------------------------------------- oracle gains
def oracle_gain(tp, k, n_true, add_true=None, remove_false=None) -> float:
    """F0.5 gain (macro per S1) if ``add_true`` true pairs were added and ``remove_false`` false pairs removed per S1."""
    from ber.forensics.shadow import weighted_metrics
    tp, k, n_true = (np.asarray(x, np.float64) for x in (tp, k, n_true))
    a = np.zeros(len(tp)) if add_true is None else np.asarray(add_true, np.float64)
    r = np.zeros(len(tp)) if remove_false is None else np.asarray(remove_false, np.float64)
    return weighted_metrics(tp + a, k + a - r, n_true)["f05"] - weighted_metrics(tp, k, n_true)["f05"]


def cardinality_bin(n_true) -> np.ndarray:
    """0 = singleton, 1, 2, 3 = 3-5, 4 = 6+ true matches."""
    n = np.asarray(n_true)
    return np.select([n == 0, n == 1, n == 2, n <= 5], [0, 1, 2, 3], 4)


def pool_key_index(rec: np.ndarray, key: np.ndarray, n_s1: int):
    """Sorted unique pool keys with their POOL block size (distinct pool records), plus the S1 (record, key) rows."""
    is_pool = rec >= n_s1
    uk, cnt = np.unique(key[is_pool], return_counts=True)
    return uk, cnt, rec[~is_pool], key[~is_pool]


def gross_pairs(uk, cnt, s_rec, s_key, cap: int) -> tuple[np.ndarray, int]:
    """Per S1 row: pairs its keys would emit (sum of pool block sizes of keys with size <= cap), without materialising."""
    pos = np.minimum(np.searchsorted(uk, s_key), max(len(uk) - 1, 0))
    hit = (uk[pos] == s_key) if len(uk) else np.zeros(len(s_key), bool)
    ok = hit & (cnt[pos] <= cap)
    return np.bincount(s_rec[ok], weights=cnt[pos[ok]]).astype(np.int64), int((hit & (cnt[pos] > cap)).sum())


def share_key(rec: np.ndarray, key: np.ndarray, n_s1: int, pairs_s: np.ndarray, pairs_p: np.ndarray, cap: int | None = None) -> np.ndarray:
    """For given (S1 record, pool record) pairs (pool record relative to the pool): do they share a key (whose pool block
    size is <= cap when ``cap`` is given)? Nothing is emitted, so this is safe for unlimited block sizes."""
    uk, cnt, _, _ = pool_key_index(rec, key, n_s1)
    big = uk[cnt > cap] if cap is not None else np.zeros(0, key.dtype)
    s_set = np.unique(np.asarray(pairs_s, np.int64))
    p_set = np.unique(np.asarray(pairs_p, np.int64) + n_s1)
    ms, mp = np.isin(rec, s_set), np.isin(rec, p_set)
    sk = np.unique(rec[ms].astype(np.int64) * (1 << 40) + key[ms])  # (record, key) of the involved S1
    pk = np.unique(rec[mp].astype(np.int64) * (1 << 40) + key[mp])
    if len(big):
        sk = sk[~np.isin(sk % (1 << 40), big)]
    # join on key: for every pool (record, key) find S1 records with the same key
    s_by_key = {}
    for r, k in zip((sk >> 40).tolist(), (sk % (1 << 40)).tolist()):
        s_by_key.setdefault(k, set()).add(r)
    linked = set()
    for r, k in zip((pk >> 40).tolist(), (pk % (1 << 40)).tolist()):
        for s in s_by_key.get(k, ()):
            linked.add((s, r - n_s1))
    return np.array([(int(a), int(b)) in linked for a, b in zip(pairs_s, pairs_p)], bool)
