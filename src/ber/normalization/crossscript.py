"""EXP-N-02: cross-script name view learned ONLY from provided train true-match pairs.

No external dictionary, transliteration package, pretrained model or internet resource is used,
and no language / country rule is hard-coded: scripts are detected from Unicode letters, and every
correspondence comes from co-occurrence in labelled TRAIN pairs.

Learning (``learn_token_map``)
  For each true pair (S1 name_core, pool name_core):
    L = distinct Latin tokens of the S1 name, N = distinct non-Latin tokens of the pool name.
  For every (n, l) in N x L the pair contributes to ``support(n, l)`` -- counted once per distinct
  S1 entity, so one entity with many pool records (or one noisy pair) cannot create a mapping.
  ``occurrences(n)`` = distinct S1 entities whose pairs contain n (with a non-empty L).
  For each n: best l = highest support (ties -> ambiguous). Accepted iff
    support >= min_support, support / occurrences(n) >= min_confidence, and the best support is
    strictly larger than the runner-up (otherwise "ambiguous", rejected).
  Word order, extra tokens (legal forms are already removed from name_core), duplicates and
  multi-token names are handled because correspondences are counted over token SETS across many
  entities, not by position.

Applying (``apply_map``): every token found in the map is replaced by its Latin correspondence;
all other tokens (Latin, unseen, unmapped) are kept unchanged; duplicate tokens are removed
(order preserved). Names without non-ASCII characters are returned untouched (fast path).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

DEFAULT_PARAMS = {"min_support": 3, "min_confidence": 0.5}


def _latin_letter(ch: str) -> bool:
    o = ord(ch)
    return ch.isascii() or 0x00C0 <= o <= 0x024F or 0x1E00 <= o <= 0x1EFF


def token_script(tok: str) -> str:
    """'latin' (all letters Latin), 'nonlatin' (at least one non-Latin letter) or 'none' (no letters)."""
    has_letter = False
    for ch in tok:
        if ch.isalpha():
            has_letter = True
            if not _latin_letter(ch):
                return "nonlatin"
    return "latin" if has_letter else "none"


def pair_tokens(s1_name: str, pool_name: str) -> tuple[set[str], set[str]]:
    """(Latin tokens of the S1 name, non-Latin tokens of the pool name) as sets."""
    lat = {t for t in (s1_name or "").split() if token_script(t) == "latin"}
    non = {t for t in (pool_name or "").split() if token_script(t) == "nonlatin"}
    return lat, non


def learn_token_map(pairs, min_support: int = 3, min_confidence: float = 0.5) -> tuple[dict[str, str], dict]:
    """``pairs``: iterable of (s1_key, s1_name_core, pool_name_core). Returns (map, stats)."""
    seen_nl: set[tuple] = set()
    seen_n: set[tuple] = set()
    support: dict[str, Counter] = defaultdict(Counter)
    occ: Counter = Counter()
    n_pairs = n_used = 0
    for key, a, b in pairs:
        n_pairs += 1
        lat, non = pair_tokens(a, b)
        if not lat or not non:
            continue
        n_used += 1
        for n in non:
            if (key, n) not in seen_n:
                seen_n.add((key, n))
                occ[n] += 1
            for l_ in lat:
                if (key, n, l_) not in seen_nl:
                    seen_nl.add((key, n, l_))
                    support[n][l_] += 1
    mapping: dict[str, str] = {}
    rej = Counter()
    sup_hist, conf_vals = Counter(), []
    for n in sorted(support):
        ranked = sorted(support[n].items(), key=lambda kv: (-kv[1], kv[0]))
        best_l, best = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0
        conf = best / occ[n]
        if best < min_support:
            rej["low_support"] += 1
        elif conf < min_confidence:
            rej["low_confidence"] += 1
        elif best == second:
            rej["ambiguous_tie"] += 1
        else:
            mapping[n] = best_l
            sup_hist[min(best, 50)] += 1
            conf_vals.append(conf)
    cv = np.asarray(conf_vals) if conf_vals else np.zeros(1)
    stats = {"pairs_seen": n_pairs, "pairs_usable_latin_x_nonlatin": n_used, "nonlatin_token_types": len(support),
             "mappings": len(mapping), "rejected": dict(rej),
             "support_hist_capped50": {str(k): v for k, v in sorted(sup_hist.items())},
             "confidence_quantiles": {q: float(np.quantile(cv, float(q))) for q in ("0.05", "0.25", "0.5", "0.75", "0.95")},
             "params": {"min_support": min_support, "min_confidence": min_confidence}}
    return mapping, stats


def learn_legal_tokens(pairs, mapped, legal_forms, min_support: int = 20, min_confidence: float = 0.8,
                       max_content_confidence: float = 0.5) -> tuple[set[str], dict]:
    """EXP-N-03 stage 2: non-Latin tokens that are LEGAL-FORM words (e.g. an Indic "private limited").

    ``pairs``: (s1_key, S1 name_norm (legal forms kept), pool name). Only non-Latin tokens NOT mapped
    by stage 1 are considered. Counting is per distinct S1 entity, as in stage 1. A token r is a
    legal-form token iff
      legal(r) >= min_support          entities whose S1 name carries a legal form, and
      legal(r) / occ(r) >= min_confidence, and
      max over Latin content tokens c of support(r, c) / occ(r) < max_content_confidence
    -- the last condition keeps real name words (always paired with the same content word, even if
    every such business is "... Pvt Ltd") out. Accepted tokens are DROPPED from the view, exactly as
    ``name_views`` drops Latin legal forms from name_core."""
    occ, leg = Counter(), Counter()
    content: dict[str, Counter] = defaultdict(Counter)
    seen_r, seen_rc = set(), set()
    for key, a, b in pairs:
        non = {t for t in (b or "").split() if token_script(t) == "nonlatin" and t not in mapped}
        if not non:
            continue
        lat = {t for t in (a or "").split() if token_script(t) == "latin"}
        has_legal = bool(lat & legal_forms)
        contents = lat - legal_forms
        for r in non:
            if (key, r) in seen_r:
                continue
            seen_r.add((key, r))
            occ[r] += 1
            leg[r] += has_legal
            for c in contents:
                if (key, r, c) not in seen_rc:
                    seen_rc.add((key, r, c))
                    content[r][c] += 1
    accepted, rej = set(), Counter()
    for r in sorted(occ):
        max_c = max(content[r].values(), default=0) / occ[r]
        if leg[r] < min_support:
            rej["low_support"] += 1
        elif leg[r] / occ[r] < min_confidence:
            rej["low_confidence"] += 1
        elif max_c >= max_content_confidence:
            rej["explained_by_content_word"] += 1
        else:
            accepted.add(r)
    stats = {"residual_nonlatin_token_types": len(occ), "legal_tokens": len(accepted), "rejected": dict(rej),
             "params": {"min_support": min_support, "min_confidence": min_confidence, "max_content_confidence": max_content_confidence}}
    return accepted, stats


DROP = ""  # map target meaning "remove this token from the view" (EXP-N-03 legal-form tokens)


def apply_map_one(name: str, mapping: dict[str, str]) -> str:
    if not name or name.isascii():
        return name
    out, seen = [], set()
    for t in name.split():
        m = mapping.get(t, t)
        if m == DROP or m in seen:
            continue
        seen.add(m)
        out.append(m)
    return " ".join(out) if out else name  # like name_views: never produce an empty name


def apply_map(names, mapping: dict[str, str]):
    """Map a list / pyarrow array of name_core strings. Returns the same kind (list or pa.Array)."""
    if isinstance(names, (pa.Array, pa.ChunkedArray)):
        arr = names.combine_chunks() if isinstance(names, pa.ChunkedArray) else names
        if not mapping or len(arr) == 0:
            return arr
        ascii_ = pc.string_is_ascii(pc.fill_null(arr, "")).to_numpy(zero_copy_only=False)
        idx = np.flatnonzero(~ascii_)
        if len(idx) == 0:
            return arr
        vals = arr.to_pylist()
        for i in idx.tolist():
            vals[i] = apply_map_one(vals[i], mapping)
        return pa.array(vals, pa.string())
    return [apply_map_one(x, mapping) for x in names]


class LeakageError(RuntimeError):
    pass


class XlitMap:
    """A learned token map plus provenance (which S1 entities it was learned from)."""

    def __init__(self, mapping: dict[str, str], meta: dict):
        self.mapping, self.meta = mapping, meta

    @staticmethod
    def ids_digest(ids) -> str:
        h = hashlib.sha256()
        for s in sorted(ids):
            h.update(s.encode())
            h.update(b"\n")
        return h.hexdigest()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"meta": self.meta, "map": self.mapping}, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "XlitMap":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["map"], d["meta"])

    def apply(self, names):
        return apply_map(names, self.mapping)
