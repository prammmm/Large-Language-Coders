"""Compact pairwise features (float32). Country-agnostic: no country / id features.

All similarity features are in [0, 1]; counts are raw counts. Missing text (empty
string on either side) yields 0 for that similarity and sets ``addr_any_empty``.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from ber.normalization.views import ADDRESS_STOPWORDS, LEGAL_FORMS

FEATURE_NAMES = (
    "name_norm_eq",
    "name_core_eq",
    "name_core_jaccard",
    "name_shared_tokens",
    "name_ratio",
    "name_token_set",
    "name_concat_jw",
    "name_len_diff",
    "addr_norm_eq",
    "addr_jaccard",
    "addr_ratio",
    "addr_token_set",
    "addr_num_jaccard",
    "addr_num_shared",
    "addr_shared_useful",
    "addr_any_empty",
)

# Optional features (EXP-F-01), each switched on by name via ``options["extra"]``.
EXTRA_FEATURES = ("num_conflict", "legal_form_conflict", "xlit_core_jaccard", "xlit_token_set")
XLIT_FEATURES = ("xlit_core_jaccard", "xlit_token_set")  # EXP-N-02: need the "name_xlit" column on both sides

# Legal-form canonicalisation for the conflict feature: synonyms share one label; the
# generic words co/company/private are ignored (they co-occur with many forms).
_LEGAL_CANON = {"limited": "ltd", "incorporated": "inc", "corporation": "corp"}
_LEGAL_IGNORE = {"co", "company", "private"}


def legal_forms(name_norm: str) -> set[str]:
    return {_LEGAL_CANON.get(t, t) for t in name_norm.split() if t in LEGAL_FORMS and t not in _LEGAL_IGNORE}


def _strip0(tok: str) -> str:
    """Leading zeros removed from a token starting with 0s followed by a digit."""
    if len(tok) > 1 and tok[0] == "0":
        t = tok.lstrip("0")
        return t if t and t[0].isdigit() else ("0" + t if t else "0")
    return tok


def strip_zeros_list(values: list[str]) -> list[str]:
    return [" ".join(_strip0(t) for t in v.split()) if v else v for v in values]


S1_COLUMNS = ("name_norm", "name_core", "name_concat", "addr_norm", "addr_nums", "addr_words")


def _take(table: pa.Table, col: str, idx: np.ndarray) -> pa.Array:
    return _combine(table.column(col).take(pa.array(idx, pa.int32())))


def _combine(a):
    return a.combine_chunks() if isinstance(a, pa.ChunkedArray) else a


def _eq(a: pa.Array, b: pa.Array) -> np.ndarray:
    both = pc.and_(pc.equal(a, b), pc.not_equal(a, ""))
    return both.to_numpy(zero_copy_only=False).astype(np.float32)


def _nonempty(a: pa.Array, b: pa.Array) -> np.ndarray:
    return pc.and_(pc.not_equal(a, ""), pc.not_equal(b, "")).to_numpy(zero_copy_only=False)


def _fuzz(a: list[str], b: list[str], scorer, workers: int, mask: np.ndarray) -> np.ndarray:
    out = process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)
    out = np.asarray(out, dtype=np.float32)
    if scorer is not JaroWinkler.normalized_similarity:
        out /= 100.0
    out[~mask] = 0.0
    return out


def _set_features(a: list[str], b: list[str], stop: frozenset | None = None):
    """Token Jaccard, shared-token count and (optionally) shared non-stopword count."""
    jac, shared, useful = [], [], []
    for x, y in zip(a, b):
        if not x or not y:
            jac.append(0.0)
            shared.append(0)
            useful.append(0)
            continue
        sx, sy = set(x.split()), set(y.split())
        inter = sx & sy
        ni = len(inter)
        jac.append(ni / (len(sx) + len(sy) - ni))
        shared.append(ni)
        if stop is not None:
            useful.append(sum(1 for t in inter if t not in stop and len(t) > 1))
    return (
        np.asarray(jac, np.float32),
        np.asarray(shared, np.float32),
        np.asarray(useful, np.float32) if stop is not None else None,
    )


def pair_features(
    s1: pa.Table, pool: pa.Table, s1_idx: np.ndarray, pool_idx: np.ndarray, workers: int = 1, options: dict | None = None
) -> dict[str, np.ndarray]:
    """Features for pairs (s1[s1_idx[i]], pool[pool_idx[i]]).

    options:
      strip_leading_zeros (bool)  compare address numbers after removing leading zeros
      extra (list[str])           optional features from EXTRA_FEATURES
    """
    options = options or {}
    extra = list(options.get("extra", []))
    unknown = set(extra) - set(EXTRA_FEATURES)
    if unknown:
        raise ValueError(f"unknown extra features: {sorted(unknown)}")
    cols = S1_COLUMNS + (("name_xlit",) if set(extra) & set(XLIT_FEATURES) else ())
    A = {c: _take(s1, c, s1_idx) for c in cols}
    B = {c: _take(pool, c, pool_idx) for c in cols}
    f: dict[str, np.ndarray] = {}

    f["name_norm_eq"] = _eq(A["name_norm"], B["name_norm"])
    f["name_core_eq"] = _eq(A["name_core"], B["name_core"])
    f["addr_norm_eq"] = _eq(A["addr_norm"], B["addr_norm"])

    name_ok = _nonempty(A["name_core"], B["name_core"])
    addr_ok = _nonempty(A["addr_norm"], B["addr_norm"])
    f["addr_any_empty"] = (~addr_ok).astype(np.float32)

    a_core, b_core = A["name_core"].to_pylist(), B["name_core"].to_pylist()
    f["name_ratio"] = _fuzz(a_core, b_core, fuzz.ratio, workers, name_ok)
    jac, shared, _ = _set_features(a_core, b_core)
    f["name_core_jaccard"], f["name_shared_tokens"] = jac, shared
    del a_core, b_core

    a_norm, b_norm = A["name_norm"].to_pylist(), B["name_norm"].to_pylist()
    f["name_token_set"] = _fuzz(a_norm, b_norm, fuzz.token_set_ratio, workers, name_ok)
    la = np.fromiter((len(x) for x in a_norm), np.float32, len(a_norm))
    lb = np.fromiter((len(x) for x in b_norm), np.float32, len(b_norm))
    f["name_len_diff"] = np.abs(la - lb) / np.maximum(np.maximum(la, lb), 1.0)
    del a_norm, b_norm

    f["name_concat_jw"] = _fuzz(
        A["name_concat"].to_pylist(), B["name_concat"].to_pylist(), JaroWinkler.normalized_similarity, workers, name_ok
    )

    a_addr, b_addr = A["addr_norm"].to_pylist(), B["addr_norm"].to_pylist()
    f["addr_ratio"] = _fuzz(a_addr, b_addr, fuzz.ratio, workers, addr_ok)
    f["addr_token_set"] = _fuzz(a_addr, b_addr, fuzz.token_set_ratio, workers, addr_ok)
    jac, _, useful = _set_features(a_addr, b_addr, ADDRESS_STOPWORDS)
    f["addr_jaccard"], f["addr_shared_useful"] = jac, useful
    del a_addr, b_addr

    a_nums, b_nums = A["addr_nums"].to_pylist(), B["addr_nums"].to_pylist()
    if options.get("strip_leading_zeros", False):
        a_nums, b_nums = strip_zeros_list(a_nums), strip_zeros_list(b_nums)
    jac, shared, _ = _set_features(a_nums, b_nums)
    f["addr_num_jaccard"], f["addr_num_shared"] = jac, shared

    if "num_conflict" in extra:
        # both sides have address numbers but share none (after the zero option above)
        both = np.fromiter((bool(x) and bool(y) for x, y in zip(a_nums, b_nums)), bool, len(a_nums))
        f["num_conflict"] = (both & (shared == 0)).astype(np.float32)
    if "legal_form_conflict" in extra:
        la_, lb_ = A["name_norm"].to_pylist(), B["name_norm"].to_pylist()
        vals = []
        for x, y in zip(la_, lb_):
            fx, fy = legal_forms(x), legal_forms(y)
            vals.append(1.0 if (fx and fy and not (fx & fy)) else 0.0)
        f["legal_form_conflict"] = np.asarray(vals, np.float32)

    if set(extra) & set(XLIT_FEATURES):
        # EXP-N-02: agreement of the train-learned cross-script name views (== name_core when unmapped)
        xa, xb = A["name_xlit"].to_pylist(), B["name_xlit"].to_pylist()
        x_ok = _nonempty(A["name_xlit"], B["name_xlit"])
        if "xlit_core_jaccard" in extra:
            f["xlit_core_jaccard"], _, _ = _set_features(xa, xb)
        if "xlit_token_set" in extra:
            f["xlit_token_set"] = _fuzz(xa, xb, fuzz.token_set_ratio, workers, x_ok)

    return {k: f[k] for k in (*FEATURE_NAMES, *extra)}
