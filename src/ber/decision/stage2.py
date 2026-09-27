"""EXP-S-01: collective second-stage decision model -- feature construction (pure functions).

Stage 2 re-decides every retrieved pair whose stage-1 score >= ``min_score`` using three groups of
label-free, country-agnostic features:

* stage-1 score;
* S1 context   -- where the pair sits among the S1's eligible candidates (rank, best, 2nd best,
                  gap, number of strong candidates, relative position);
* record context -- how the S1 compares with the OTHER S1 that claim the same S2/S3 record
                  (claimants, best / 2nd best claimant score, rank, gap to the best other claimant);
                  this lets the model learn what D-01 hard-codes;
* collective   -- similarity of the candidate RECORD to the S1's top-k OTHER candidate records
                  (the pair itself is always excluded from its own peer set), on name / cross-script
                  name / address / number / useful-address-word views, plus the peers' scores and
                  score-weighted aggregates.  "Does this record look like the records this S1 already
                  matches strongly?"

Every function is deterministic: ties are broken by row ids. Missing peers -> NaN (LightGBM native).
"""

from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz, process

from ber.features.pair_features import strip_zeros_list
from ber.normalization.views import ADDRESS_STOPWORDS

CONTEXT_FEATURES = [
    "score", "s1_rank", "s1_n_cand", "s1_rel_pos", "s1_best", "s1_second", "s1_gap_best", "s1_n_ge_05", "s1_n_ge_08",
    "rec_n_claim", "rec_n_claim_ge_05", "rec_best", "rec_second", "rec_rank", "rec_gap_best_other", "rec_is_best",
]
PEER_SIMS = ["name_jacc", "xlit_jacc", "name_tokset", "addr_tokset", "num_jacc", "word_jacc"]
VIEW_COLUMNS = ["name_core", "name_xlit", "addr_norm", "addr_nums", "addr_words"]


def peer_feature_names(k: int) -> list[str]:
    names = []
    for j in range(1, k + 1):
        names += [f"p{j}_{s}" for s in PEER_SIMS] + [f"p{j}_score"]
    return names + ["peer_n", "peer_max_sim", "peer_wmean_sim", "peer_best_sim"]


def feature_names(k: int) -> list[str]:
    return CONTEXT_FEATURES + peer_feature_names(k)


def _group_rank(g: np.ndarray, v: np.ndarray, tie: np.ndarray):
    """Order by (g asc, v desc, tie asc). Returns order, group start per element (in order space), rank."""
    o = np.lexsort((tie, -v.astype(np.float64), g))
    gs = g[o]
    first = np.ones(len(o), bool)
    first[1:] = gs[1:] != gs[:-1]
    start = np.flatnonzero(first)
    gid = np.cumsum(first) - 1
    rank = np.arange(len(o)) - start[gid]
    size = np.diff(np.append(start, len(o)))[gid]
    return o, start, gid, rank, size


def context_features(s1: np.ndarray, pool: np.ndarray, score: np.ndarray) -> dict[str, np.ndarray]:
    """S1- and record-level context for every eligible pair (inputs: all eligible pairs of a split)."""
    n = len(s1)
    out = {f: np.full(n, np.nan, np.float32) for f in CONTEXT_FEATURES}
    out["score"] = score.astype(np.float32)
    if n == 0:
        return out
    sc = score.astype(np.float32)
    # ---- S1 level (tie -> lower pool row first)
    o, start, gid, rank, size = _group_rank(s1, sc, pool)
    so = sc[o]
    best = so[start][gid]
    has2 = size > 1
    second = np.where(has2, so[np.minimum(start + 1, len(o) - 1)][gid], np.nan)
    ge05 = np.add.reduceat((so >= 0.5).astype(np.int64), start)[gid]
    ge08 = np.add.reduceat((so >= 0.8).astype(np.int64), start)[gid]
    vals = {"s1_rank": rank, "s1_n_cand": size, "s1_rel_pos": rank / np.maximum(size - 1, 1), "s1_best": best, "s1_second": second,
            "s1_gap_best": best - so, "s1_n_ge_05": ge05, "s1_n_ge_08": ge08}
    for k, v in vals.items():
        out[k][o] = np.asarray(v, np.float32)
    # ---- record level (tie -> lower S1 row first, as in D-01)
    o, start, gid, rank, size = _group_rank(pool, sc, s1)
    so = sc[o]
    best = so[start][gid]
    has2 = size > 1
    second = np.where(has2, so[np.minimum(start + 1, len(o) - 1)][gid], np.nan)
    ge05 = np.add.reduceat((so >= 0.5).astype(np.int64), start)[gid]
    best_other = np.where(rank == 0, second, best)
    vals = {"rec_n_claim": size, "rec_n_claim_ge_05": ge05, "rec_best": best, "rec_second": second, "rec_rank": rank,
            "rec_gap_best_other": so - best_other, "rec_is_best": (rank == 0).astype(np.float32)}
    for k, v in vals.items():
        out[k][o] = np.asarray(v, np.float32)
    return out


def peer_index(s1: np.ndarray, pool: np.ndarray, score: np.ndarray, k: int) -> list[np.ndarray]:
    """For every pair: indices (into the inputs) of the S1's k best OTHER eligible pairs, -1 if absent.
    Peers are ordered by stage-1 score (ties: lower pool row first); a pair is never its own peer."""
    n = len(s1)
    peers = [np.full(n, -1, np.int64) for _ in range(k)]
    if n == 0:
        return peers
    o, start, gid, rank, size = _group_rank(s1, score, pool)
    pos = np.arange(n)
    for j in range(k):
        # j-th best other = position j in the group if the pair ranks after j, else position j+1
        cand = start[gid] + np.where(rank > j, j, j + 1)
        ok = cand < start[gid] + size
        tgt = np.full(n, -1, np.int64)
        tgt[ok] = o[cand[ok]]
        peers[j][o[pos]] = tgt
    return peers


def _tok_jacc(a: list[str], b: list[str], useful: bool = False) -> np.ndarray:
    out = np.zeros(len(a), np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        if not x or not y:
            continue
        sx, sy = set(x.split()), set(y.split())
        if useful:
            sx = {t for t in sx if t not in ADDRESS_STOPWORDS and len(t) > 1}
            sy = {t for t in sy if t not in ADDRESS_STOPWORDS and len(t) > 1}
        u = len(sx | sy)
        out[i] = len(sx & sy) / u if u else 0.0
    return out


def _tokset(a: list[str], b: list[str], workers: int) -> np.ndarray:
    if not a:
        return np.zeros(0, np.float32)
    r = np.asarray(process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=workers, dtype=np.float32), np.float32) / 100.0
    empty = np.fromiter((not x or not y for x, y in zip(a, b)), bool, len(a))
    r[empty] = 0.0
    return r


def record_similarity(A: dict[str, list[str]], B: dict[str, list[str]], workers: int = 1) -> dict[str, np.ndarray]:
    """Record-to-record similarity (S2/S3 vs S2/S3) on aligned view lists."""
    na, nb = strip_zeros_list(A["addr_nums"]), strip_zeros_list(B["addr_nums"])
    return {"name_jacc": _tok_jacc(A["name_core"], B["name_core"]), "xlit_jacc": _tok_jacc(A["name_xlit"], B["name_xlit"]),
            "name_tokset": _tokset(A["name_xlit"], B["name_xlit"], workers), "addr_tokset": _tokset(A["addr_norm"], B["addr_norm"], workers),
            "num_jacc": _tok_jacc(na, nb), "word_jacc": _tok_jacc(A["addr_words"], B["addr_words"], useful=True)}


def peer_features(pool: np.ndarray, score: np.ndarray, peers: list[np.ndarray], views, workers: int = 1) -> dict[str, np.ndarray]:
    """``views(rows) -> {view: list[str]}`` for global pool rows (S2 rows, then S3 rows + n_s2)."""
    n = len(pool)
    k = len(peers)
    out: dict[str, np.ndarray] = {}
    self_views = views(pool)
    comb = []
    pscores = []
    for j, pj in enumerate(peers):
        has = pj >= 0
        idx = np.flatnonzero(has)
        sims = {s: np.full(n, np.nan, np.float32) for s in PEER_SIMS}
        ps = np.full(n, np.nan, np.float32)
        if len(idx):
            A = {c: [self_views[c][i] for i in idx] for c in VIEW_COLUMNS}
            other = views(pool[pj[idx]])
            r = record_similarity(A, other, workers)
            for s in PEER_SIMS:
                sims[s][idx] = r[s]
            ps[idx] = score[pj[idx]]
        for s in PEER_SIMS:
            out[f"p{j + 1}_{s}"] = sims[s]
        out[f"p{j + 1}_score"] = ps
        S = np.vstack([sims["xlit_jacc"], sims["name_tokset"], sims["addr_tokset"]])
        cnt = np.sum(~np.isnan(S), axis=0)
        tot = np.sum(np.where(np.isnan(S), 0.0, S), axis=0)
        comb.append(np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan).astype(np.float32))  # mean of the 3 views, NaN if no peer
        pscores.append(ps)
    C = np.vstack(comb) if comb else np.full((1, n), np.nan, np.float32)
    P = np.vstack(pscores) if pscores else np.full((1, n), np.nan, np.float32)
    npeer = np.sum(~np.isnan(P), axis=0).astype(np.float32)
    with np.errstate(all="ignore"):
        out["peer_n"] = npeer
        out["peer_max_sim"] = np.where(npeer > 0, np.nanmax(np.where(np.isnan(C), -np.inf, C), axis=0), np.nan).astype(np.float32)
        w = np.where(np.isnan(P), 0.0, P)
        cc = np.where(np.isnan(C), 0.0, C)
        out["peer_wmean_sim"] = np.where(w.sum(0) > 0, (w * cc).sum(0) / w.sum(0), np.nan).astype(np.float32)
        out["peer_best_sim"] = C[0].astype(np.float32)
    assert len(peer_feature_names(k)) == len(out)
    return out


# ----------------------------------------------------------------------------- decision restriction (EXP-S-02)
def restrict_additions(control: np.ndarray, stage2: np.ndarray, stage1_score: np.ndarray, add_floor: float) -> np.ndarray:
    """Final decisions when stage 2 may ADD a pair (accept it although the control rejects it) only if its
    stage-1 score >= ``add_floor``. Removals and agreements are unchanged. ``add_floor = inf`` makes stage 2
    a pure veto: final = control AND stage 2 (EXP-S-02). All inputs are aligned per pair."""
    c, s = np.asarray(control, bool), np.asarray(stage2, bool)
    if not (len(c) == len(s) == len(stage1_score)):
        raise ValueError("control, stage2 and stage1_score must be aligned")
    return (s & c) | (s & ~c & (np.asarray(stage1_score, np.float32) >= np.float32(add_floor)))
