"""Shadow benchmark (POST-S-01 forensics): held-out labelled train S1 that never enter any training,
map learning, threshold / early-stopping / rule selection, used to judge public-transfer risk.

Definitions (all deterministic, seed 42):
  SHADOW-BASE          E half B  AND  u("shadow") < 0.5         (E = train S1 not in the stage-1 fit set)
  GATE-B (the rest)    E half B  AND  NOT shadow                 (keeps an ordinary held-out gate set)
  ORPHANED competition  u("orphan") < orphan_frac over ALL train S1: those S1 are removed from the
                        competition, so their true S2/S3 records stay in the pool as unclaimed "orphan
                        entities" -- the observed test condition (denser pool, same confident matches per S1)
  SHADOW-TEST-MATCHED  shadow S1 not orphaned, evaluated in the orphaned competition, re-weighted so that
                        (country, eligible-candidate-count bin) matches the labelled test countries
  SHADOW-HARD          shadow S1 with at least one measured hard property (see hard_flags)
where u(salt) = uniform[0,1) from sha1(f"{salt}|{seed}|{s1_id}").
"""

from __future__ import annotations

import hashlib

import numpy as np

from ber.evaluation.metrics import f05_from_counts

CAND_BINS = [(1, 3), (4, 5), (6, 7), (8, 10), (11, 10**9)]


def u01(ids: list[str], salt: str, seed: int = 42) -> np.ndarray:
    return np.array([int(hashlib.sha1(f"{salt}|{seed}|{v}".encode()).hexdigest()[:12], 16) / float(16**12) for v in ids])


def shadow_split(E_half_B: np.ndarray, ids: list[str], seed: int = 42, frac: float = 0.5,
                 exclude: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """``exclude``: S1 that may never be shadow (e.g. the frozen validation fold, used for past model selection)."""
    u = u01(ids, "shadow", seed)
    pool = E_half_B & (~exclude if exclude is not None else True)
    shadow = pool & (u < frac)
    return {"shadow_base": shadow, "gate_B": E_half_B & ~shadow}


def chain_orphan_mask(ids: list[str], name_key: np.ndarray, protect: np.ndarray, frac: float = 0.5, seed: int = 42) -> np.ndarray:
    """Remove from the competition a fraction of 'chain' S1 (S1 whose name key is shared by >= 2 S1), never the
    ``protect``ed ones. Their records stay in the pool as uncontested LOOK-ALIKE records for the kept siblings."""
    keys, inv, cnt = np.unique(np.asarray(name_key, dtype=object).astype(str), return_inverse=True, return_counts=True)
    chain = cnt[inv] >= 2
    return chain & ~protect & (u01(ids, "chain_orphan", seed) < frac)


def orphan_mask(ids: list[str], frac: float, seed: int = 42) -> np.ndarray:
    return u01(ids, "orphan", seed) < frac


def orphan_fraction(train_pool: int, train_s1: int, test_pool: int, test_s1: int) -> float:
    """Fraction of train S1 to remove so that train pool records per remaining S1 == test's."""
    target_s1 = train_pool / (test_pool / test_s1)
    return float(max(0.0, 1.0 - target_s1 / train_s1))


def cand_bin(n: np.ndarray) -> np.ndarray:
    out = np.full(len(n), -1, np.int8)
    for i, (lo, hi) in enumerate(CAND_BINS):
        out[(n >= lo) & (n <= hi)] = i
    return out


def match_weights(src_cells: np.ndarray, target_share: dict[int, float]) -> np.ndarray:
    """Per-S1 weights so the weighted share of each cell equals ``target_share`` (cells absent in the
    source get no mass; the remaining shares are renormalised). Mean weight = 1."""
    w = np.zeros(len(src_cells), np.float64)
    cells, counts = np.unique(src_cells, return_counts=True)
    avail = {int(c): n for c, n in zip(cells, counts)}
    tot = sum(target_share.get(c, 0.0) for c in avail)
    if tot <= 0:
        return np.ones(len(src_cells))
    for c, n in avail.items():
        w[src_cells == c] = target_share.get(c, 0.0) / tot / n
    return w * len(src_cells)


def weighted_metrics(tp: np.ndarray, k: np.ndarray, n: np.ndarray, w: np.ndarray | None = None) -> dict:
    """Entity-level macro F0.5 / P / R with optional per-S1 weights (same conventions as evaluate_arrays)."""
    tp, k, n = (np.asarray(x, np.float64) for x in (tp, k, n))
    w = np.ones(len(tp)) if w is None else np.asarray(w, np.float64)
    f = f05_from_counts(tp, k, n)
    has = k > 0
    pos = n > 0
    return {"f05": float((w * f).sum() / w.sum()),
            "precision": float((w[has] * (tp[has] / k[has])).sum() / max(w[has].sum(), 1e-12)),
            "recall": float((w[pos] * (tp[pos] / n[pos])).sum() / max(w[pos].sum(), 1e-12)),
            "false_merges_per_1k_s1": float(1000 * (w * (k - tp)).sum() / w.sum()),
            "singleton_fp_rate": float((w[~pos] * (k[~pos] > 0)).sum() / max(w[~pos].sum(), 1e-12)),
            "s1": int(len(tp)), "weight_sum": float(w.sum())}


def leakage_checks(shadow: np.ndarray, **others: np.ndarray) -> dict[str, int]:
    """Number of shadow S1 inside each named partition; every value must be 0."""
    return {f"shadow_in_{k}": int((shadow & v).sum()) for k, v in others.items()}


# ----------------------------------------------------------------------------- v2 (EXP-S-02): locked sets + density tiers
def assert_disjoint(**masks: np.ndarray) -> None:
    """Raise if any two named S1 masks overlap."""
    names = list(masks)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            k = int((np.asarray(masks[a], bool) & np.asarray(masks[b], bool)).sum())
            if k:
                raise AssertionError(f"{a} and {b} overlap in {k} S1")


def locked_split(shadow: np.ndarray, ids: list[str], seed: int = 42) -> dict[str, np.ndarray]:
    """SHADOW-BASE -> SHADOW-DEV (50%, may be used while iterating), SHADOW-LOCKED-A / -B (25% each; evaluated
    only after an experiment is frozen). Disjoint by construction (asserted); union = SHADOW-BASE."""
    u = u01(ids, "locked", seed)
    shadow = np.asarray(shadow, bool)
    out = {"dev": shadow & (u < 0.5), "locked_A": shadow & (u >= 0.5) & (u < 0.75), "locked_B": shadow & (u >= 0.75)}
    assert_disjoint(**out)
    if not ((out["dev"] | out["locked_A"] | out["locked_B"]) == shadow).all():
        raise AssertionError("locked split does not cover SHADOW-BASE")
    return out


def tier_orphan_mask(ids: list[str], chain_member: np.ndarray, protect: np.ndarray, q: float, seed: int = 42) -> np.ndarray:
    """One-parameter family of competitions (q in [0, 2]): S1 removed from the competition (records stay in the
    pool, unclaimed). Chain members first (q <= 1: those with u < q), then the other S1 (q > 1: u < q - 1).
    Protected (evaluated) S1 are never removed. Nested: q1 <= q2 -> mask(q1) is a subset of mask(q2)."""
    u = u01(ids, "tier_orphan", seed)
    chain_member, protect = np.asarray(chain_member, bool), np.asarray(protect, bool)
    return (chain_member & ~protect & (u < min(q, 1.0))) | (~chain_member & ~protect & (u < max(q - 1.0, 0.0)))


def lookalike_pairs(s1: np.ndarray, pool: np.ndarray, score: np.ndarray, peer_max_sim: np.ndarray, drop: np.ndarray | None = None) -> np.ndarray:
    """Label-free look-alike candidates under a competition (``drop`` = S1 removed): stage-1 score < 0.5, at most
    one remaining claimant among eligible pairs (= the feature ``rec_n_claim`` <= 1), peer similarity >= 0.8.
    Peer similarity and the score depend only on the S1's own candidates, so they do not change with ``drop``."""
    s1, pool = np.asarray(s1, np.int64), np.asarray(pool, np.int64)
    keep = np.ones(len(s1), bool) if drop is None else ~np.asarray(drop, bool)[s1]
    claim = np.bincount(pool[keep], minlength=int(pool.max()) + 1 if len(pool) else 0)
    return keep & (np.asarray(score) < 0.5) & (claim[pool] <= 1) & (np.asarray(peer_max_sim) >= 0.8)
