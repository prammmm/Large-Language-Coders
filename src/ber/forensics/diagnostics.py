"""POST-S-01 forensics helpers (pure, deterministic): decision flips, distribution shift (PSI),
summary statistics and cell-wise expected precision of label-free test decisions."""

from __future__ import annotations

import numpy as np

FLIP_NAMES = {0: "unchanged_0", 1: "added_0_to_1", 2: "removed_1_to_0", 3: "unchanged_1"}


def flip_codes(control: np.ndarray, new: np.ndarray) -> np.ndarray:
    """0 unchanged-0, 1 added (0->1), 2 removed (1->0), 3 unchanged-1."""
    c, n = np.asarray(control, bool), np.asarray(new, bool)
    return (np.where(~c & n, 1, 0) + np.where(c & ~n, 2, 0) + np.where(c & n, 3, 0)).astype(np.int8)


def flip_counts(codes: np.ndarray) -> dict[str, int]:
    b = np.bincount(np.asarray(codes, np.int64), minlength=4)
    return {FLIP_NAMES[i]: int(b[i]) for i in range(4)}


def summary(x: np.ndarray) -> dict:
    x = np.asarray(x, np.float64)
    miss = np.isnan(x)
    v = x[~miss]
    out = {"n": int(len(x)), "missing_rate": float(miss.mean()) if len(x) else None}
    if len(v) == 0:
        return out | {"mean": None}
    q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return out | {"mean": float(v.mean()), "p05": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]),
                  "p95": float(q[4]), "p99": float(q[5])}


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population stability index of ``cur`` against ``ref``; quantile bins of ref, NaN is its own bin.
    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift."""
    ref, cur = np.asarray(ref, np.float64), np.asarray(cur, np.float64)
    rv = ref[~np.isnan(ref)]
    if len(rv) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(rv, np.linspace(0, 1, bins + 1)[1:-1]))

    def dist(x):
        nan = np.isnan(x)
        b = np.searchsorted(edges, x[~nan], side="right")
        h = np.bincount(b, minlength=len(edges) + 1).astype(np.float64)
        h = np.append(h, nan.sum())
        return h / max(h.sum(), 1)

    p, q = dist(ref) + eps, dist(cur) + eps
    return float(np.sum((q - p) * np.log(q / p)))


def binned(x: np.ndarray, edges: list[float]) -> np.ndarray:
    """Bin index by right-open edges; NaN -> len(edges)+1 (its own bin)."""
    x = np.asarray(x, np.float64)
    out = np.searchsorted(np.asarray(edges, np.float64), np.nan_to_num(x, nan=-np.inf), side="right").astype(np.int64)
    out[np.isnan(x)] = len(edges) + 1
    return out


def cell_ids(*bins: np.ndarray) -> np.ndarray:
    """Combine several small non-negative integer bin arrays into one cell id."""
    out = np.zeros(len(bins[0]), np.int64)
    for b in bins:
        out = out * 64 + np.asarray(b, np.int64)
    return out


def expected_precision(train_cells: np.ndarray, train_true: np.ndarray, test_cells: np.ndarray, min_count: int = 50) -> dict:
    """If test decisions in each cell were as precise as the train decisions in that cell, what would their
    precision be? Cells with fewer than ``min_count`` train examples are reported as uncovered."""
    tc, tt = np.asarray(train_cells), np.asarray(train_true, bool)
    cells, inv = np.unique(tc, return_inverse=True)
    cnt = np.bincount(inv)
    prec = np.bincount(inv, weights=tt.astype(np.float64)) / np.maximum(cnt, 1)
    lut = dict(zip(cells.tolist(), zip(prec.tolist(), cnt.tolist())))
    exp, covered = 0.0, 0
    for c in np.asarray(test_cells).tolist():
        p = lut.get(c)
        if p is not None and p[1] >= min_count:
            exp += p[0]
            covered += 1
    n = len(test_cells)
    return {"test_decisions": int(n), "covered": int(covered), "coverage": covered / max(n, 1),
            "expected_precision_on_covered": exp / max(covered, 1), "train_decisions": int(len(tc)),
            "train_precision": float(tt.mean()) if len(tt) else None}


def floor_decision(control: np.ndarray, stage2: np.ndarray, stage1_score: np.ndarray, floor: float) -> np.ndarray:
    """Stage-2 decisions where ADDITIONS (accepted by stage 2 but not by the control) are only allowed when
    the stage-1 score is >= ``floor``; removals and agreements are unchanged. floor=0 -> stage 2 as is,
    floor > max score -> stage 2 can only remove pairs."""
    from ber.decision.stage2 import restrict_additions  # the production rule (EXP-S-02); identical semantics

    return restrict_additions(control, stage2, stage1_score, floor)
