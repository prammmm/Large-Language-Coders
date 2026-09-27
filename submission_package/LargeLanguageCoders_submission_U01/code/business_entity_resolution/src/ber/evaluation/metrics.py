"""Official-definition entity-level evaluation (DEC-005).

Per Source-1 entity e with predicted set P, true set T, TP = |P & T|, k = |P|, n = |T|:
    n == 0:  score = 1.0 if k == 0 else 0.0            (singleton rule)
    n  > 0:  F0.5 = 1.25*TP / (0.25*n + k)              (0.0 when TP == 0 or k == 0)
which equals 1.25*P*R / (0.25*P + R). The headline number is the plain mean over ALL
evaluated S1 entities (macro).

Auxiliary definitions (documented so numbers are comparable across experiments):
    macro_precision  mean of TP/k over entities with k > 0 (singleton with k>0 -> 0)
    macro_recall     mean of TP/n over entities with n > 0 (k == 0 -> 0)
    blocking_recall  micro: sum |C & T| / sum |T|   (C = candidate set)
    oracle_f05       F0.5 of the oracle predicting exactly C & T (recall ceiling of blocking)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

BETA2 = 0.25


def f05_from_counts(tp, k, n) -> np.ndarray:
    """Vectorised per-entity F0.5 from integer arrays (official conventions)."""
    tp = np.asarray(tp, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    denom = BETA2 * n + k
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(denom > 0, (1 + BETA2) * tp / np.where(denom > 0, denom, 1.0), 0.0)
    singleton = n == 0
    f = np.where(singleton, np.where(k == 0, 1.0, 0.0), f)
    return f


def entity_f05(pred: Iterable[str], truth: Iterable[str]) -> float:
    p, t = set(pred), set(truth)
    return float(f05_from_counts(len(p & t), len(p), len(t)))


def match_bucket(n: int) -> str:
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 5:
        return "4-5"
    return "6+"


def _dist(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": int(x.max()),
    }


def _summary(tp, k, n, c_tp=None) -> dict:
    f = f05_from_counts(tp, k, n)
    out = {"entities": int(len(f)), "f05": float(f.mean()) if len(f) else float("nan")}
    has_pred = k > 0
    out["macro_precision"] = float((tp[has_pred] / k[has_pred]).mean()) if has_pred.any() else float("nan")
    pos = n > 0
    out["macro_recall"] = float((tp[pos] / n[pos]).mean()) if pos.any() else float("nan")
    out["singletons"] = int((~pos).sum())
    out["singleton_correct_empty"] = int(((~pos) & (k == 0)).sum())
    out["singleton_false_positive"] = int(((~pos) & (k > 0)).sum())
    out["empty_on_positive"] = int((pos & (k == 0)).sum())
    out["predicted_pairs"] = int(k.sum())
    out["true_pairs"] = int(n.sum())
    out["tp_pairs"] = int(tp.sum())
    out["false_merges"] = int(k.sum() - tp.sum())  # predicted pairs that are not true matches
    out["missed_true_matches"] = int(n.sum() - tp.sum())
    out["predicted_per_s1"] = float(k.mean()) if len(k) else float("nan")
    if c_tp is not None:
        out["blocking_recall"] = float(c_tp.sum() / n.sum()) if n.sum() else float("nan")
        out["oracle_f05"] = float(f05_from_counts(c_tp, c_tp, n).mean()) if len(f) else float("nan")
    return out


def evaluate(
    pred: Mapping[str, Iterable[str]],
    truth: Mapping[str, Iterable[str]],
    candidates: Mapping[str, Iterable[str]] | None = None,
    country: Mapping[str, str] | None = None,
) -> dict:
    """Evaluate over every S1 id in ``truth`` (missing predictions count as empty).

    Predictions for S1 ids not in ``truth`` are ignored (the scorer evaluates the
    held-out entity set). Returns overall metrics plus breakdowns by country and by
    number of true matches, and candidate-count statistics when ``candidates`` given.
    """
    ids = list(truth.keys())
    m = len(ids)
    tp = np.zeros(m, np.int64)
    k = np.zeros(m, np.int64)
    n = np.zeros(m, np.int64)
    c_tp = np.zeros(m, np.int64) if candidates is not None else None
    c_n = np.zeros(m, np.int64) if candidates is not None else None
    not_subset = 0
    for i, s1 in enumerate(ids):
        t = set(truth[s1])
        p = set(pred.get(s1, ()))
        tp[i] = len(p & t)
        k[i] = len(p)
        n[i] = len(t)
        if candidates is not None:
            c = set(candidates.get(s1, ()))
            c_tp[i] = len(c & t)
            c_n[i] = len(c)
            not_subset += bool(p - c)
    countries = np.array([country.get(s1, "?") for s1 in ids], dtype=object) if country is not None else None
    out = evaluate_arrays(tp, k, n, c_tp, c_n, countries)
    if candidates is not None:
        out["matches_not_in_candidates"] = int(not_subset)
    return out


def evaluate_arrays(tp, k, n, c_tp=None, c_n=None, countries=None) -> dict:
    """Same report as ``evaluate`` from per-entity count arrays (vectorised path)."""
    tp, k, n = (np.asarray(x, np.int64) for x in (tp, k, n))
    out = _summary(tp, k, n, c_tp)
    if c_n is not None:
        out["candidates_per_s1"] = _dist(np.asarray(c_n))
        out["candidate_pairs"] = int(np.asarray(c_n).sum())

    def breakdown(keys: np.ndarray) -> dict:
        res = {}
        for key in sorted(set(keys.tolist())):
            sel = keys == key
            res[key] = _summary(tp[sel], k[sel], n[sel], c_tp[sel] if c_tp is not None else None)
            if c_n is not None:
                res[key]["candidates_per_s1_mean"] = float(np.asarray(c_n)[sel].mean())
        return res

    out["by_match_count"] = breakdown(np.array([match_bucket(int(x)) for x in n], dtype=object))
    if countries is not None:
        out["by_country"] = breakdown(np.asarray(countries, dtype=object))
    return out


def threshold_sweep(s1_index: np.ndarray, is_true: np.ndarray, score: np.ndarray, n_true: np.ndarray, thresholds) -> list[dict]:
    """Vectorised F0.5 for many global thresholds.

    s1_index: per candidate pair, index into ``n_true`` (0..m-1)
    is_true:  per pair, whether the pair is a true match
    n_true:   per evaluated S1, number of true matches (INCLUDING ones blocking missed)
    """
    m = len(n_true)
    rows = []
    for t in thresholds:
        sel = score >= t
        k = np.bincount(s1_index[sel], minlength=m)
        tp = np.bincount(s1_index[sel & is_true], minlength=m)
        s = _summary(tp, k, n_true)
        s["threshold"] = float(t)
        rows.append(s)
    return rows
