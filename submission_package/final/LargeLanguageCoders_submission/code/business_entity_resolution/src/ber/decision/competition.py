from __future__ import annotations
import numpy as np

def _claim_order(pool_row: np.ndarray, s1_row: np.ndarray, score: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """``idx`` sorted by pool record, then score descending, then s1_row ascending."""
    return idx[np.lexsort((s1_row[idx], -score[idx].astype(np.float64), pool_row[idx]))]

def best_claimant_keep(pool_row, s1_row, score, pred, margin: float=0.0) -> np.ndarray:
    """Boolean mask over all pairs: ``pred`` with the losing claims of each pool record removed."""
    pool_row, s1_row = (np.asarray(pool_row), np.asarray(s1_row))
    score, pred = (np.asarray(score), np.asarray(pred, dtype=bool))
    keep = pred.copy()
    idx = np.flatnonzero(pred)
    if len(idx) < 2:
        return keep
    o = _claim_order(pool_row, s1_row, score, idx)
    pr = pool_row[o]
    first = np.ones(len(o), bool)
    first[1:] = pr[1:] != pr[:-1]
    loser = o[~first]
    if margin > 0:
        start = np.flatnonzero(first)
        winner_score = score[o][start][np.cumsum(first) - 1][~first]
        loser = loser[winner_score.astype(np.float64) - score[loser].astype(np.float64) >= margin]
    keep[loser] = False
    return keep

def claim_stats(pool_row, s1_row, pred) -> dict:
    """Conflict statistics of a prediction set: records claimed by >= 2 distinct S1."""
    pool_row, pred = (np.asarray(pool_row), np.asarray(pred, dtype=bool))
    pr = pool_row[pred]
    if len(pr) == 0:
        return {'predicted_pairs': 0, 'claimed_records': 0, 'conflicted_records': 0, 'pairs_on_conflicted_records': 0, 'losers_lower_bound': 0, 'conflict_size': {'2': 0, '3': 0, '4': 0, '5+': 0}}
    _, cnt = np.unique(pr, return_counts=True)
    c = cnt[cnt >= 2]
    return {'predicted_pairs': int(len(pr)), 'claimed_records': int(len(cnt)), 'conflicted_records': int(len(c)), 'pairs_on_conflicted_records': int(c.sum()), 'losers_lower_bound': int((c - 1).sum()), 'conflict_size': {'2': int((c == 2).sum()), '3': int((c == 3).sum()), '4': int((c == 4).sum()), '5+': int((c >= 5).sum())}}

def claimants_per_pair(pool_row, pred) -> np.ndarray:
    """For every pair: number of predicted claims on its pool record (0 for non-predicted pairs)."""
    pool_row, pred = (np.asarray(pool_row), np.asarray(pred, dtype=bool))
    out = np.zeros(len(pool_row), np.int32)
    if pred.any():
        u, inv, cnt = np.unique(pool_row[pred], return_inverse=True, return_counts=True)
        out[pred] = cnt[inv]
    return out

def winner_score_per_pair(pool_row, s1_row, score, pred) -> np.ndarray:
    """For every predicted pair: the best claimant's score on its pool record (NaN otherwise)."""
    pool_row, s1_row, score = (np.asarray(pool_row), np.asarray(s1_row), np.asarray(score))
    out = np.full(len(pool_row), np.nan, np.float32)
    idx = np.flatnonzero(np.asarray(pred, dtype=bool))
    if len(idx) == 0:
        return out
    o = _claim_order(pool_row, s1_row, score, idx)
    pr = pool_row[o]
    first = np.ones(len(o), bool)
    first[1:] = pr[1:] != pr[:-1]
    out[o] = score[o][np.flatnonzero(first)][np.cumsum(first) - 1]
    return out
