from __future__ import annotations
import numpy as np

def _add_bernoulli(dist: np.ndarray, p: np.ndarray) -> np.ndarray:
    """dist (N, L) of a count; returns (N, L + 1) after adding one Bernoulli(p) (p: (N,))."""
    p = p[:, None]
    out = np.zeros((dist.shape[0], dist.shape[1] + 1))
    out[:, :-1] = dist * (1.0 - p)
    out[:, 1:] += dist * p
    return out

def _best_k(pa: np.ndarray, pr: np.ndarray, beta2: float) -> np.ndarray:
    N, m = pa.shape
    rest = np.ones((N, 1))
    for j in range(pr.shape[1]):
        rest = _add_bernoulli(rest, pr[:, j])
    B = [None] * (m + 1)
    B[m] = rest
    for k in range(m - 1, -1, -1):
        B[k] = _add_bernoulli(B[k + 1], pa[:, k])
    best_e = B[0][:, 0].copy()
    best_k = np.zeros(N, np.int64)
    pref = np.ones((N, 1))
    for k in range(1, m + 1):
        pref = _add_bernoulli(pref, pa[:, k - 1])
        i = np.arange(k + 1)[:, None]
        j = np.arange(B[k].shape[1])[None, :]
        W = (1.0 + beta2) * i / (beta2 * (i + j) + k)
        e = np.einsum('ni,nj,ij->n', pref, B[k], W)
        better = e >= best_e - 1e-12
        best_e = np.where(better, np.maximum(e, best_e), best_e)
        best_k = np.where(better, k, best_k)
    return best_k

def expected_f_prune(s1: np.ndarray, pool: np.ndarray, prob: np.ndarray, acc: np.ndarray, beta: float=0.5, chunk: int=200000) -> np.ndarray:
    s1 = np.asarray(s1, np.int64)
    pool = np.asarray(pool, np.int64)
    acc = np.asarray(acc, bool)
    keep = acc.copy()
    if not acc.any():
        return keep
    p = np.asarray(prob, np.float64).copy()
    taken = np.unique(pool[acc])
    pos = np.minimum(np.searchsorted(taken, pool), len(taken) - 1)
    p[~acc & (taken[pos] == pool)] = 0.0
    np.clip(p, 0.0, 1.0, out=p)
    s1_acc = np.unique(s1[acc])
    pos = np.minimum(np.searchsorted(s1_acc, s1), len(s1_acc) - 1)
    sel = np.flatnonzero(s1_acc[pos] == s1)
    o = sel[np.lexsort((pool[sel], -p[sel], ~acc[sel], s1[sel]))]
    so, ao = (s1[o], acc[o])
    starts = np.flatnonzero(np.r_[True, so[1:] != so[:-1]])
    n = np.diff(np.r_[starts, len(o)])
    m = np.add.reduceat(ao.astype(np.int64), starts)
    beta2 = beta * beta
    kstar = np.zeros(len(starts), np.int64)
    for mm in np.unique(m):
        for nn in np.unique(n[m == mm]):
            g = np.flatnonzero((m == mm) & (n == nn))
            for lo in range(0, len(g), chunk):
                gg = g[lo:lo + chunk]
                idx = starts[gg][:, None] + np.arange(nn)[None, :]
                P = p[o[idx]]
                kstar[gg] = _best_k(P[:, :mm], P[:, mm:], beta2)
    rank = np.arange(len(o)) - np.repeat(starts, n)
    drop = ao & (rank >= np.repeat(kstar, n))
    keep[o[drop]] = False
    return keep
