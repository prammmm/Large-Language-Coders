"""PROBE (diagnostic, not production): collective / cluster-consistency evidence for the decision.

Idea: an S1's true records resemble EACH OTHER. For every candidate we measure its similarity to
the S1's best OTHER candidate record (record-to-record name Jaccard, name token-set ratio, address
token-set ratio, number overlap) and add it to the context features of stage2_probe.py.
Runs on a deterministic 20% sample of train S1 rows (row % 5 == 0) to stay light; context features
are computed on ALL pairs (full competition) before sampling.
Train on sample & (E half A minus validation); evaluate on held-out sample & E half B and
sample & validation, against production (t + D-01) and the context-only stage 2 on the SAME S1.

    python scripts/analysis/collective_probe.py --config configs/exp_b07_name_x_addr.yaml --tag forensic_alltrain_B07 --t 0.67
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from eval_exp_d01 import half_of  # noqa: E402
from full_competition import SHIFT, gt_codes  # noqa: E402
from stage2_probe import group_stats  # noqa: E402

from ber.config import Paths, load_config  # noqa: E402
from ber.data.normalize_records import load_manifest  # noqa: E402
from ber.data.split import load_val_ids  # noqa: E402
from ber.decision.competition import best_claimant_keep  # noqa: E402
from ber.evaluation.metrics import evaluate_arrays  # noqa: E402
from ber.pipeline.run_baseline import _ids, _read_parts, fit_s1_subset  # noqa: E402
from ber.utils.runtime import LOG, configure_arrow, setup_logging, write_json  # noqa: E402


def pool_views(paths, rows, n2):
    """name_core, addr_norm, addr_nums for global pool rows (S2 rows, then S3 rows + n2)."""
    out_n, out_a, out_u = {}, {}, {}
    man = load_manifest(paths, "train")
    for src, off in (("source2", 0), ("source3", n2)):
        want = pa.array((np.unique(rows[(rows >= off) & (rows < off + (n2 if off == 0 else 10**9))]) - off).astype(np.int32))
        for k, meta in man[src]["countries"].items():
            t = pq.read_table(paths.processed_dir("train") / src / meta["file"], columns=["row", "name_core", "addr_norm", "addr_nums"])
            t = t.filter(pc.is_in(t.column("row"), value_set=want))
            for r, a_, b_, c_ in zip(t.column("row").to_pylist(), t.column("name_core").to_pylist(),
                                     t.column("addr_norm").to_pylist(), t.column("addr_nums").to_pylist()):
                out_n[r + off], out_a[r + off], out_u[r + off] = a_ or "", b_ or "", c_ or ""
    return out_n, out_a, out_u


def jac(a, b):
    x, y = set(a.split()), set(b.split())
    return len(x & y) / len(x | y) if x and y else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--min-score", type=float, default=0.02)
    a = ap.parse_args()
    setup_logging()
    configure_arrow()
    cfg = load_config(a.config)
    paths = Paths(cfg)
    gtc, n_true, n = gt_codes(paths)
    ids = _ids(paths, "train", "source1")
    ix = lambda v: pc.index_in(pa.array(v, pa.string()), value_set=ids).to_numpy(zero_copy_only=False).astype(np.int64)  # noqa: E731
    fit = np.zeros(n, bool)
    fit[ix(fit_s1_subset(cfg, paths))] = True
    val = np.zeros(n, bool)
    val[ix(load_val_ids(paths))] = True
    half = half_of(ids.to_pylist())
    samp = (np.arange(n) % 5) == 0
    E = ~fit
    pops = {"train": samp & E & (half == 0) & ~val, "heldout_E_B": samp & E & (half == 1), "heldout_validation": samp & val}

    d = _read_parts(paths.scored_dir("train", a.tag), ["s1_row", "pool_row", "score"])
    s1, pool, sc = d["s1_row"].astype(np.int64), d["pool_row"].astype(np.int64), d["score"].astype(np.float32)
    del d
    prod = best_claimant_keep(pool, s1, sc, sc >= np.float32(a.t))
    k_prod = np.bincount(s1[prod], minlength=n)
    keep = sc >= np.float32(a.min_score)
    s1, pool, sc, prod = s1[keep], pool[keep], sc[keep], prod[keep]
    true = np.isin(s1 * SHIFT + pool, gtc)
    tp_prod = np.bincount(s1[prod & true], minlength=n)
    # context features on ALL pairs (full competition)
    r1, m1, s1b = group_stats(s1, sc)
    hi = (sc >= 0.5).astype(np.int64)
    n_hi_s1 = np.bincount(s1, weights=hi, minlength=n)[s1].astype(np.float32)
    r2, m2, s2b = group_stats(pool, sc)
    n_hi_pool = np.bincount(pool, weights=hi, minlength=int(pool.max()) + 1)[pool].astype(np.float32)
    ctx = np.column_stack([sc, r1, m1 - sc, s1b, n_hi_s1, r2, m2 - sc, s2b, n_hi_pool]).astype(np.float32)
    # ---- sample
    sel = samp[s1]
    s1, pool, sc, true, ctx, r1 = s1[sel], pool[sel], sc[sel], true[sel], ctx[sel], r1[sel]
    LOG.info("sample pairs: %d", len(s1))
    # best OTHER candidate record per pair: top1 of the S1 unless the pair is the top1 -> top2
    o = np.lexsort((-sc, s1))
    gs = s1[o]
    first = np.ones(len(o), bool)
    first[1:] = gs[1:] != gs[:-1]
    start = np.flatnonzero(first)
    gid = np.cumsum(first) - 1
    top1 = pool[o][start][gid]
    size = np.diff(np.append(start, len(o)))[gid]
    top2 = np.where(size > 1, pool[o][np.minimum(start + 1, len(o) - 1)][gid], -1)
    is_top = (np.arange(len(o)) - start[gid]) == 0
    ref = np.where(is_top, top2, top1)
    ref_all = np.empty(len(o), np.int64)
    ref_all[o] = ref
    n2 = int(load_manifest(paths, "train")["source2"]["rows"])
    need = np.unique(np.concatenate([pool, ref_all[ref_all >= 0]]))
    vn, va, vu = pool_views(paths, need, n2)
    A_n = [vn.get(int(p), "") for p in pool]
    B_n = [vn.get(int(r), "") if r >= 0 else "" for r in ref_all]
    A_a = [va.get(int(p), "") for p in pool]
    B_a = [va.get(int(r), "") if r >= 0 else "" for r in ref_all]
    A_u = [vu.get(int(p), "") for p in pool]
    B_u = [vu.get(int(r), "") if r >= 0 else "" for r in ref_all]
    del vn, va, vu
    c_name_j = np.fromiter((jac(x, y) for x, y in zip(A_n, B_n)), np.float32, len(A_n))
    c_name_ts = np.asarray(process.cpdist(A_n, B_n, scorer=fuzz.token_set_ratio, workers=8), np.float32) / 100
    c_addr_ts = np.asarray(process.cpdist(A_a, B_a, scorer=fuzz.token_set_ratio, workers=8), np.float32) / 100
    c_num_j = np.fromiter((jac(x, y) for x, y in zip(A_u, B_u)), np.float32, len(A_u))
    ref_score = np.full(len(s1), -1.0, np.float32)
    ref_score[o] = np.where(is_top, np.where(size > 1, sc[o][np.minimum(start + 1, len(o) - 1)][gid], -1.0), sc[o][start][gid])
    coll = np.column_stack([c_name_j, c_name_ts, c_addr_ts, c_num_j, ref_score, (ref_all < 0).astype(np.float32)]).astype(np.float32)
    del A_n, B_n, A_a, B_a, A_u, B_u
    params = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 200, "bagging_fraction": 0.8,
              "bagging_freq": 1, "seed": 42, "deterministic": True, "force_row_wise": True, "verbosity": -1, "num_threads": 8}
    tr = pops["train"][s1]
    res = {}
    for label, X in (("context_only", ctx), ("context+collective", np.hstack([ctx, coll]))):
        bst = lgb.train(params, lgb.Dataset(X[tr], true[tr].astype(np.float32)), num_boost_round=300)
        p = bst.predict(X, num_threads=8).astype(np.float32)
        best, best_f = None, -1
        for t2 in np.round(np.arange(0.30, 0.81, 0.02), 2):
            acc = p >= t2
            f = evaluate_arrays(np.bincount(s1[acc & true], minlength=n)[pops["train"]], np.bincount(s1[acc], minlength=n)[pops["train"]], n_true[pops["train"]])["f05"]
            if f > best_f:
                best, best_f = t2, f
        acc = p >= best
        k = np.bincount(s1[acc], minlength=n)
        tp = np.bincount(s1[acc & true], minlength=n)
        res[label] = {"threshold_on_train": float(best)}
        for nm, m in pops.items():
            r = evaluate_arrays(tp[m], k[m], n_true[m])
            b = evaluate_arrays(tp_prod[m], k_prod[m], n_true[m])
            res[label][nm] = {"f05": r["f05"], "P": r["macro_precision"], "R": r["macro_recall"], "production_f05": b["f05"],
                              "production_P": b["macro_precision"], "delta": r["f05"] - b["f05"]}
            LOG.info("%-20s %-20s prod %.4f (P %.4f) -> %.4f (P %.4f R %.4f)  delta %+.4f", label, nm, b["f05"], b["macro_precision"],
                     r["f05"], r["macro_precision"], r["macro_recall"], r["f05"] - b["f05"])
        if label != "context_only":
            g = bst.feature_importance("gain")
            res[label]["importance"] = dict(zip(["score", "s1_rank", "s1_gap", "s1_second", "s1_n_hi", "pool_rank", "pool_gap", "pool_second",
                                                 "pool_n_hi", "coll_name_jacc", "coll_name_tokset", "coll_addr_tokset", "coll_num_jacc",
                                                 "ref_score", "no_ref"], (g / g.sum()).round(4).tolist()))
            LOG.info("importance %s", res[label]["importance"])
    write_json(paths.run_dir(cfg["experiment"]["id"]) / "collective_probe.json", res)


if __name__ == "__main__":
    main()
