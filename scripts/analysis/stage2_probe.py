"""PROBE (diagnostic, not production): does a context-aware second-stage decision model beat
"global threshold + D-01"?

Uses an arm's all-train scored parts (all 2.2M train S1 compete). Stage-1 scores of every non-fit
S1 are out-of-sample. Pairs with stage-1 score < --min-score are never accepted (they are ~never
true) and are only used to build context.
Per-pair context features (country-agnostic, label-free):
  score; S1 level: rank of the pair among the S1's candidates, gap to the S1's best, the S1's 2nd
  best, number of the S1's candidates >= 0.5; pool-record level: number of S1 claiming the record
  with score >= 0.5, the record's best score, gap to it, this S1's rank on the record, 2nd best.
Train LightGBM on E half A minus validation; choose the probability threshold on that set
(entity-level F0.5); evaluate on held-out E half B and the frozen validation fold, against the
production decision (threshold t + D-01) on the same S1.

    python scripts/analysis/stage2_probe.py --config configs/exp_b07_name_x_addr.yaml --tag forensic_alltrain_B07 --t 0.67
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from eval_exp_d01 import half_of  # noqa: E402
from full_competition import SHIFT, gt_codes  # noqa: E402

from ber.config import Paths, load_config  # noqa: E402
from ber.data.split import load_val_ids  # noqa: E402
from ber.decision.competition import best_claimant_keep  # noqa: E402
from ber.evaluation.metrics import evaluate_arrays  # noqa: E402
from ber.pipeline.run_baseline import _ids, _read_parts, fit_s1_subset  # noqa: E402
from ber.utils.runtime import LOG, configure_arrow, setup_logging, write_json  # noqa: E402


def group_stats(g, v):
    """Per element: rank of v within its group (desc), group max, group 2nd max. g any int labels."""
    o = np.lexsort((-v, g))
    gs, vs = g[o], v[o]
    first = np.ones(len(o), bool)
    first[1:] = gs[1:] != gs[:-1]
    start = np.flatnonzero(first)
    gid = np.cumsum(first) - 1
    rank = np.arange(len(o)) - start[gid]
    mx = vs[start][gid]
    second_pos = start + 1
    has2 = np.append(start[1:], len(o)) - start > 1
    sec = np.where(has2, vs[np.minimum(second_pos, len(o) - 1)], -1.0)[gid]
    out_rank, out_max, out_sec = (np.empty(len(o), np.float32) for _ in range(3))
    out_rank[o], out_max[o], out_sec[o] = rank, mx, sec
    return out_rank, out_max, out_sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--t", type=float, required=True, help="production threshold of this arm")
    ap.add_argument("--min-score", type=float, default=0.02)
    ap.add_argument("--out", default=None)
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
    E = ~fit
    pops = {"train(E_A-minus-val)": E & (half == 0) & ~val, "heldout_E_B": E & (half == 1), "heldout_validation": val}

    d = _read_parts(paths.scored_dir("train", a.tag), ["s1_row", "pool_row", "score"])
    s1, pool, sc = d["s1_row"].astype(np.int64), d["pool_row"].astype(np.int64), d["score"].astype(np.float32)
    del d
    # production decision on ALL pairs (reference)
    prod = best_claimant_keep(pool, s1, sc, sc >= np.float32(a.t))
    k_prod = np.bincount(s1[prod], minlength=n)
    keep = sc >= np.float32(a.min_score)
    s1, pool, sc, prod = s1[keep], pool[keep], sc[keep], prod[keep]
    true = np.isin(s1 * SHIFT + pool, gtc)
    tp_prod = np.bincount(s1[prod & true], minlength=n)
    LOG.info("pairs kept (score >= %.2f): %d", a.min_score, len(s1))
    r1, m1, s1b = group_stats(s1, sc)
    hi = (sc >= 0.5).astype(np.int64)
    n_hi_s1 = np.bincount(s1, weights=hi, minlength=n)[s1].astype(np.float32)
    r2, m2, s2b = group_stats(pool, sc)
    n_hi_pool = np.bincount(pool, weights=hi, minlength=int(pool.max()) + 1)[pool].astype(np.float32)
    X = np.column_stack([sc, r1, m1 - sc, s1b, n_hi_s1, r2, m2 - sc, s2b, n_hi_pool, (r2 == 0).astype(np.float32)]).astype(np.float32)
    names = ["score", "s1_rank", "s1_gap_to_best", "s1_second", "s1_n_ge_0.5", "pool_rank", "pool_gap_to_best", "pool_second",
             "pool_claimants_ge_0.5", "is_pool_best"]
    del r1, m1, s1b, r2, m2, s2b, hi
    tr = pops["train(E_A-minus-val)"][s1]
    params = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 1.0,
              "bagging_fraction": 0.8, "bagging_freq": 1, "seed": 42, "deterministic": True, "force_row_wise": True, "verbosity": -1, "num_threads": 8}
    ds = lgb.Dataset(X[tr], true[tr].astype(np.float32), feature_name=names, free_raw_data=True)
    bst = lgb.train(params, ds, num_boost_round=300)
    p = bst.predict(X, num_threads=8).astype(np.float32)
    rep = {"config": a.config, "tag": a.tag, "production_threshold": a.t, "min_score": a.min_score, "pairs": int(len(s1)),
           "feature_importance_gain": dict(zip(names, (bst.feature_importance("gain") / bst.feature_importance("gain").sum()).round(4).tolist()))}
    table = {}
    for t2 in np.round(np.arange(0.30, 0.81, 0.02), 2):
        acc = p >= t2
        k = np.bincount(s1[acc], minlength=n)
        tp = np.bincount(s1[acc & true], minlength=n)
        table[f"{t2:.2f}"] = {nm: evaluate_arrays(tp[m], k[m], n_true[m])["f05"] for nm, m in pops.items()}
    best = max(table, key=lambda t2: table[t2]["train(E_A-minus-val)"])
    acc = p >= float(best)
    k = np.bincount(s1[acc], minlength=n)
    tp = np.bincount(s1[acc & true], minlength=n)
    res = {}
    for nm, m in pops.items():
        base = evaluate_arrays(tp_prod[m], k_prod[m], n_true[m])
        st2 = evaluate_arrays(tp[m], k[m], n_true[m])
        res[nm] = {"production_t+D01": {x: base[x] for x in ("f05", "macro_precision", "macro_recall")},
                   "stage2": {x: st2[x] for x in ("f05", "macro_precision", "macro_recall")}, "delta_f05": st2["f05"] - base["f05"]}
    rep.update({"stage2_threshold_chosen_on_train": float(best), "results": res, "table": table})
    out = Path(a.out or paths.run_dir(cfg["experiment"]["id"]) / "stage2_probe.json")
    write_json(out, rep)
    for nm, v in res.items():
        LOG.info("%-22s prod %.4f (P %.4f R %.4f) -> stage2 %.4f (P %.4f R %.4f)  delta %+.4f", nm, v["production_t+D01"]["f05"],
                 v["production_t+D01"]["macro_precision"], v["production_t+D01"]["macro_recall"], v["stage2"]["f05"],
                 v["stage2"]["macro_precision"], v["stage2"]["macro_recall"], v["delta_f05"])
    LOG.info("importance %s", rep["feature_importance_gain"])


if __name__ == "__main__":
    main()
