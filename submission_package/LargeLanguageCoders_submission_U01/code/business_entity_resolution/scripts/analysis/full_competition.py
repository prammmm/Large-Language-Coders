"""Full-competition analysis (READ-ONLY w.r.t. production; forensic diagnostic).

Why: the frozen validation fold contains only 10% of train S1s, so a pool record's other
claimants are mostly absent there. On test ALL S1s compete. This script reproduces test-like
competition on TRAIN, where labels exist:

1. score ALL train S1s with the given model/blocker (same code path as `infer`; features not
   stored) -> data/processed/train/<scored_tag>/  (skipped if it already exists);
2. evaluation population E = every train S1 NOT used to fit the model (the fit's 200k
   train-fold S1s are in-sample and excluded; they still act as COMPETITORS);
3. compare, on E (and on the frozen validation subset):
     unrestricted predictions (the production rule)
     vs. "best claimant only": drop a predicted pair if another S1 claims the same pool record
         with a higher score (optionally only when the winner leads by >= margin);
   plus conflict statistics comparable to the test-side ones.
Sanity check: the unrestricted validation-subset F0.5 must equal the production validation value.

    python scripts/analysis/full_competition.py --config configs/exp_b04_blocking_v2.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ber.config import Paths, load_config
from ber.data.normalize_records import load_manifest, read_country
from ber.data.split import load_val_ids
from ber.evaluation.metrics import evaluate_arrays
from ber.models import make_scorer
from ber.pipeline.run_baseline import _ids, _read_parts, fit_s1_subset, pool_ids, score_split
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, read_json, setup_logging, stage, write_json

SHIFT = np.int64(1) << np.int64(24)


def gt_codes(paths):
    """Ground truth as int codes s1_row*2^24+pool_row, and n_true per train S1 row."""
    s1_ids, p_ids = _ids(paths, "train", "source1"), pool_ids(paths, "train")
    gt = pq.read_table(paths.interim_file("train", "ground_truth"))
    L = pc.split_pattern(gt.column("matched_entity_ids"), ",")
    flat, par = pc.list_flatten(L), pc.list_parent_indices(L)
    keep = pc.not_equal(flat, "")
    s = pc.index_in(pc.take(gt.column("source1_entity_id"), pc.filter(par, keep)), value_set=s1_ids).to_numpy(zero_copy_only=False).astype(np.int64)
    p = pc.index_in(pc.filter(flat, keep), value_set=p_ids).to_numpy(zero_copy_only=False).astype(np.int64)
    n_true = np.bincount(s, minlength=len(s1_ids))
    return np.sort(s * SHIFT + p), n_true, len(s1_ids)


def summarise(rows_mask_s1, s1_row, pred, true, n_true, country_of_row, n_rows):
    """Entity-level report over the S1 rows selected by rows_mask_s1 (bool over all S1 rows)."""
    sel_pairs = rows_mask_s1[s1_row]
    k = np.bincount(s1_row[sel_pairs & pred], minlength=n_rows)
    tp = np.bincount(s1_row[sel_pairs & pred & true], minlength=n_rows)
    ids = np.flatnonzero(rows_mask_s1)
    r = evaluate_arrays(tp[ids], k[ids], n_true[ids], countries=country_of_row[ids])
    keep = ("entities", "f05", "macro_precision", "macro_recall", "false_merges", "singleton_false_positive", "tp_pairs", "predicted_pairs")
    return {**{x: r[x] for x in keep}, "by_country": {c: {x: v[x] for x in keep} for c, v in r["by_country"].items()},
            "by_match_count": {b: {x: v[x] for x in ("entities", "f05", "macro_precision", "macro_recall")} for b, v in r["by_match_count"].items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_b04_blocking_v2.yaml")
    ap.add_argument("--scored-tag", default="forensic_alltrain_B04")
    ap.add_argument("--out", default="experiments/runs/POST-0.91-FORENSIC/full_competition.json")
    a = ap.parse_args()
    setup_logging()
    configure_arrow()
    cfg = load_config(a.config)
    paths = Paths(cfg)
    exp = cfg["experiment"]["id"]
    scorer = make_scorer(cfg, paths)
    t = float(json.loads((paths.run_dir(exp) / "model.json").read_text(encoding="utf-8"))["meta"]["threshold"])
    out_dir = paths.scored_dir("train", a.scored_tag)
    rep = {"config": a.config, "model": str(paths.run_dir(exp) / "model.json"), "threshold": t, "scored_dir": str(out_dir)}

    # ---------------- 1. score ALL train S1 (same code path as `infer`)
    if not any(out_dir.glob("*.parquet")):
        with stage("score ALL train S1"), PeakMemory() as pm:
            stats = score_split(cfg, paths, "train", None, out_dir, scorer, store_features=False)
        rep["scoring"] = {"candidate_pairs": stats["candidate_pairs"], "s1": stats["evaluated_s1"],
                          "runtime_seconds": round(pm.seconds, 1), "peak_rss_gb": round(pm.peak_gb, 2)}
    else:
        rep["scoring"] = {"reused": True}
        LOG.info("re-using existing scored parts in %s", out_dir)

    # ---------------- 2. load, label
    d = _read_parts(out_dir, ["s1_row", "pool_row", "score", "cheap_rank"])
    s1_row, pool_row, score = d["s1_row"].astype(np.int64), d["pool_row"].astype(np.int64), d["score"].astype(np.float32)
    del d
    gtc, n_true, n_rows = gt_codes(paths)
    code = s1_row * SHIFT + pool_row
    true = np.isin(code, gtc, assume_unique=False)
    del code
    pred = score >= np.float32(t)
    country_of_row = np.empty(n_rows, dtype=object)
    for k in load_manifest(paths, "train")["source1"]["countries"]:
        country_of_row[read_country(paths, "train", "source1", k, ["row"]).column("row").to_numpy()] = k
    s1_ids = _ids(paths, "train", "source1")
    fit_rows = pc.index_in(pa.array(fit_s1_subset(cfg, paths), pa.string()), value_set=s1_ids).to_numpy(zero_copy_only=False).astype(np.int64)
    val_rows = pc.index_in(pa.array(load_val_ids(paths), pa.string()), value_set=s1_ids).to_numpy(zero_copy_only=False).astype(np.int64)
    is_fit = np.zeros(n_rows, bool)
    is_fit[fit_rows] = True
    is_val = np.zeros(n_rows, bool)
    is_val[val_rows] = True
    E = ~is_fit
    rep["populations"] = {"all_train_s1": int(n_rows), "fit_s1_in_sample": int(is_fit.sum()), "E_out_of_sample": int(E.sum()),
                          "validation_subset": int(is_val.sum()), "pairs": int(len(s1_row)), "predicted_pairs": int(pred.sum())}

    # ---------------- 3. conflict statistics (all S1 compete, as on test)
    pp = np.flatnonzero(pred)
    cnt = np.bincount(pool_row[pp], minlength=int(pool_row.max()) + 1)
    conflicted = cnt[pool_row[pp]] >= 2
    order = pp[np.lexsort((-score[pp], pool_row[pp]))]
    first = np.ones(len(order), bool)
    first[1:] = pool_row[order][1:] != pool_row[order][:-1]
    loser = np.zeros(len(s1_row), bool)
    loser[order[~first]] = True
    winner_score = np.zeros(int(pool_row.max()) + 1, np.float32)
    winner_score[pool_row[order[first]]] = score[order[first]]
    stats_c = {}
    for c in sorted(set(country_of_row.tolist())):
        cs = country_of_row[s1_row[pp]] == c
        lp = loser[pp] & cs
        stats_c[c] = {"predicted": int(cs.sum()), "conflicted_pairs_share": float(conflicted[cs].mean()),
                      "certainly_wrong_share": float(lp.sum() / max(cs.sum(), 1)),
                      "loser_pairs_that_are_actually_true": int((lp & true[pp]).sum()), "loser_pairs": int(lp.sum()),
                      "winner_pairs_on_conflicted_records_true_share": float(true[pp][cs & conflicted & ~loser[pp]].mean()) if (cs & conflicted & ~loser[pp]).any() else None}
    claims = cnt[cnt >= 2]
    rep["train_conflicts_all_s1_compete"] = {"by_country": stats_c,
                                             "claims_per_conflicted_record": {str(k): int((claims == k).sum()) for k in (2, 3, 4)} | {"5+": int((claims >= 5).sum())}}

    # ---------------- 4. unrestricted vs best-claimant-only (with margins), on E and on validation
    variants = {}
    base = {"E": summarise(E, s1_row, pred, true, n_true, country_of_row, n_rows),
            "validation": summarise(is_val, s1_row, pred, true, n_true, country_of_row, n_rows)}
    variants["unrestricted"] = base
    margin_of_loser = winner_score[pool_row] - score  # >0 for losers
    for m in (0.0, 0.05, 0.1, 0.2, 0.3):
        drop = loser & (margin_of_loser >= np.float32(m))
        p2 = pred & ~drop
        variants[f"best_claimant_margin_{m:.2f}"] = {"E": summarise(E, s1_row, p2, true, n_true, country_of_row, n_rows),
                                                     "validation": summarise(is_val, s1_row, p2, true, n_true, country_of_row, n_rows),
                                                     "dropped_pairs_in_E": int((drop & E[s1_row]).sum()),
                                                     "dropped_true_pairs_in_E": int((drop & E[s1_row] & true).sum())}
    # sensitivity: competitors restricted to out-of-sample S1 (exclude in-sample fit S1 as competitors)
    pp2 = np.flatnonzero(pred & ~is_fit[s1_row])
    o2 = pp2[np.lexsort((-score[pp2], pool_row[pp2]))]
    f2 = np.ones(len(o2), bool)
    f2[1:] = pool_row[o2][1:] != pool_row[o2][:-1]
    loser2 = np.zeros(len(s1_row), bool)
    loser2[o2[~f2]] = True
    variants["best_claimant_margin_0.00_oos_competitors_only"] = {"E": summarise(E, s1_row, pred & ~loser2, true, n_true, country_of_row, n_rows)}
    rep["variants"] = variants
    prod = read_json(paths.run_dir(exp) / "metrics.json")["at_threshold_used"]["f05"]
    rep["sanity_validation_unrestricted_equals_production"] = {"production": prod, "here": base["validation"]["f05"],
                                                               "equal": abs(prod - base["validation"]["f05"]) < 1e-9}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(a.out), rep)
    for name, v in variants.items():
        e = v["E"]
        LOG.info("%-45s E: F0.5 %.4f P %.4f R %.4f FM %d | %s", name, e["f05"], e["macro_precision"], e["macro_recall"], e["false_merges"],
                 " ".join(f"{c} {x['f05']:.4f}" for c, x in e["by_country"].items()))
    LOG.info("sanity: %s", rep["sanity_validation_unrestricted_equals_production"])


if __name__ == "__main__":
    main()
