"""EXP-D-01 evaluation: best-claimant resolution under FULL competition (all train S1 compete).

Re-uses the EXP-B-04 scores of ALL 2,206,821 train S1 produced by
scripts/analysis/full_competition.py (data/processed/train/forensic_alltrain_B04, same model,
blocker, features and code path as `infer`). Nothing is fitted here: model and threshold come
from experiments/runs/EXP-B-04/model.json; the rule has no parameter (margin 0).

Populations
  E           all train S1 NOT in the 200k fit set (out-of-sample; primary)
  validation  the frozen validation S1 (subset of E) + its seeded halves A/B
  E halves    same sha1 hash rule as scripts/analysis/compare_val_halves.py
The 200k fit S1 are in-sample and only act as competitors (sensitivity: excluded as competitors).

The decision rule used is the production function ``ber.decision.competition.best_claimant_keep``.

    python scripts/experiments/eval_exp_d01.py --config configs/exp_d01_best_claimant.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from full_competition import SHIFT, gt_codes  # noqa: E402

from ber.config import Paths, load_config  # noqa: E402
from ber.data.normalize_records import load_manifest, read_country  # noqa: E402
from ber.data.split import load_val_ids  # noqa: E402
from ber.decision.competition import best_claimant_keep, claim_stats, claimants_per_pair, winner_score_per_pair  # noqa: E402
from ber.evaluation.metrics import evaluate_arrays  # noqa: E402
from ber.pipeline.run_baseline import _ids, _read_parts, fit_s1_subset, scored_dir  # noqa: E402
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, read_json, setup_logging, write_json  # noqa: E402

KEYS = ("entities", "f05", "macro_precision", "macro_recall", "false_merges", "singleton_false_positive",
        "empty_on_positive", "predicted_pairs", "tp_pairs", "predicted_per_s1")
SCORE_BINS = [0.67, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001]
MARGIN_BINS = [0.0, 1e-6, 0.01, 0.05, 0.1, 0.2, 0.3, 1.0001]


def summarise(mask_s1, s1_row, pred, true, n_true, country_of_row, n_rows):
    sel = mask_s1[s1_row]
    k = np.bincount(s1_row[sel & pred], minlength=n_rows)
    tp = np.bincount(s1_row[sel & pred & true], minlength=n_rows)
    ids = np.flatnonzero(mask_s1)
    r = evaluate_arrays(tp[ids], k[ids], n_true[ids], countries=country_of_row[ids])
    return {**{x: r[x] for x in KEYS}, "by_country": {c: {x: v[x] for x in KEYS} for c, v in r["by_country"].items()}}


def hist(x, bins):
    h, _ = np.histogram(x, bins=bins)
    return {f"[{bins[i]:g},{bins[i + 1]:g})": int(h[i]) for i in range(len(bins) - 1)}


def half_of(ids):
    return np.array([int(hashlib.sha1(f"42|{v}".encode()).hexdigest(), 16) % 2 for v in ids], np.int8)


def delta(a, b):
    return {x: b[x] - a[x] for x in KEYS if isinstance(a.get(x), (int, float))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_d01_best_claimant.yaml")
    ap.add_argument("--scored-tag", default="forensic_alltrain_B04")
    ap.add_argument("--out-dir", default="experiments/runs/EXP-D-01")
    ap.add_argument("--skip-test", action="store_true")
    a = ap.parse_args()
    setup_logging()
    configure_arrow()
    cfg = load_config(a.config)
    paths = Paths(cfg)
    model_file = Path(cfg["matcher"]["model_path"])
    t = float(json.loads(model_file.read_text(encoding="utf-8"))["meta"]["threshold"])
    margin = float(cfg["decision"]["competition"].get("margin", 0.0))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"experiment": cfg["experiment"]["id"], "config": a.config, "model": str(model_file), "threshold": t,
           "rule": {"name": "best_claimant", "margin": margin,
                    "definition": "after the threshold, per S2/S3 record keep only the predicted pair with the highest score",
                    "tie_break": "equal score -> lowest s1_row (row order of the S1 file)"},
           "scored_parts": str(paths.scored_dir("train", a.scored_tag)), "seed": cfg["seed"]}

    # ---------------- load all-train scores + labels
    d = _read_parts(paths.scored_dir("train", a.scored_tag), ["s1_row", "pool_row", "score"])
    s1_row, pool_row, score = d["s1_row"].astype(np.int64), d["pool_row"].astype(np.int64), d["score"].astype(np.float32)
    del d
    gtc, n_true, n_rows = gt_codes(paths)
    true = np.isin(s1_row * SHIFT + pool_row, gtc)
    pred = score >= np.float32(t)
    country_of_row = np.empty(n_rows, dtype=object)
    for k in load_manifest(paths, "train")["source1"]["countries"]:
        country_of_row[read_country(paths, "train", "source1", k, ["row"]).column("row").to_numpy()] = k
    s1_ids = _ids(paths, "train", "source1")
    fit_rows = pc.index_in(pa.array(fit_s1_subset(cfg, paths), pa.string()), value_set=s1_ids).to_numpy(zero_copy_only=False).astype(np.int64)
    val = load_val_ids(paths)
    val_rows = pc.index_in(pa.array(val, pa.string()), value_set=s1_ids).to_numpy(zero_copy_only=False).astype(np.int64)
    is_fit = np.zeros(n_rows, bool)
    is_fit[fit_rows] = True
    is_val = np.zeros(n_rows, bool)
    is_val[val_rows] = True
    E = ~is_fit
    if (is_fit & is_val).any():
        raise RuntimeError("leakage: fit and validation S1 overlap")
    half = half_of(s1_ids.to_pylist())
    rep["populations"] = {"all_train_s1_competing": int(n_rows), "fit_s1_in_sample_competitors_only": int(is_fit.sum()),
                          "E_out_of_sample": int(E.sum()), "validation": int(is_val.sum()),
                          "validation_half_A": int((is_val & (half == 0)).sum()), "validation_half_B": int((is_val & (half == 1)).sum()),
                          "pairs": int(len(s1_row)), "fit_validation_overlap": 0}

    # ---------------- the rule (production function), timed
    with PeakMemory() as pm:
        t0 = time.perf_counter()
        keep = best_claimant_keep(pool_row, s1_row, score, pred, margin=margin)
        rule_s = time.perf_counter() - t0
    rep["rule_runtime"] = {"pairs_in": int(len(s1_row)), "predicted_in": int(pred.sum()), "seconds": round(rule_s, 2),
                           "process_peak_rss_gb": round(pm.peak_gb, 2)}
    assert not (keep & ~pred).any()

    # ---------------- control vs EXP-D-01 on every population
    pops = {"E": E, "validation": is_val, "validation_half_A": is_val & (half == 0), "validation_half_B": is_val & (half == 1),
            "E_half_A": E & (half == 0), "E_half_B": E & (half == 1)}
    res = {}
    for name, m in pops.items():
        c, x = summarise(m, s1_row, pred, true, n_true, country_of_row, n_rows), summarise(m, s1_row, keep, true, n_true, country_of_row, n_rows)
        res[name] = {"control_B04": c, "EXP_D01": x, "delta": delta(c, x),
                     "delta_by_country": {cc: delta(c["by_country"][cc], x["by_country"][cc]) for cc in c["by_country"]}}
    rep["results"] = res
    prod = read_json(paths.run_dir("EXP-B-04") / "metrics.json")["at_threshold_used"]["f05"]
    rep["sanity_control_validation_vs_production"] = {"production": prod, "here": res["validation"]["control_B04"]["f05"],
                                                      "abs_diff": abs(prod - res["validation"]["control_B04"]["f05"])}

    # ---------------- competition statistics (all train S1 compete), overall and by S1 country
    comp = {"all": {"before": claim_stats(pool_row, s1_row, pred), "after": claim_stats(pool_row, s1_row, keep)}}
    for cc in sorted(set(country_of_row.tolist())):
        sel = country_of_row[s1_row] == cc
        comp[cc] = {"before": claim_stats(pool_row[sel], s1_row[sel], pred[sel]), "after": claim_stats(pool_row[sel], s1_row[sel], keep[sel])}
    removed = pred & ~keep
    comp["all"]["claims_removed"] = int(removed.sum())
    comp["all"]["true_claims_removed"] = int((removed & true).sum())
    comp["all"]["false_claims_removed"] = int((removed & ~true).sum())
    claimants = claimants_per_pair(pool_row, pred)
    conflicted = pred & (claimants >= 2)
    comp["all"]["winners_on_conflicted_records"] = int((conflicted & keep).sum())
    comp["all"]["losers"] = int(removed.sum())
    comp["all"]["winners_true_share"] = float(true[conflicted & keep].mean()) if (conflicted & keep).any() else None
    comp["all"]["conflict_rate_pairs"] = float(conflicted.sum() / max(pred.sum(), 1))
    rep["competition_all_train"] = comp

    # ---------------- removed-claim analysis (losers whose S1 is in E)
    wscore = winner_score_per_pair(pool_row, s1_row, score, pred)
    rm = np.flatnonzero(removed & E[s1_row])
    size = np.minimum(claimants[rm], 5)
    rt = true[rm]
    singleton = n_true[s1_row[rm]] == 0
    ctry = country_of_row[s1_row[rm]]
    marg = wscore[rm].astype(np.float64) - score[rm].astype(np.float64)

    def split(sel):
        return {"removed": int(sel.sum()), "true_removed": int((sel & rt).sum()), "false_removed": int((sel & ~rt).sum()),
                "true_share": float(rt[sel].mean()) if sel.any() else None}

    rep["removed_claims_in_E"] = {
        "all": split(np.ones(len(rm), bool)),
        "by_country": {cc: split(ctry == cc) for cc in sorted(set(ctry.tolist()))},
        "loser_s1_singleton": split(singleton), "loser_s1_positive": split(~singleton),
        "by_conflict_size": {("5+" if z == 5 else str(z)): split(size == z) for z in (2, 3, 4, 5)},
        "true_removed_loser_score": hist(score[rm][rt], SCORE_BINS), "false_removed_loser_score": hist(score[rm][~rt], SCORE_BINS),
        "true_removed_winner_score": hist(wscore[rm][rt], SCORE_BINS), "false_removed_winner_score": hist(wscore[rm][~rt], SCORE_BINS),
        "true_removed_margin": hist(marg[rt], MARGIN_BINS), "false_removed_margin": hist(marg[~rt], MARGIN_BINS),
        "exact_score_ties_removed": int((marg == 0).sum()),
    }
    # when a TRUE claim is removed, the ground truth says its winner must be a false claim
    rep["removed_claims_in_E"]["note"] = ("A removed true claim implies the kept winner is false (each train record has at most one S1), "
                                          "so each true removal trades one TP for one FP.")

    # ---------------- sensitivity (reported, not optimised)
    sens = {}
    for m in (0.05, 0.10, 0.20, 0.30):
        k2 = best_claimant_keep(pool_row, s1_row, score, pred, margin=m)
        sens[f"margin_{m:.2f}"] = {"E": {x: summarise(E, s1_row, k2, true, n_true, country_of_row, n_rows)[x] for x in KEYS},
                                   "claims_removed_E": int((pred & ~k2 & E[s1_row]).sum()),
                                   "true_claims_removed_E": int((pred & ~k2 & E[s1_row] & true).sum())}
    # out-of-sample competitors only: the 200k fit S1 neither compete nor are evaluated
    oos = pred & E[s1_row]
    k3 = best_claimant_keep(pool_row, s1_row, score, oos)
    sens["oos_competitors_only_margin_0"] = {"E": {x: summarise(E, s1_row, k3, true, n_true, country_of_row, n_rows)[x] for x in KEYS},
                                             "E_control": {x: res["E"]["control_B04"][x] for x in KEYS}}
    rep["sensitivity"] = sens
    del true

    # ---------------- unlabeled test diagnostics (no ground truth; diagnostics only)
    if not a.skip_test:
        td = _read_parts(scored_dir(cfg, paths, "test"), ["s1_row", "pool_row", "score"])
        ts, tp_, tsc = td["s1_row"].astype(np.int64), td["pool_row"].astype(np.int64), td["score"].astype(np.float32)
        del td
        tpred = tsc >= np.float32(t)
        with PeakMemory() as pm2:
            t0 = time.perf_counter()
            tkeep = best_claimant_keep(tp_, ts, tsc, tpred, margin=margin)
            test_rule_s = time.perf_counter() - t0
        man = load_manifest(paths, "test")
        tc = np.empty(int(man["source1"]["rows"]), dtype=object)
        for k in man["source1"]["countries"]:
            tc[read_country(paths, "test", "source1", k, ["row"]).column("row").to_numpy()] = k
        T = {"label": "UNLABELED TEST DIAGNOSTICS (no ground truth; not a quality measurement)",
             "rule_seconds": round(test_rule_s, 2), "process_peak_rss_gb": round(pm2.peak_gb, 2),
             "all": {"before": claim_stats(tp_, ts, tpred), "after": claim_stats(tp_, ts, tkeep), "removed": int((tpred & ~tkeep).sum())}}
        for cc in sorted(set(tc.tolist())):
            sel = tc[ts] == cc
            n_s1 = int((tc == cc).sum())
            kb = np.bincount(ts[sel & tpred], minlength=len(tc))[tc == cc]
            ka = np.bincount(ts[sel & tkeep], minlength=len(tc))[tc == cc]
            T[cc] = {"s1": n_s1, "before": claim_stats(tp_[sel], ts[sel], tpred[sel]), "after": claim_stats(tp_[sel], ts[sel], tkeep[sel]),
                     "removed": int((sel & tpred & ~tkeep).sum()),
                     "predicted_per_s1_before": float(kb.mean()), "predicted_per_s1_after": float(ka.mean()),
                     "empty_rate_before": float((kb == 0).mean()), "empty_rate_after": float((ka == 0).mean()),
                     "s1_losing_at_least_one_prediction": int(((kb - ka) > 0).sum()),
                     "s1_becoming_empty": int(((kb > 0) & (ka == 0)).sum())}
        rep["test_unlabeled_diagnostics"] = T

    write_json(out / "full_competition_eval.json", rep)
    for name in ("E", "validation", "validation_half_A", "validation_half_B", "E_half_A", "E_half_B"):
        c, x = res[name]["control_B04"], res[name]["EXP_D01"]
        LOG.info("%-18s F0.5 %.4f -> %.4f (%+.4f) | P %.4f -> %.4f | R %.4f -> %.4f | FM %d -> %d | sFP %d -> %d", name,
                 c["f05"], x["f05"], x["f05"] - c["f05"], c["macro_precision"], x["macro_precision"], c["macro_recall"], x["macro_recall"],
                 c["false_merges"], x["false_merges"], c["singleton_false_positive"], x["singleton_false_positive"])
    LOG.info("sanity %s", rep["sanity_control_validation_vs_production"])
    LOG.info("removed in E %s", rep["removed_claims_in_E"]["all"])
    if "test_unlabeled_diagnostics" in rep:
        for cc, v in rep["test_unlabeled_diagnostics"].items():
            if isinstance(v, dict) and "before" in v:
                LOG.info("TEST %-6s conflicted records %d -> %d, removed %d", cc, v["before"]["conflicted_records"], v["after"]["conflicted_records"], v["removed"])


if __name__ == "__main__":
    main()
