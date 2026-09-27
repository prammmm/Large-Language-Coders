"""Stage-2 robustness check under the train -> test candidate-density shift (label-free on test).

For each stage-2 experiment: acceptance vs the stage-1 control (t + D-01) per S1 candidate-count bin
(number of eligible candidates, stage-1 score >= min_score).
  train  held-out E half B: S1 share, predictions per S1 (control / stage 2), pair precision, entity F0.5
  test   (no labels): S1 share and predictions per S1 per country x bin
If the test acceptance inflation equals the train within-bin inflation re-weighted by the test bin mix,
the test rise is explained by covariate shift (more candidates), not by a changed decision.

    python scripts/analysis/stage2_shift_check.py --configs configs/exp_s01_stage2.yaml configs/exp_s01b_stage2_robust.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ber.config import Paths, load_config
from ber.data.normalize_records import load_manifest, read_country
from ber.decision.competition import best_claimant_keep
from ber.evaluation.metrics import evaluate_arrays
from ber.pipeline import run_stage2 as S
from ber.pipeline.run_baseline import _read_parts
from ber.utils.runtime import LOG, configure_arrow, setup_logging, write_json

BINS = [(1, 3), (4, 5), (6, 7), (8, 10), (11, 10**9)]


def bin_of(n):
    out = np.full(len(n), -1, np.int8)
    for i, (lo, hi) in enumerate(BINS):
        out[(n >= lo) & (n <= hi)] = i
    return out


def control(cfg, paths, split, n_rows):
    d = _read_parts(S.stage1_parts(cfg, paths, split), ["s1_row", "pool_row", "score"])
    s1, pool, sc = d["s1_row"].astype(np.int64), d["pool_row"].astype(np.int64), d["score"].astype(np.float32)
    acc = best_claimant_keep(pool, s1, sc, sc >= np.float32(S.stage1_threshold(cfg)))
    return s1[acc] * S.SHIFT + pool[acc]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--out", default="experiments/runs/EXP-S-01b/stage2_shift_check.json")
    a = ap.parse_args()
    setup_logging()
    configure_arrow()
    rep = {}
    cfg0 = load_config(a.configs[0])
    paths = Paths(cfg0)
    P, n_tr = S.populations(cfg0, paths)
    gtc, n_true = S.gt_codes(paths)
    man = load_manifest(paths, "test")
    n_te = int(man["source1"]["rows"])
    ctry = np.empty(n_te, dtype=object)
    for k in man["source1"]["countries"]:
        ctry[read_country(paths, "test", "source1", k, ["row"]).column("row").to_numpy()] = k
    ctrl_tr = control(cfg0, paths, "train", n_tr)
    ctrl_te = control(cfg0, paths, "test", n_te)
    for cf in a.configs:
        cfg = load_config(cf)
        booster, meta = S.load_stage2(cfg, paths)
        r = {"decision": {"variant": meta["variant"], "threshold": meta["threshold"]}, "features": len(meta["features"])}
        for split in ("train", "test"):
            s1, pool, prob, _ = S.predict_all(cfg, paths, split, booster, meta["features"])
            n = n_tr if split == "train" else n_te
            ncand = np.bincount(s1, minlength=n)
            b = bin_of(ncand)
            acc = S.decide(s1, pool, prob, meta["threshold"], meta["variant"])
            code2 = s1[acc] * S.SHIFT + pool[acc]
            code1 = ctrl_tr if split == "train" else ctrl_te
            k1 = np.bincount(code1 // S.SHIFT, minlength=n)
            k2 = np.bincount(code2 // S.SHIFT, minlength=n)
            if split == "train":
                t1 = np.bincount((code1 // S.SHIFT)[np.isin(code1, gtc)], minlength=n)
                t2 = np.bincount((code2 // S.SHIFT)[np.isin(code2, gtc)], minlength=n)
                m0 = P["E_half_B"]
                rows = {}
                for i, (lo, hi) in enumerate(BINS):
                    m = m0 & (b == i)
                    if not m.any():
                        continue
                    f1, f2 = evaluate_arrays(t1[m], k1[m], n_true[m]), evaluate_arrays(t2[m], k2[m], n_true[m])
                    rows[f"{lo}-{hi if hi < 10**9 else '+'}"] = {
                        "s1_share": float(m.sum() / m0.sum()), "pred_per_s1_control": float(k1[m].mean()), "pred_per_s1_stage2": float(k2[m].mean()),
                        "inflation": float(k2[m].sum() / max(k1[m].sum(), 1)), "pair_precision_control": float(t1[m].sum() / max(k1[m].sum(), 1)),
                        "pair_precision_stage2": float(t2[m].sum() / max(k2[m].sum(), 1)), "f05_control": f1["f05"], "f05_stage2": f2["f05"]}
                r["train_E_half_B_by_bin"] = rows
                r["train_E_half_B_inflation"] = float(k2[m0].sum() / max(k1[m0].sum(), 1))
            else:
                per = {}
                for c in ["all"] + sorted(set(ctry.tolist())):
                    mc = np.ones(n, bool) if c == "all" else ctry == c
                    rows = {}
                    for i, (lo, hi) in enumerate(BINS):
                        m = mc & (b == i)
                        if m.any():
                            rows[f"{lo}-{hi if hi < 10**9 else '+'}"] = {"s1_share": float(m.sum() / mc.sum()), "inflation": float(k2[m].sum() / max(k1[m].sum(), 1))}
                    tb = r["train_E_half_B_by_bin"]
                    expected = sum(rows[k]["s1_share"] * tb[k]["inflation"] * tb[k]["pred_per_s1_control"] for k in rows if k in tb) / \
                        max(sum(rows[k]["s1_share"] * tb[k]["pred_per_s1_control"] for k in rows if k in tb), 1e-9)
                    per[c] = {"observed_inflation": float(k2[mc].sum() / max(k1[mc].sum(), 1)), "expected_from_train_bins": float(expected),
                              "pred_per_s1_control": float(k1[mc].mean()), "pred_per_s1_stage2": float(k2[mc].mean()), "by_bin": rows}
                r["test_by_country"] = per
            del s1, pool, prob, acc
        rep[cfg["experiment"]["id"]] = r
        LOG.info("%s train E_B inflation %.4f | test: %s", cfg["experiment"]["id"], r["train_E_half_B_inflation"],
                 {c: (round(v["observed_inflation"], 4), round(v["expected_from_train_bins"], 4)) for c, v in r["test_by_country"].items()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(a.out), rep)


if __name__ == "__main__":
    sys.exit(main())
