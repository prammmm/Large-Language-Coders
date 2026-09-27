"""EXP-D-01 (diagnostic, NOT adopted): how much could a LEARNED matcher gain over the
hand-weighted rule, using exactly the same candidates and features?

Protocol (no leakage across S1 entities):
  * validation S1 entities are split into two halves A/B (seeded hash of the id);
  * fit logistic regression on half A pairs, tune the global threshold on half A,
    evaluate entity-level F0.5 on half B; then swap (B -> A);
  * the weighted rule gets the identical treatment (threshold tuned on the fit half);
  * blocking is untouched, so both are capped by the same oracle F0.5.
Uses torch (CPU) only as an optimiser; no pretrained weights.

    python scripts/analysis/matcher_headroom_probe.py --config configs/baseline_v0.yaml
"""

from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

from ber.config import Paths, load_config
from ber.data.split import load_val_ids
from ber.evaluation.metrics import f05_from_counts
from ber.features.pair_features import FEATURE_NAMES
from ber.pipeline.run_baseline import _ids, pool_ids
from ber.utils.runtime import configure_arrow, write_json

GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)


def entity_f05(pos, is_true, score, n_true, t):
    sel = score >= t
    m = len(n_true)
    return float(f05_from_counts(np.bincount(pos[sel & is_true], minlength=m), np.bincount(pos[sel], minlength=m), n_true).mean())


def tune(pos, is_true, score, n_true):
    vals = [entity_f05(pos, is_true, score, n_true, t) for t in GRID]
    i = int(np.argmax(vals))
    return float(GRID[i]), vals[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_v0.yaml")
    a = ap.parse_args()
    configure_arrow()
    torch.manual_seed(42)
    cfg = load_config(a.config)
    paths = Paths(cfg)
    val = load_val_ids(paths)
    val_arr = pa.array(val, pa.string())
    m = len(val)
    half = np.array([int(hashlib.sha1(f"42|{v}".encode()).hexdigest(), 16) % 2 for v in val])

    gt = pq.read_table(paths.interim_file("train", "ground_truth"))
    gt = gt.filter(pc.is_in(gt.column("source1_entity_id"), value_set=val_arr))
    lists = pc.split_pattern(gt.column("matched_entity_ids"), ",")
    flat, parent = pc.list_flatten(lists), pc.list_parent_indices(lists)
    keep = pc.not_equal(flat, "")
    gt_pairs = pc.binary_join_element_wise(pc.take(gt.column("source1_entity_id"), pc.filter(parent, keep)), pc.filter(flat, keep), "|")
    n_true = np.zeros(m, np.int64)
    gpos = pc.index_in(pc.take(gt.column("source1_entity_id"), pc.filter(parent, keep)), value_set=val_arr).to_numpy(zero_copy_only=False)
    np.add.at(n_true, gpos, 1)

    t = pa.concat_tables([pq.read_table(p) for p in sorted(paths.scored_dir("train").glob("*.parquet"))])
    s1s = _ids(paths, "train", "source1").take(t.column("s1_row").combine_chunks())
    cands = pool_ids(paths, "train").take(t.column("pool_row").combine_chunks())
    is_true = pc.is_in(pc.binary_join_element_wise(s1s, cands, "|"), value_set=gt_pairs).to_numpy(zero_copy_only=False)
    pos = pc.index_in(s1s, value_set=val_arr).to_numpy(zero_copy_only=False).astype(np.int64)
    rule = t.column("score").to_numpy()
    X = np.stack([t.column(f).to_numpy().astype(np.float32) for f in FEATURE_NAMES]
                 + [t.column("cheap_score").to_numpy().astype(np.float32), t.column("cheap_rank").to_numpy().astype(np.float32),
                    (t.column("name_core_jaccard").to_numpy() * t.column("addr_num_jaccard").to_numpy()).astype(np.float32),
                    (t.column("name_ratio").to_numpy() * t.column("addr_token_set").to_numpy()).astype(np.float32)], axis=1)
    names = list(FEATURE_NAMES) + ["cheap_score", "cheap_rank", "name_core_jaccard*addr_num_jaccard", "name_ratio*addr_token_set"]
    del t

    pair_half = half[pos]
    report = {"pairs": int(len(pos)), "positives": int(is_true.sum()), "folds": []}
    for fit_h in (0, 1):
        tr, te = pair_half == fit_h, pair_half != fit_h
        ent_tr, ent_te = half == fit_h, half != fit_h
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt = torch.from_numpy((X[tr] - mu) / sd)
        yt = torch.from_numpy(is_true[tr].astype(np.float32))
        w = torch.zeros(X.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")
        lossf = torch.nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad()
            loss = lossf(Xt @ w + b, yt) + 1e-4 * (w * w).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            lr_score = torch.sigmoid(torch.from_numpy((X - mu) / sd) @ w + b).numpy()

        def fold_eval(score):
            # entity-index remap per half so singleton/empty conventions hold per subset
            res = {}
            for tag, psel, esel in (("fit_half", tr, ent_tr), ("eval_half", te, ent_te)):
                remap = -np.ones(m, np.int64)
                remap[np.flatnonzero(esel)] = np.arange(esel.sum())
                res[tag] = (remap[pos[psel]], is_true[psel], score[psel], n_true[esel])
            thr, f_fit = tune(*res["fit_half"])
            return {"threshold_tuned_on_fit_half": thr, "f05_fit_half": f_fit, "f05_eval_half": entity_f05(*res["eval_half"], thr)}

        report["folds"].append({
            "fit_half": fit_h,
            "weighted_rule": fold_eval(rule),
            "logistic_regression": fold_eval(lr_score),
            "lr_standardised_weights": dict(zip(names, [round(float(x), 3) for x in w.detach().numpy()])),
        })
    for key in ("weighted_rule", "logistic_regression"):
        report[f"{key}_eval_f05_mean"] = float(np.mean([f[key]["f05_eval_half"] for f in report["folds"]]))
    out = paths.run_dir(cfg["experiment"]["id"]) / "analysis" / "matcher_headroom_probe.json"
    write_json(out, report)
    print({k: v for k, v in report.items() if k != "folds"})
    for f in report["folds"]:
        print(f["fit_half"], "rule", f["weighted_rule"], "| LR", f["logistic_regression"])


if __name__ == "__main__":
    main()
