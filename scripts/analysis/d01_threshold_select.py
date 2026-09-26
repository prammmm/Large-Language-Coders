"""Step 3: choose the global threshold WITH D-01 in the loop, under full competition, without
touching the frozen validation set or the evaluation half.

Tuning set T  = E half A minus the frozen validation S1   (E = train S1 not in the fit set)
Held-out      = E half B, and the frozen validation fold (both never used for the choice)
All train S1 compete (all-train scored parts of the arm); D-01 = production ``best_claimant_keep``.
Nothing here writes a config: the chosen value is reported, and adopted only if both held-out
populations agree it beats the model's inner-split threshold.

    python scripts/analysis/d01_threshold_select.py --config configs/exp_n03_legal_tokens.yaml --tag forensic_alltrain_N03
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

GRID = np.round(np.arange(0.50, 0.801, 0.01), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True, help="all-train scored parts of this arm")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    setup_logging()
    configure_arrow()
    cfg = load_config(a.config)
    paths = Paths(cfg)
    exp = cfg["experiment"]["id"]
    t_model = float(json.loads((paths.run_dir(exp) / "model.json").read_text(encoding="utf-8"))["meta"]["threshold"])
    gtc, n_true, n_rows = gt_codes(paths)
    s1_ids = _ids(paths, "train", "source1")
    idx = lambda ids: pc.index_in(pa.array(ids, pa.string()), value_set=s1_ids).to_numpy(zero_copy_only=False).astype(np.int64)  # noqa: E731
    fit = np.zeros(n_rows, bool)
    fit[idx(fit_s1_subset(cfg, paths))] = True
    val = np.zeros(n_rows, bool)
    val[idx(load_val_ids(paths))] = True
    half = half_of(s1_ids.to_pylist())
    E = ~fit
    pops = {"tune(E_A-minus-val)": E & (half == 0) & ~val, "heldout_E_B": E & (half == 1), "heldout_validation": val}
    if (pops["tune(E_A-minus-val)"] & val).any() or (pops["tune(E_A-minus-val)"] & fit).any():
        raise RuntimeError("tuning set overlaps validation or fit")
    d = _read_parts(paths.scored_dir("train", a.tag), ["s1_row", "pool_row", "score"])
    s1, pool, score = d["s1_row"].astype(np.int64), d["pool_row"].astype(np.int64), d["score"].astype(np.float32)
    del d
    true = np.isin(s1 * SHIFT + pool, gtc)
    table = {}
    for t in sorted(set(GRID.tolist()) | {t_model}):
        keep = best_claimant_keep(pool, s1, score, score >= np.float32(t))
        k = np.bincount(s1[keep], minlength=n_rows)
        tp = np.bincount(s1[keep & true], minlength=n_rows)
        row = {}
        for name, m in pops.items():
            r = evaluate_arrays(tp[m], k[m], n_true[m])
            row[name] = {x: r[x] for x in ("f05", "macro_precision", "macro_recall", "false_merges")}
        table[f"{t:.2f}"] = row
    tune = "tune(E_A-minus-val)"
    best = max(table, key=lambda t: (round(table[t][tune]["f05"], 6), float(t)))
    m = f"{t_model:.2f}"
    rep = {"config": a.config, "tag": a.tag, "model_threshold": t_model, "chosen_on_tuning_set": float(best),
           "populations": {k: int(v.sum()) for k, v in pops.items()},
           "gain_vs_model_threshold": {p: table[best][p]["f05"] - table[m][p]["f05"] for p in pops},
           "precision_change": {p: table[best][p]["macro_precision"] - table[m][p]["macro_precision"] for p in pops},
           "table": table}
    rep["adopt"] = bool(best != m and all(rep["gain_vs_model_threshold"][p] > 0 for p in ("heldout_E_B", "heldout_validation")))
    out = Path(a.out or paths.run_dir(exp) / "d01_threshold_select.json")
    write_json(out, rep)
    LOG.info("model t=%.2f | chosen on tuning set t=%s | held-out gains %s | precision change %s | adopt=%s", t_model, best,
             {k: round(v, 5) for k, v in rep["gain_vs_model_threshold"].items()}, {k: round(v, 4) for k, v in rep["precision_change"].items()}, rep["adopt"])


if __name__ == "__main__":
    main()
