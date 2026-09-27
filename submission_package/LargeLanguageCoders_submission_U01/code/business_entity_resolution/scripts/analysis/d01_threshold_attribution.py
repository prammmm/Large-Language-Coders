"""DIAGNOSTIC ONLY (attribution, no selection): F0.5 on E under full competition + D-01 for a
grid of thresholds, for each arm's all-train scores. Answers: how much of an arm-vs-arm
difference is a threshold shift rather than the blocker/model change? Nothing here is used to
choose a production threshold (that stays the inner train-fold procedure).

    python scripts/analysis/d01_threshold_attribution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_competition import SHIFT, gt_codes  # noqa: E402

from ber.config import Paths, load_config  # noqa: E402
from ber.decision.competition import best_claimant_keep  # noqa: E402
from ber.evaluation.metrics import evaluate_arrays  # noqa: E402
from ber.pipeline.run_baseline import _ids, _read_parts, fit_s1_subset  # noqa: E402
from ber.utils.runtime import configure_arrow, write_json  # noqa: E402

GRID = [0.55, 0.60, 0.62, 0.65, 0.67, 0.70, 0.75]


def main():
    configure_arrow()
    cfg = load_config("configs/exp_b04_blocking_v2.yaml")
    paths = Paths(cfg)
    gtc, n_true, n_rows = gt_codes(paths)
    s1_ids = _ids(paths, "train", "source1")
    fit = pc.index_in(pa.array(fit_s1_subset(cfg, paths), pa.string()), value_set=s1_ids).to_numpy(zero_copy_only=False).astype(np.int64)
    E = np.ones(n_rows, bool)
    E[fit] = False
    ids = np.flatnonzero(E)
    rep = {"label": "DIAGNOSTIC threshold attribution on E (not used for selection)", "grid": GRID}
    for arm, tag in (("B04", "forensic_alltrain_B04"), ("B05", "forensic_alltrain_B05")):
        d = _read_parts(paths.scored_dir("train", tag), ["s1_row", "pool_row", "score"])
        s1, pool, score = d["s1_row"].astype(np.int64), d["pool_row"].astype(np.int64), d["score"].astype(np.float32)
        true = np.isin(s1 * SHIFT + pool, gtc)
        rep[arm] = {}
        for t in GRID:
            row = {}
            for name, pred in (("unrestricted", score >= np.float32(t)),):
                keep = best_claimant_keep(pool, s1, score, pred)
                for lab, m in (("unrestricted", pred), ("D01", keep)):
                    k = np.bincount(s1[m], minlength=n_rows)[ids]
                    tp = np.bincount(s1[m & true], minlength=n_rows)[ids]
                    r = evaluate_arrays(tp, k, n_true[ids])
                    row[lab] = {x: r[x] for x in ("f05", "macro_precision", "macro_recall", "false_merges")}
            rep[arm][f"{t:.2f}"] = row
            print(arm, t, {lab: (round(v["f05"], 4), round(v["macro_precision"], 4), round(v["macro_recall"], 4)) for lab, v in row.items()}, flush=True)
        del d, s1, pool, score, true
    write_json(Path("experiments/runs/EXP-B-05/d01_threshold_attribution.json"), rep)


if __name__ == "__main__":
    main()
