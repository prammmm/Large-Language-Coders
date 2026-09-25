"""Submission gate check: compare experiments on the two halves of the frozen validation
split (seeded hash of the S1 id), each at the threshold that experiment actually uses.

    python scripts/analysis/compare_val_halves.py --runs EXP-B-01-recheck:scored EXP-F-01b-dev:scored_EXP-F-01b
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ber.config import Paths, load_config
from ber.evaluation.metrics import evaluate_arrays
from ber.models.logistic import LogisticModel
from ber.pipeline.run_baseline import _read_parts, label_pairs
from ber.data.split import load_val_ids
from ber.utils.runtime import configure_arrow, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_v0.yaml")
    ap.add_argument("--runs", nargs="+", required=True, help="RUN_ID:scored_tag (threshold read from the run's metrics.json)")
    a = ap.parse_args()
    configure_arrow()
    paths = Paths(load_config(a.config))
    val = load_val_ids(paths)
    half = np.array([int(hashlib.sha1(f"42|{v}".encode()).hexdigest(), 16) % 2 for v in val])
    out = {}
    for spec in a.runs:
        run, tag = spec.split(":")
        t = json.loads((paths.run_dir(run) / "metrics.json").read_text(encoding="utf-8"))["threshold_used"]
        mfile = paths.run_dir(run) / "model.json"
        model = LogisticModel.load(mfile) if mfile.exists() else None
        cols = ["s1_row", "pool_row", "score"] + ([c for c in model.required_features if c != "score"] if model else [])
        d = _read_parts(paths.scored_dir("train", tag), list(dict.fromkeys(cols)))
        # stored "score" may come from another matcher (evaluate re-scores in memory): re-score with the run's model
        score = model.score(d) if model else d["score"]
        is_true, pos, n_true, countries = label_pairs(paths, val, d["s1_row"], d["pool_row"])
        sel = score >= np.float32(t)
        m = len(val)
        tp = np.bincount(pos[sel & is_true], minlength=m)
        k = np.bincount(pos[sel], minlength=m)
        res = {}
        for h in (0, 1):
            e = half == h
            r = evaluate_arrays(tp[e], k[e], n_true[e])
            res[f"half_{'AB'[h]}"] = {x: r[x] for x in ("f05", "macro_precision", "macro_recall", "false_merges", "singleton_false_positive")}
        out[run] = {"threshold": t, **res}
        print(run, "t=%.2f" % t, {h: round(v["f05"], 4) for h, v in res.items() if h.startswith("half")},
              {h: round(v["macro_precision"], 4) for h, v in res.items() if h.startswith("half")})
    write_json(Path("experiments/runs/val_halves_comparison.json"), out)


if __name__ == "__main__":
    main()
