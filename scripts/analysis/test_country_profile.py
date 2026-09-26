"""Test-side sanity profile per country (no labels needed): candidates/S1, predicted/S1,
empty-prediction share and accepted-candidate share, for one or more experiments' test parts.
Used as the France guard for EXP-M-03: LightGBM must not become much more aggressive in France
than in India/US relative to the Submission-2 LR.

    python scripts/analysis/test_country_profile.py --runs EXP-F-01b:scored_EXP-F-01b EXP-M-03:scored_EXP-M-03
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ber.config import Paths, load_config
from ber.data.normalize_records import load_manifest, read_country
from ber.utils.runtime import configure_arrow, write_json


def run_model_file(paths, run):
    """model.json of a run, or the model_path recorded in its config snapshot (fixed-model runs)."""
    from pathlib import Path as _P
    f = paths.run_dir(run) / "model.json"
    if f.exists():
        return f
    import yaml
    snap = paths.run_dir(run) / "config_snapshot.yaml"
    mp = (yaml.safe_load(snap.read_text(encoding="utf-8")).get("matcher") or {}).get("model_path")
    return _P(mp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_v0.yaml")
    ap.add_argument("--runs", nargs="+", required=True, help="RUN_ID:test_scored_tag (threshold from the run's submission/model)")
    ap.add_argument("--out", default="experiments/runs/EXP-M-03/test_country_profile.json")
    a = ap.parse_args()
    configure_arrow()
    paths = Paths(load_config(a.config))
    m = load_manifest(paths, "test")
    n = m["source1"]["rows"]
    country = np.empty(n, dtype=object)
    for k in m["source1"]["countries"]:
        country[read_country(paths, "test", "source1", k, ["row"]).column("row").to_numpy()] = k
    out = {}
    for spec in a.runs:
        run, tag = spec.split(":")
        t = json.loads(run_model_file(paths, run).read_text(encoding="utf-8"))["meta"]["threshold"]
        parts = sorted(paths.scored_dir("test", tag).glob("*.parquet"))
        tb = pa.concat_tables([pq.read_table(p, columns=["s1_row", "score"]) for p in parts])
        r, s = tb.column("s1_row").to_numpy().astype(np.int64), tb.column("score").to_numpy()
        c_n = np.bincount(r, minlength=n)
        k = np.bincount(r[s >= t], minlength=n)
        res = {"threshold": t}
        for c in sorted(set(country.tolist())):
            sel = country == c
            res[c] = {"s1": int(sel.sum()), "candidates_per_s1": float(c_n[sel].mean()), "predicted_per_s1": float(k[sel].mean()),
                      "empty_prediction_share": float((k[sel] == 0).mean()),
                      "accepted_candidate_share": float(k[sel].sum() / max(c_n[sel].sum(), 1))}
        out[run] = res
        print(run, "t=%.2f" % t, {c: (round(v["predicted_per_s1"], 3), round(v["empty_prediction_share"], 3)) for c, v in res.items() if c != "threshold"})
    runs = list(out)
    if len(runs) == 2:
        a0, b0 = out[runs[0]], out[runs[1]]
        ratio = {c: b0[c]["predicted_per_s1"] / max(a0[c]["predicted_per_s1"], 1e-9) for c in a0 if c != "threshold"}
        out["predicted_per_s1_ratio_second_vs_first"] = ratio
        print("predicted/S1 ratio (second / first):", {c: round(v, 3) for c, v in ratio.items()})
    write_json(Path(a.out), out)


if __name__ == "__main__":
    main()
