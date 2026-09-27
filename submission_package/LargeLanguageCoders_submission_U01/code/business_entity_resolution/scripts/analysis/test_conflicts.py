"""Test-side conflict lower bound (READ-ONLY, no labels): in training every S2/S3 record
belongs to at most one S1, so a test record predicted for k >= 2 S1s carries at least k-1
wrong predictions. Reports that lower bound per country for each run.

    python scripts/analysis/test_conflicts.py --runs EXP-F-01b:scored_EXP-F-01b EXP-M-03:scored_EXP-M-03
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
    ap.add_argument("--runs", nargs="+", required=True, help="RUN_ID:test_scored_tag (threshold from the run's model.json)")
    ap.add_argument("--out", default="experiments/runs/EXP-M-03/analysis/test_conflicts.json")
    a = ap.parse_args()
    configure_arrow()
    paths = Paths(load_config(a.config))
    m = load_manifest(paths, "test")
    country = np.empty(m["source1"]["rows"], dtype=object)
    for k in m["source1"]["countries"]:
        country[read_country(paths, "test", "source1", k, ["row"]).column("row").to_numpy()] = k
    out = {}
    for spec in a.runs:
        run, tag = spec.split(":")
        t = json.loads(run_model_file(paths, run).read_text(encoding="utf-8"))["meta"]["threshold"]
        tb = pa.concat_tables([pq.read_table(p, columns=["s1_row", "pool_row", "score"]) for p in sorted(paths.scored_dir("test", tag).glob("*.parquet"))])
        sr, pr, sc = (tb.column(c).to_numpy() for c in ("s1_row", "pool_row", "score"))
        sel = sc >= np.float32(t)
        sr, pr, sc = sr[sel], pr[sel].astype(np.int64), sc[sel]
        cnt = np.bincount(pr)
        multi = cnt[pr] >= 2
        o = np.lexsort((-sc, pr))
        first = np.ones(len(o), bool)
        first[1:] = pr[o][1:] != pr[o][:-1]
        loser = np.zeros(len(pr), bool)
        loser[o[~first]] = True  # every claim except the best-scoring one is certainly wrong
        res = {"threshold": t, "predicted_pairs": int(len(pr)), "pairs_on_conflicted_records": int(multi.sum()),
               "certainly_wrong_lower_bound": int(loser.sum())}
        for c in sorted(set(country.tolist())):
            cs = country[sr] == c
            res[c] = {"predicted": int(cs.sum()), "conflicted_pairs_share": float(multi[cs].mean()),
                      "certainly_wrong": int((loser & cs).sum()), "certainly_wrong_share": float((loser & cs).sum() / max(cs.sum(), 1))}
        out[run] = res
        print(run, {c: (v["certainly_wrong"], round(v["certainly_wrong_share"], 4)) for c, v in res.items() if isinstance(v, dict)})
    write_json(Path(a.out), out)


if __name__ == "__main__":
    main()
