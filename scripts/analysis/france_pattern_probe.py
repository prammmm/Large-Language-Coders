"""France risk probe (READ-ONLY, no France labels needed).

Two "risky acceptance" patterns, computed identically for every country:
  location_name  every name_core token shared by the two records also appears in an address
                 of either record (the name match is carried by a place word, e.g. "bordeaux")
  num_diff_street both addresses share a house number (leading zeros ignored) but their street
                 words (address words minus stop-words and name tokens) overlap little (Jaccard < 0.25)

1. VALIDATION (India/US, labelled): precision of accepted pairs with / without each pattern.
2. TEST (France/India/US, unlabelled): how often accepted pairs show each pattern.
=> if France accepts far more pattern pairs, and those pairs are measurably less precise in
   India/US, France carries extra false merges from these patterns (estimate, not a measurement).

    python scripts/analysis/france_pattern_probe.py --config configs/exp_m03_lgbm.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_exp_b01 import load_views  # noqa: E402

from ber.config import Paths, load_config  # noqa: E402
from ber.data.normalize_records import load_manifest, read_country  # noqa: E402
from ber.data.split import load_val_ids  # noqa: E402
from ber.features.pair_features import strip_zeros_list  # noqa: E402
from ber.models import make_scorer  # noqa: E402
from ber.normalization.views import ADDRESS_STOPWORDS  # noqa: E402
from ber.pipeline.run_baseline import _read_parts, label_pairs, scored_dir  # noqa: E402
from ber.utils.runtime import configure_arrow, read_json, write_json  # noqa: E402


def patterns(a, b):
    """a, b = view tuples (name_norm, name_core, script, addr_norm, addr_nums, addr_words)."""
    na, nb = set(a[1].split()), set(b[1].split())
    shared = na & nb
    addr_tokens = set(a[5].split()) | set(b[5].split())
    location_name = bool(shared) and shared <= addr_tokens
    nums_a, nums_b = set(strip_zeros_list([a[4]])[0].split()), set(strip_zeros_list([b[4]])[0].split())
    street_a = {w for w in a[5].split() if w not in ADDRESS_STOPWORDS and w not in na and len(w) > 1}
    street_b = {w for w in b[5].split() if w not in ADDRESS_STOPWORDS and w not in nb and len(w) > 1}
    j = len(street_a & street_b) / max(len(street_a | street_b), 1)
    num_diff_street = bool(nums_a & nums_b) and (street_a or street_b) and j < 0.25
    return location_name, bool(num_diff_street)


def views_for(paths, split, s1_rows, pool_rows, n_s2):
    v1 = load_views(paths, split, "source1", np.unique(s1_rows))
    v2 = load_views(paths, split, "source2", np.unique(pool_rows[pool_rows < n_s2]))
    v3 = load_views(paths, split, "source3", np.unique(pool_rows[pool_rows >= n_s2] - n_s2))
    return v1, (lambda p: v2.get(int(p)) if p < n_s2 else v3.get(int(p - n_s2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_m03_lgbm.yaml")
    ap.add_argument("--per-country", type=int, default=100_000)
    a = ap.parse_args()
    configure_arrow()
    cfg = load_config(a.config)
    paths = Paths(cfg)
    exp = cfg["experiment"]["id"]
    t = float(read_json(paths.run_dir(exp) / "metrics.json")["threshold_used"])
    rng = np.random.default_rng(cfg["seed"])
    out = {"threshold": t, "validation": {}, "test": {}}

    # ---------------- validation (labelled India/US)
    scorer = make_scorer(cfg, paths)
    need = [c for c in scorer.required_features if c != "cheap_rank"]
    d = _read_parts(scored_dir(cfg, paths, "train", "eval_read"), list(dict.fromkeys(["s1_row", "pool_row", "cheap_rank"] + need)))
    val = load_val_ids(paths)
    is_true, pos, n_true, countries = label_pairs(paths, val, d["s1_row"], d["pool_row"])
    acc = np.flatnonzero(scorer.score(d) >= np.float32(t))
    n_s2 = int(load_manifest(paths, "train")["source2"]["rows"])
    for c in sorted(set(countries.tolist())):
        idx = acc[countries[pos[acc]] == c]
        idx = rng.choice(idx, min(a.per_country, len(idx)), replace=False)
        v1, pv = views_for(paths, "train", d["s1_row"][idx].astype(np.int64), d["pool_row"][idx].astype(np.int64), n_s2)
        flags = np.array([patterns(v1[int(d["s1_row"][i])], pv(d["pool_row"][i])) for i in idx])
        tr = is_true[idx]
        res = {"accepted_sampled": int(len(idx)), "precision_all": float(tr.mean())}
        for k, name in enumerate(("location_name", "num_diff_street")):
            f = flags[:, k]
            res[name] = {"rate": float(f.mean()), "precision_with": float(tr[f].mean()) if f.any() else None,
                         "precision_without": float(tr[~f].mean()) if (~f).any() else None}
        either = flags.any(axis=1)
        res["either"] = {"rate": float(either.mean()), "precision_with": float(tr[either].mean()), "precision_without": float(tr[~either].mean())}
        out["validation"][c] = res
        print("VAL", c, res)

    # ---------------- test (unlabelled, all countries)
    tdir = paths.scored_dir("test", cfg["scoring"]["tag"])
    tb = pa.concat_tables([pq.read_table(p, columns=["s1_row", "pool_row", "score"]) for p in sorted(tdir.glob("*.parquet"))])
    sr, pr, sc = (tb.column(c).to_numpy() for c in ("s1_row", "pool_row", "score"))
    del tb
    m = load_manifest(paths, "test")
    tn_s2 = int(m["source2"]["rows"])
    country = np.empty(m["source1"]["rows"], dtype=object)
    for k in m["source1"]["countries"]:
        country[read_country(paths, "test", "source1", k, ["row"]).column("row").to_numpy()] = k
    tacc = np.flatnonzero(sc >= np.float32(t))
    for c in sorted(set(country.tolist())):
        idx = tacc[country[sr[tacc]] == c]
        idx = rng.choice(idx, min(a.per_country, len(idx)), replace=False)
        v1, pv = views_for(paths, "test", sr[idx].astype(np.int64), pr[idx].astype(np.int64), tn_s2)
        flags = np.array([patterns(v1[int(sr[i])], pv(pr[i])) for i in idx])
        out["test"][c] = {"accepted_sampled": int(len(idx)), "location_name_rate": float(flags[:, 0].mean()),
                          "num_diff_street_rate": float(flags[:, 1].mean()), "either_rate": float(flags.any(axis=1).mean())}
        print("TEST", c, out["test"][c])

    # ---------------- rough France estimate: pattern-driven false-merge share of accepted pairs
    vv = [out["validation"][c]["either"] for c in out["validation"]]
    p_with = float(np.mean([v["precision_with"] for v in vv]))
    p_without = float(np.mean([v["precision_without"] for v in vv]))
    est = {}
    for c, v in out["test"].items():
        r = v["either_rate"]
        est[c] = {"estimated_pair_precision_if_val_pattern_precision_holds": r * p_with + (1 - r) * p_without}
    out["estimate_pair_precision_by_country"] = est
    print("ESTIMATE", est)
    write_json(paths.run_dir(exp) / "analysis" / "france_pattern_probe.json", out)


if __name__ == "__main__":
    main()
