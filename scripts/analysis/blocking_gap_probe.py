"""Blocking-gap probe (READ-ONLY): for every validation TRUE pair that is NOT in the candidate
list of a run, which kind of blocking key could reach it? (Upper-bound coverage per key family;
ignores block-size caps and the top-K limit, so real keys will reach fewer.)

Key families checked (all country-agnostic, from the provided data only):
  rare_name_token        shares a name_core token with pool DF <= 200 (current name_token pass)
  name_token+addr_word   shares ANY name_core token AND a useful address word
  addr_word_pair         shares >= 2 useful address words
  addr_number            shares a house number (leading zeros ignored)
  name_only              shares a name_core token (any DF)
  nothing_shared         none of the above
also: address empty on either side, non-Latin pool name vs Latin S1 name.

    python scripts/analysis/blocking_gap_probe.py --config configs/exp_m03_lgbm.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_exp_b01 import load_views  # noqa: E402

from ber.config import Paths, load_config  # noqa: E402
from ber.data.normalize_records import load_manifest, read_country  # noqa: E402
from ber.data.split import load_val_ids  # noqa: E402
from ber.features.pair_features import strip_zeros_list  # noqa: E402
from ber.normalization.views import ADDRESS_STOPWORDS  # noqa: E402
from ber.pipeline.run_baseline import _ids, _read_parts, pool_ids, scored_dir  # noqa: E402
from ber.utils.runtime import configure_arrow, write_json  # noqa: E402

SHIFT = np.int64(1) << np.int64(24)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_m03_lgbm.yaml")
    ap.add_argument("--out", default="experiments/runs/EXP-M-03/analysis/blocking_gap_probe.json")
    a = ap.parse_args()
    configure_arrow()
    cfg = load_config(a.config)
    paths = Paths(cfg)
    n_s2 = int(load_manifest(paths, "train")["source2"]["rows"])
    val = pa.array(load_val_ids(paths), pa.string())
    s1_ids, p_ids = _ids(paths, "train", "source1"), pool_ids(paths, "train")

    gt = pq.read_table(paths.interim_file("train", "ground_truth"))
    gt = gt.filter(pc.is_in(gt.column("source1_entity_id"), value_set=val))
    L = pc.split_pattern(gt.column("matched_entity_ids"), ",")
    flat, par = pc.list_flatten(L), pc.list_parent_indices(L)
    keep = pc.not_equal(flat, "")
    gs = pc.index_in(pc.take(gt.column("source1_entity_id"), pc.filter(par, keep)), value_set=s1_ids).to_numpy(zero_copy_only=False).astype(np.int64)
    gp = pc.index_in(pc.filter(flat, keep), value_set=p_ids).to_numpy(zero_copy_only=False).astype(np.int64)
    d = _read_parts(scored_dir(cfg, paths, "train", "eval_read"), ["s1_row", "pool_row"])
    missing = ~np.isin(gs * SHIFT + gp, d["s1_row"].astype(np.int64) * SHIFT + d["pool_row"].astype(np.int64))
    gs, gp = gs[missing], gp[missing]

    # pool DF of name_core tokens per country (for the "rare" criterion)
    df: dict[str, Counter] = {}
    m = load_manifest(paths, "train")
    for k in m["source1"]["countries"]:
        cnt = Counter()
        for src in ("source2", "source3"):
            for s in read_country(paths, "train", src, k, ["name_core"]).column("name_core").to_pylist():
                cnt.update(set(s.split()))
        df[k] = cnt
    country_of = {}
    for k in m["source1"]["countries"]:
        for r in read_country(paths, "train", "source1", k, ["row"]).column("row").to_numpy():
            country_of[int(r)] = k

    v1 = load_views(paths, "train", "source1", np.unique(gs))
    v2 = load_views(paths, "train", "source2", np.unique(gp[gp < n_s2]))
    v3 = load_views(paths, "train", "source3", np.unique(gp[gp >= n_s2] - n_s2))

    fam = Counter()
    by_c: dict[str, Counter] = {}
    n_by_c = Counter()
    for s, p in zip(gs, gp):
        x = v1.get(int(s))
        y = v2.get(int(p)) if p < n_s2 else v3.get(int(p - n_s2))
        if not x or not y:
            continue
        c = country_of.get(int(s), "?")
        n_by_c[c] += 1
        bc = by_c.setdefault(c, Counter())
        nt = set(x[1].split()) & set(y[1].split())
        uw = {w for w in set(x[5].split()) & set(y[5].split()) if w not in ADDRESS_STOPWORDS and len(w) > 1}
        nums = set(strip_zeros_list([x[4]])[0].split()) & set(strip_zeros_list([y[4]])[0].split())
        flags = {
            "rare_name_token(df<=200)": any(df[c][t] <= 200 for t in nt),
            "name_token+addr_word": bool(nt) and bool(uw),
            "addr_word_pair(>=2)": len(uw) >= 2,
            "addr_number": bool(nums),
            "name_only(any df)": bool(nt),
            "address_empty": x[3] == "" or y[3] == "",
            "script_mismatch": x[2] == 0 and y[2] in (1, 2),
        }
        flags["nothing_shared"] = not (nt or uw or nums)
        flags["reachable_by_new_families(name+addr OR addr_pair)"] = flags["name_token+addr_word"] or flags["addr_word_pair(>=2)"]
        for k2, v in flags.items():
            if v:
                fam[k2] += 1
                bc[k2] += 1
    n = sum(n_by_c.values())
    out = {"missing_true_pairs": int(n), "share": {k: v / n for k, v in fam.items()},
           "by_country": {c: {"pairs": n_by_c[c], **{k: v / n_by_c[c] for k, v in by_c[c].items()}} for c in by_c}}
    write_json(Path(a.out), out)
    print("missing true pairs:", n)
    for k, v in sorted(out["share"].items(), key=lambda x: -x[1]):
        print(f"{v:.3f}  {k}  " + "  ".join(f"{c} {out['by_country'][c].get(k, 0):.3f}" for c in sorted(out['by_country'])))


if __name__ == "__main__":
    main()
