"""Compare two validation candidate sets scored by the SAME model (READ-ONLY diagnostic).

For a blocking experiment (e.g. EXP-B-03a vs EXP-M-03) this answers where the score change
comes from:
  new-only pairs   : candidates only the new blocker produces (recovered true pairs / new FPs)
  dropped pairs    : old candidates pushed out of the top-K by the new keys (lost TPs / FPs)
  common pairs     : predictions that flip because blocking inputs (cheap_score/rank) changed
Recovered true pairs and new false merges are broken down by country, shared name tokens,
address overlap, conflicts and whether the pool record belongs to another S1.

    python scripts/analysis/compare_candidate_sets.py --base EXP-M-03:scored_EXP-F-01b --new EXP-B-03a:scored_EXP-B-03a
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ber.blocking.blocker import PASS_ORDER
from ber.config import Paths, load_config
from ber.data.split import load_val_ids
from ber.models.gbdt import LightGBMModel
from ber.models.logistic import LogisticModel
from ber.pipeline.run_baseline import _ids, _read_parts, label_pairs, pool_ids
from ber.utils.runtime import configure_arrow, write_json

SHIFT = np.int64(1) << np.int64(24)
CAT_COLS = ["name_shared_tokens", "name_core_eq", "name_core_jaccard", "addr_token_set", "addr_any_empty", "num_conflict",
            "legal_form_conflict", "pass_flags"]


def load_model(path: Path):
    kind = json.loads(Path(path).read_text(encoding="utf-8")).get("type")
    return LightGBMModel.load(path) if kind == "lightgbm" else LogisticModel.load(path)


def model_file(paths: Paths, run: str) -> Path:
    cfg_snap = paths.run_dir(run) / "config_snapshot.yaml"
    if cfg_snap.exists():
        import yaml

        mp = (yaml.safe_load(cfg_snap.read_text(encoding="utf-8")).get("matcher") or {}).get("model_path")
        if mp:
            return Path(mp)
    return paths.run_dir(run) / "model.json"


def scored(paths, run, tag, val):
    mp = model_file(paths, run)
    model = load_model(mp)
    t = json.loads(mp.read_text(encoding="utf-8"))["meta"]["threshold"]
    cols = list(dict.fromkeys(["s1_row", "pool_row"] + [c for c in model.required_features] + CAT_COLS))
    d = _read_parts(paths.scored_dir("train", tag), cols)
    score = model.score(d)
    is_true, pos, n_true, countries = label_pairs(paths, val, d["s1_row"], d["pool_row"])
    code = d["s1_row"].astype(np.int64) * SHIFT + d["pool_row"].astype(np.int64)
    return {"d": d, "score": score, "pred": score >= np.float32(t), "true": is_true, "pos": pos, "country": countries,
            "code": code, "t": t, "model": str(mp)}


def categories(d, idx, country_of_pair, owner_other):
    def share(mask):
        return float(mask.mean()) if len(mask) else float("nan")

    n1 = d["name_shared_tokens"][idx]
    ats = d["addr_token_set"][idx]
    flags = d["pass_flags"][idx]
    nt_bit = np.uint8(1 << PASS_ORDER.index("name_token"))
    return {
        "pairs": int(len(idx)),
        "india_share": share(country_of_pair[idx] == "india"),
        "us_share": share(country_of_pair[idx] == "us"),
        "shared_name_tokens_1": share(n1 == 1),
        "shared_name_tokens_2plus": share(n1 >= 2),
        "same_core_name": share(d["name_core_eq"][idx] > 0),
        "address_strong(token_set>=0.8)": share(ats >= 0.8),
        "address_weak(token_set<0.5)": share(ats < 0.5),
        "address_empty": share(d["addr_any_empty"][idx] > 0),
        "num_conflict": share(d["num_conflict"][idx] > 0),
        "legal_form_conflict": share(d["legal_form_conflict"][idx] > 0),
        "found_only_by_name_token": share(flags == nt_bit),
        "pool_record_owned_by_other_S1": share(owner_other[idx]) if owner_other is not None else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_b03a_name_token.yaml")
    ap.add_argument("--base", required=True, help="RUN:val_tag")
    ap.add_argument("--new", required=True, help="RUN:val_tag")
    ap.add_argument("--out", default="experiments/runs/EXP-B-03a/candidate_comparison.json")
    a = ap.parse_args()
    configure_arrow()
    cfg = load_config(a.config)
    paths = Paths(cfg)
    val = load_val_ids(paths)
    B = scored(paths, *a.base.split(":"), val)
    N = scored(paths, *a.new.split(":"), val)

    # which pool records belong (in the ground truth) to SOME S1 -> for "owned by another S1"
    gt = pq.read_table(paths.interim_file("train", "ground_truth"))
    flat = pc.list_flatten(pc.split_pattern(gt.column("matched_entity_ids"), ","))
    flat = pc.filter(flat, pc.not_equal(flat, ""))
    owned = np.zeros(len(pool_ids(paths, "train")), bool)
    owned[pc.index_in(flat, value_set=pool_ids(paths, "train")).to_numpy(zero_copy_only=False).astype(np.int64)] = True

    def pair_country(X):
        return X["country"][X["pos"]]

    in_base = np.isin(N["code"], B["code"])
    in_new = np.isin(B["code"], N["code"])
    new_only = np.flatnonzero(~in_base)
    dropped = np.flatnonzero(~in_new)
    Nown = owned[N["d"]["pool_row"].astype(np.int64)] & ~N["true"]
    Nc = pair_country(N)

    rec = new_only[N["pred"][new_only] & N["true"][new_only]]
    newfp = new_only[N["pred"][new_only] & ~N["true"][new_only]]
    # common pairs: prediction flips caused by changed blocking inputs
    common_new = np.flatnonzero(in_base)
    order_b = np.argsort(B["code"])
    j = order_b[np.searchsorted(B["code"][order_b], N["code"][common_new])]
    flip_on = common_new[N["pred"][common_new] & ~B["pred"][j]]
    flip_off = common_new[~N["pred"][common_new] & B["pred"][j]]
    rep = {
        "base": a.base, "new": a.new, "model_base": B["model"], "model_new": N["model"], "threshold": N["t"],
        "candidates": {"base": int(len(B["code"])), "new": int(len(N["code"])), "new_only": int(len(new_only)), "dropped_by_topk": int(len(dropped))},
        "true_pairs_in_candidates": {"base": int(B["true"].sum()), "new": int(N["true"].sum()),
                                     "new_only_true": int(N["true"][new_only].sum()), "dropped_true": int(B["true"][dropped].sum())},
        "predictions": {
            "new_only_accepted_true (recovered TPs)": int(len(rec)),
            "new_only_accepted_false (new FPs)": int(len(newfp)),
            "new_only_true_rejected_by_model": int((N["true"][new_only] & ~N["pred"][new_only]).sum()),
            "dropped_accepted_true (TPs lost to top-K)": int((B["pred"][dropped] & B["true"][dropped]).sum()),
            "dropped_accepted_false (FPs removed)": int((B["pred"][dropped] & ~B["true"][dropped]).sum()),
            "common_flipped_to_accept_true": int(N["true"][flip_on].sum()),
            "common_flipped_to_accept_false": int((~N["true"][flip_on]).sum()),
            "common_flipped_to_reject_true": int(N["true"][flip_off].sum()),
            "common_flipped_to_reject_false": int((~N["true"][flip_off]).sum()),
        },
        "new_only_candidates_rejection_rate": float(1 - N["pred"][new_only].mean()) if len(new_only) else None,
        "recovered_true_pairs_profile": categories(N["d"], rec, Nc, Nown),
        "new_false_merges_profile": categories(N["d"], newfp, Nc, Nown),
        "new_only_false_candidates_profile": categories(N["d"], new_only[~N["true"][new_only]], Nc, Nown),
    }
    p = rep["predictions"]
    rep["net_true_matches"] = (p["new_only_accepted_true (recovered TPs)"] - p["dropped_accepted_true (TPs lost to top-K)"]
                               + p["common_flipped_to_accept_true"] - p["common_flipped_to_reject_true"])
    rep["net_false_merges"] = (p["new_only_accepted_false (new FPs)"] - p["dropped_accepted_false (FPs removed)"]
                               + p["common_flipped_to_accept_false"] - p["common_flipped_to_reject_false"])
    # entity-level F0.5: base, new, and the counterfactual "new candidates only" (common pairs keep
    # their base prediction -> isolates candidate recovery from the cheap_score/rank input shift)
    from ber.evaluation.metrics import evaluate_arrays

    m = len(val)
    n_true = B_n_true = None
    _, _, n_true, countries = label_pairs(paths, val, np.zeros(0, np.int64), np.zeros(0, np.int64))

    def ent(pos_arr, pred_mask, true_mask):
        return evaluate_arrays(np.bincount(pos_arr[pred_mask & true_mask], minlength=m), np.bincount(pos_arr[pred_mask], minlength=m),
                               n_true, countries=countries)

    cf_pos = np.concatenate([B["pos"][in_new], N["pos"][new_only]])
    cf_pred = np.concatenate([B["pred"][in_new], N["pred"][new_only]])
    cf_true = np.concatenate([B["true"][in_new], N["true"][new_only]])
    keep = ("f05", "macro_precision", "macro_recall", "false_merges", "singleton_false_positive", "tp_pairs")
    rep["entity_level"] = {}
    for name, r in (("base", ent(B["pos"], B["pred"], B["true"])), ("new", ent(N["pos"], N["pred"], N["true"])),
                    ("counterfactual_new_candidates_only", ent(cf_pos, cf_pred, cf_true))):
        rep["entity_level"][name] = {**{k: r[k] for k in keep}, "by_country": {c: {k: v[k] for k in keep[:3]} for c, v in r["by_country"].items()}}
        print("ENTITY", name, {k: round(r[k], 4) if isinstance(r[k], float) else r[k] for k in keep},
              {c: round(v["f05"], 4) for c, v in r["by_country"].items()})
    write_json(Path(a.out), rep)
    print(json.dumps({k: v for k, v in rep.items() if "profile" not in k}, indent=1))
    for k in ("recovered_true_pairs_profile", "new_false_merges_profile", "new_only_false_candidates_profile"):
        print(k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in rep[k].items()})


if __name__ == "__main__":
    main()
