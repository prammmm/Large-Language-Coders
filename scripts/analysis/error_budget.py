"""Error budget (READ-ONLY diagnostic): how many validation F0.5 points each error type
costs, measured as the gain from an ORACLE fix of that bucket alone (everything else as is).

Buckets
  missed, scored below threshold (true candidate pairs with score < t):
      script_mismatch | address_empty | name_weak_address_strong | name_strong_address_weak | other
  never a candidate (blocking miss; true pairs not in the top-K list):
      address_empty | script_mismatch | shares_name_token | other
  false merges (predicted, not true):
      on_singleton | num_conflict | legal_form_conflict | same_core_name | other
Fixing a missed bucket = predicting those pairs; fixing a false-merge bucket = removing them.

    python scripts/analysis/error_budget.py --config configs/exp_f01b_conflicts_zeros.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_exp_b01 import load_views  # noqa: E402

from ber.config import Paths, load_config, parse_sets  # noqa: E402
from ber.data.normalize_records import load_manifest  # noqa: E402
from ber.data.split import load_val_ids  # noqa: E402
from ber.evaluation.metrics import f05_from_counts  # noqa: E402
from ber.pipeline.run_baseline import _ids, _read_parts, label_pairs, pool_ids, scored_dir  # noqa: E402
from ber.models import make_scorer  # noqa: E402
from ber.utils.runtime import configure_arrow, read_json, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_f01b_conflicts_zeros.yaml")
    ap.add_argument("--set", dest="sets", action="append", default=[])
    a = ap.parse_args()
    configure_arrow()
    cfg = load_config(a.config, parse_sets(a.sets))
    paths = Paths(cfg)
    exp = cfg["experiment"]["id"]
    t = float(read_json(paths.run_dir(exp) / "metrics.json")["threshold_used"])
    val = load_val_ids(paths)
    m = len(val)
    scorer = make_scorer(cfg, paths)
    need = [c for c in getattr(scorer, "required_features", []) if c != "cheap_rank"]
    extra = ["name_core_jaccard", "addr_token_set", "name_ratio", "addr_any_empty", "num_conflict", "legal_form_conflict", "name_core_eq"]
    cols = list(dict.fromkeys(["s1_row", "pool_row", "cheap_rank", "score"] + need + extra))
    d = _read_parts(scored_dir(cfg, paths, "train", "eval_read"), cols)
    score = scorer.score(d)
    is_true, pos, n_true, countries = label_pairs(paths, val, d["s1_row"], d["pool_row"])
    pred = score >= np.float32(t)
    n_s2 = int(load_manifest(paths, "train")["source2"]["rows"])

    # ---- true pairs never in the candidate list
    s1_ids, p_ids = _ids(paths, "train", "source1"), pool_ids(paths, "train")
    gt = pq.read_table(paths.interim_file("train", "ground_truth"))
    val_arr = pa.array(val, pa.string())
    gt = gt.filter(pc.is_in(gt.column("source1_entity_id"), value_set=val_arr))
    L = pc.split_pattern(gt.column("matched_entity_ids"), ",")
    flat, par = pc.list_flatten(L), pc.list_parent_indices(L)
    keep = pc.not_equal(flat, "")
    g_s1 = pc.take(gt.column("source1_entity_id"), pc.filter(par, keep))
    g_pos = pc.index_in(g_s1, value_set=val_arr).to_numpy(zero_copy_only=False).astype(np.int64)
    g_srow = pc.index_in(g_s1, value_set=s1_ids).to_numpy(zero_copy_only=False).astype(np.int64)
    g_prow = pc.index_in(pc.filter(flat, keep), value_set=p_ids).to_numpy(zero_copy_only=False).astype(np.int64)
    shift = np.int64(1) << np.int64(24)
    cand_code = d["s1_row"].astype(np.int64) * shift + d["pool_row"].astype(np.int64)
    g_code = g_srow * shift + g_prow
    missing = ~np.isin(g_code, cand_code)

    # ---- descriptors needing views (script, name tokens, address emptiness)
    fn = is_true & ~pred
    need_pool = np.unique(np.concatenate([g_prow[missing], d["pool_row"][fn].astype(np.int64)]))
    v1 = load_views(paths, "train", "source1", np.unique(np.concatenate([g_srow, d["s1_row"].astype(np.int64)])))
    v2 = load_views(paths, "train", "source2", need_pool[need_pool < n_s2])
    v3 = load_views(paths, "train", "source3", need_pool[need_pool >= n_s2] - n_s2)

    def pv(p):
        return v2.get(int(p)) if p < n_s2 else v3.get(int(p - n_s2))

    def script_mm(sr, pr):
        a_, b_ = v1.get(int(sr)), pv(pr)
        return bool(a_ and b_ and a_[2] == 0 and b_[2] in (1, 2))

    # ---- baseline counts
    k0 = np.bincount(pos[pred], minlength=m).astype(np.int64)
    tp0 = np.bincount(pos[pred & is_true], minlength=m).astype(np.int64)
    base = float(f05_from_counts(tp0, k0, n_true).mean())

    def gain(add_pos=None, remove_mask=None, country=None):
        k, tp = k0.copy(), tp0.copy()
        if add_pos is not None and len(add_pos):
            np.add.at(k, add_pos, 1)
            np.add.at(tp, add_pos, 1)
        if remove_mask is not None and remove_mask.any():
            np.subtract.at(k, pos[remove_mask], 1)
        f = f05_from_counts(tp, k, n_true)
        f0 = f05_from_counts(tp0, k0, n_true)
        if country is None:
            return float(f.mean() - f0.mean())
        sel = countries == country
        return float(f[sel].mean() - f0[sel].mean())

    out = {"experiment": exp, "threshold": t, "val_f05": base, "buckets": {}}
    ctry = sorted(set(countries.tolist()))

    def record(name, n_pairs, **kw):
        out["buckets"][name] = {"pairs": int(n_pairs), "gain_f05": gain(**kw),
                                **{f"gain_{c}": gain(country=c, **kw) for c in ctry}}

    # missed but scored (FN among candidates)
    fn_idx = np.flatnonzero(fn)
    sm = np.array([script_mm(d["s1_row"][i], d["pool_row"][i]) for i in fn_idx], bool)
    ae = d["addr_any_empty"][fn_idx] > 0
    nw = (d["name_core_jaccard"][fn_idx] == 0) & (d["addr_token_set"][fn_idx] >= 0.8)
    ns = (d["name_ratio"][fn_idx] >= 0.8) & (d["addr_token_set"][fn_idx] < 0.6)
    cats = np.where(sm, "script_mismatch", np.where(ae, "address_empty", np.where(nw, "name_weak_address_strong",
                    np.where(ns, "name_strong_address_weak", "other"))))
    for c in ["script_mismatch", "address_empty", "name_weak_address_strong", "name_strong_address_weak", "other"]:
        sel = fn_idx[cats == c]
        record(f"missed_scored_low:{c}", len(sel), add_pos=pos[sel])
    record("missed_scored_low:ALL", len(fn_idx), add_pos=pos[fn_idx])

    # never a candidate (blocking)
    mi = np.flatnonzero(missing)
    bcat = []
    for i in mi:
        a_, b_ = v1.get(int(g_srow[i])), pv(g_prow[i])
        if not a_ or not b_:
            bcat.append("other")
        elif a_[3] == "" or b_[3] == "":
            bcat.append("address_empty")
        elif a_[2] == 0 and b_[2] in (1, 2):
            bcat.append("script_mismatch")
        elif set(a_[1].split()) & set(b_[1].split()):
            bcat.append("shares_name_token")
        else:
            bcat.append("other")
    bcat = np.array(bcat)
    for c in ["address_empty", "script_mismatch", "shares_name_token", "other"]:
        sel = mi[bcat == c]
        record(f"not_blocked:{c}", len(sel), add_pos=g_pos[sel])
    record("not_blocked:ALL", len(mi), add_pos=g_pos[mi])

    # false merges
    fp = pred & ~is_true
    on_single = fp & (n_true[pos] == 0)
    numc = fp & ~on_single & (d["num_conflict"] > 0)
    legal = fp & ~on_single & ~numc & (d["legal_form_conflict"] > 0)
    same = fp & ~on_single & ~numc & ~legal & (d["name_core_eq"] > 0)
    other = fp & ~on_single & ~numc & ~legal & ~same
    for name, msk in [("on_singleton", on_single), ("num_conflict", numc), ("legal_form_conflict", legal),
                      ("same_core_name", same), ("other", other), ("ALL", fp)]:
        record(f"false_merge:{name}", msk.sum(), remove_mask=msk)

    # entity-level: n=1 entities, empty-on-positive
    out["entities_exactly_one_true_match"] = {"count": int((n_true == 1).sum()), "f05": float(f05_from_counts(tp0, k0, n_true)[n_true == 1].mean())}
    out["pairs_score_band_0.45_0.60_true_share"] = float(is_true[(score >= 0.45) & (score < t)].mean())
    write_json(paths.run_dir(exp) / "analysis" / "error_budget.json", out)
    print(f"val F0.5 {base:.4f} at t={t}")
    for k2, v in sorted(out["buckets"].items(), key=lambda x: -x[1]["gain_f05"]):
        print(f"{v['gain_f05']:+.4f}  pairs {v['pairs']:>7}  " + "  ".join(f"{c} {v['gain_' + c]:+.4f}" for c in ctry) + f"  {k2}")


if __name__ == "__main__":
    main()
