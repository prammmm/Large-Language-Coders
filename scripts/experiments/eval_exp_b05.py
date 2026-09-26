"""EXP-B-05 evaluation: B-04 + D-01 (control) vs B-05 + D-01 under FULL competition.

Each arm = (all-train scored parts of that blocker+model, its model threshold). All 2,206,821 train
S1 compete; the primary population E = train S1 not in the (shared) 200k fit set. The D-01 rule is
the production function ``best_claimant_keep`` (margin 0). Arms are processed one at a time and
reduced to per-S1 count vectors, so memory stays bounded.

Also: validation false-merge categories (stored validation features of each arm, restricted to the
pairs D-01 keeps under full competition), match-count buckets, and UNLABELED test diagnostics.

    python scripts/experiments/eval_exp_b05.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_exp_d01 import half_of  # noqa: E402
from full_competition import SHIFT, gt_codes  # noqa: E402

from ber.config import Paths, load_config  # noqa: E402
from ber.data.normalize_records import load_manifest, read_country  # noqa: E402
from ber.data.split import load_val_ids  # noqa: E402
from ber.decision.competition import best_claimant_keep, claim_stats  # noqa: E402
from ber.evaluation.metrics import evaluate_arrays  # noqa: E402
from ber.pipeline.run_baseline import _ids, _read_parts, fit_s1_subset  # noqa: E402
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, write_json  # noqa: E402

KEYS = ("entities", "f05", "macro_precision", "macro_recall", "false_merges", "singleton_false_positive", "empty_on_positive",
        "predicted_pairs", "tp_pairs", "predicted_per_s1", "blocking_recall", "oracle_f05")
FM_FEATS = ["name_core_eq", "name_core_jaccard", "addr_token_set", "addr_num_shared", "num_conflict", "legal_form_conflict", "addr_any_empty"]
SCORE_BINS = np.array([0, 0.1, 0.3, 0.5, 0.67, 0.8, 0.9, 0.95, 0.99, 1.0001], np.float32)


def threshold_of(model_file: Path) -> float:
    return float(json.loads(model_file.read_text(encoding="utf-8"))["meta"]["threshold"])


def report(tp, k, n, c_tp, c_n, countries, mask):
    ids = np.flatnonzero(mask)
    r = evaluate_arrays(tp[ids], k[ids], n[ids], c_tp=c_tp[ids], c_n=c_n[ids], countries=countries[ids])
    out = {x: r.get(x) for x in KEYS}
    out["candidates_per_s1"] = float(c_n[ids].mean())
    out["by_country"] = {c: {**{x: v.get(x) for x in KEYS}, "candidates_per_s1": float(c_n[ids][countries[ids] == c].mean())}
                         for c, v in r["by_country"].items()}
    out["by_match_count"] = {b: {x: v.get(x) for x in ("entities", "f05", "macro_precision", "macro_recall", "blocking_recall")}
                             for b, v in r["by_match_count"].items()}
    return out


def script_arrays(paths, split):
    """name_script per S1 row and per pool row (0 Latin, 1 mixed, 2 non-Latin, 3 no letters)."""
    man = load_manifest(paths, split)
    n2 = int(man["source2"]["rows"])
    s1s = np.full(int(man["source1"]["rows"]), 3, np.int8)
    ps = np.full(n2 + int(man["source3"]["rows"]), 3, np.int8)
    for src, arr, off in (("source1", s1s, 0), ("source2", ps, 0), ("source3", ps, n2)):
        for k in man[src]["countries"]:
            t = read_country(paths, split, src, k, ["row", "name_script"])
            arr[t.column("row").to_numpy().astype(np.int64) + off] = t.column("name_script").to_numpy()
    return s1s, ps


def is_cross_script(s1s, ps, s1, pool):
    """Script-mismatch pair: Latin S1 name vs non-Latin / mixed pool name."""
    return (s1s[s1] == 0) & ((ps[pool] == 1) | (ps[pool] == 2))


def arm_counts(paths, tag, t, gtc, n_rows, is_val, scripts=None):
    """Per-S1 count vectors for unrestricted and D-01 predictions, plus competition + FM codes."""
    d = _read_parts(paths.scored_dir("train", tag), ["s1_row", "pool_row", "score"])
    s1, pool, score = d["s1_row"].astype(np.int64), d["pool_row"].astype(np.int64), d["score"].astype(np.float32)
    del d
    code = s1 * SHIFT + pool
    true = np.isin(code, gtc)
    pred = score >= np.float32(t)
    keep = best_claimant_keep(pool, s1, score, pred)
    bc = lambda m: np.bincount(s1[m], minlength=n_rows)  # noqa: E731
    out = {"c_n": bc(np.ones(len(s1), bool)), "c_tp": bc(true), "k_u": bc(pred), "tp_u": bc(pred & true),
           "k_d": bc(keep), "tp_d": bc(keep & true), "pairs": int(len(s1)),
           "competition": {"before": claim_stats(pool, s1, pred), "after": claim_stats(pool, s1, keep),
                           "removed": int((pred & ~keep).sum()), "true_removed": int((pred & ~keep & true).sum())}}
    if scripts is not None:  # cross-script (script-mismatch) accounting
        sm = is_cross_script(scripts[0], scripts[1], s1, pool)
        out.update({"sm_c_n": bc(sm), "sm_c_tp": bc(sm & true), "sm_tp_u": bc(sm & pred & true), "sm_tp_d": bc(sm & keep & true),
                    "sm_fp_d": bc(sm & keep & ~true)})
    vsel = is_val[s1]
    out["val_fm_codes"] = {"unrestricted": np.sort(code[vsel & pred & ~true]), "d01": np.sort(code[vsel & keep & ~true])}
    out["val_cand_codes"] = np.sort(code[vsel])
    return out


def fm_categories(paths, tag, codes, n_true_of_row, other_cand_codes):
    """Categorise validation false-merge pairs (codes) using the arm's stored validation features."""
    d = _read_parts(paths.scored_dir("train", tag), ["s1_row", "pool_row"] + FM_FEATS)
    code = d["s1_row"].astype(np.int64) * SHIFT + d["pool_row"].astype(np.int64)
    sel = np.isin(code, codes)
    f = {k: v[sel] for k, v in d.items()}
    c = code[sel]
    if len(c) != len(codes):
        LOG.warning("false-merge codes not all found in stored features (%d of %d)", len(c), len(codes))
    singleton = n_true_of_row[f["s1_row"].astype(np.int64)] == 0
    same_name = f["name_core_eq"] >= 1
    addr_driven = (f["name_core_jaccard"] < 0.34) & ~same_name
    num_c = f["num_conflict"] >= 1
    legal = f["legal_form_conflict"] >= 1
    cats = {"singleton_s1": singleton, "same_core_name": same_name & ~singleton, "address_driven(name_jacc<0.34)": addr_driven & ~singleton,
            "house_number_conflict": num_c & ~singleton, "legal_form_conflict": legal & ~singleton,
            "address_empty": (f["addr_any_empty"] >= 1) & ~singleton}
    other = ~(singleton | same_name | addr_driven | num_c | legal | (f["addr_any_empty"] >= 1))
    new = ~np.isin(c, other_cand_codes) if other_cand_codes is not None else np.zeros(len(c), bool)
    return {"false_merges": int(len(c)), **{k: int(v.sum()) for k, v in cats.items()}, "other": int(other.sum()),
            "not_a_candidate_in_other_arm (new hard negatives)": int(new.sum()),
            "new_hard_negatives_by_category": {k: int((v & new).sum()) for k, v in cats.items()} | {"other": int((other & new).sum())}}


def test_diag(paths, tag, t, scripts=None):
    d = _read_parts(paths.scored_dir("test", tag), ["s1_row", "pool_row", "score"])
    s1, pool, score = d["s1_row"].astype(np.int64), d["pool_row"].astype(np.int64), d["score"].astype(np.float32)
    del d
    pred = score >= np.float32(t)
    keep = best_claimant_keep(pool, s1, score, pred)
    man = load_manifest(paths, "test")
    n1 = int(man["source1"]["rows"])
    ctry = np.empty(n1, dtype=object)
    for k in man["source1"]["countries"]:
        ctry[read_country(paths, "test", "source1", k, ["row"]).column("row").to_numpy()] = k
    out = {"label": "UNLABELED TEST DIAGNOSTICS", "pairs": int(len(s1))}
    for c in ["all"] + sorted(set(ctry.tolist())):
        se = np.ones(n1, bool) if c == "all" else ctry == c
        sp = se[s1]
        cn = np.bincount(s1[sp], minlength=n1)[se]
        ku = np.bincount(s1[sp & pred], minlength=n1)[se]
        kd = np.bincount(s1[sp & keep], minlength=n1)[se]
        h, _ = np.histogram(score[sp], SCORE_BINS)
        if scripts is not None:
            smp = is_cross_script(scripts[0], scripts[1], s1, pool) & sp
            out.setdefault("cross_script", {})[c] = {"candidates": int(smp.sum()), "predicted_d01": int((smp & keep).sum()),
                                                     "s1_with_cross_script_prediction": int(len(np.unique(s1[smp & keep])))}
        out[c] = {"s1": int(se.sum()), "candidates_per_s1": float(cn.mean()), "candidates_p99": float(np.percentile(cn, 99)),
                  "predicted_per_s1_unrestricted": float(ku.mean()), "predicted_per_s1_d01": float(kd.mean()),
                  "empty_rate_unrestricted": float((ku == 0).mean()), "empty_rate_d01": float((kd == 0).mean()),
                  "conflicts_before": claim_stats(pool[sp], s1[sp], pred[sp]), "removed_by_d01": int((sp & pred & ~keep).sum()),
                  "conflicts_after": claim_stats(pool[sp], s1[sp], keep[sp])["conflicted_records"],
                  "candidate_score_hist_share": {f"[{SCORE_BINS[i]:g},{SCORE_BINS[i + 1]:g})": float(h[i] / max(h.sum(), 1)) for i in range(len(h))},
                  "predicted_count_hist_d01": {str(b): float((kd == b).mean()) for b in range(6)} | {"6-10": float(((kd >= 6) & (kd <= 10)).mean()), "11+": float((kd >= 11).mean())}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default="configs/exp_b04_blocking_v2.yaml")
    ap.add_argument("--control-tag", default="forensic_alltrain_B04")
    ap.add_argument("--exp", default="configs/exp_b05_addr_window5.yaml")
    ap.add_argument("--exp-tag", default="forensic_alltrain_B05")
    ap.add_argument("--out", default="experiments/runs/EXP-B-05/full_competition_comparison.json")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--control-label", default="B04", help="name of the control arm in the output")
    ap.add_argument("--label", default="B05", help="name of the experiment arm in the output")
    ap.add_argument("--script-analysis", action="store_true", help="cross-script (script-mismatch) accounting")
    a = ap.parse_args()
    X = a.label
    K0 = a.control_label
    setup_logging()
    configure_arrow()
    cfgs = {K0: load_config(a.control), X: load_config(a.exp)}
    paths = Paths(cfgs[K0])
    tags = {K0: a.control_tag, X: a.exp_tag}
    models = {k: Paths(c).run_dir(c["experiment"]["id"]) / "model.json" for k, c in cfgs.items()}
    ts = {k: threshold_of(m) for k, m in models.items()}
    f04, f05 = fit_s1_subset(cfgs[K0], paths), fit_s1_subset(cfgs[X], paths)
    if f04 != f05:
        raise RuntimeError("the two arms were fitted on different train-fold S1 sets")
    gtc, n_true, n_rows = gt_codes(paths)
    s1_ids = _ids(paths, "train", "source1")
    idx = lambda ids: pc.index_in(pa.array(ids, pa.string()), value_set=s1_ids).to_numpy(zero_copy_only=False).astype(np.int64)  # noqa: E731
    is_fit = np.zeros(n_rows, bool)
    is_fit[idx(f04)] = True
    is_val = np.zeros(n_rows, bool)
    is_val[idx(load_val_ids(paths))] = True
    if (is_fit & is_val).any():
        raise RuntimeError("leakage: fit and validation S1 overlap")
    E = ~is_fit
    half = half_of(s1_ids.to_pylist())
    countries = np.empty(n_rows, dtype=object)
    for k in load_manifest(paths, "train")["source1"]["countries"]:
        countries[read_country(paths, "train", "source1", k, ["row"]).column("row").to_numpy()] = k
    pops = {"E": E, "validation": is_val, "validation_half_A": is_val & (half == 0), "validation_half_B": is_val & (half == 1),
            "E_half_A": E & (half == 0), "E_half_B": E & (half == 1)}
    rep = {"arms": {k: {"config": getattr(a, "control" if k == K0 else "exp"), "scored_tag": tags[k], "model": str(models[k]),
                        "threshold": ts[k]} for k in cfgs},
           "populations": {k: int(v.sum()) for k, v in pops.items()}, "rule": "best claimant per pool record, margin 0, tie -> lowest s1_row"}
    arms = {}
    scripts = script_arrays(paths, "train") if a.script_analysis else None
    with PeakMemory() as pm:
        for k in (K0, X):
            LOG.info("arm %s: %s", k, tags[k])
            arms[k] = arm_counts(paths, tags[k], ts[k], gtc, n_rows, is_val, scripts)
    rep["eval_peak_rss_gb"] = round(pm.peak_gb, 2)
    res = {}
    for pname, m in pops.items():
        res[pname] = {}
        for k, v in arms.items():
            res[pname][f"{k}_unrestricted"] = report(v["tp_u"], v["k_u"], n_true, v["c_tp"], v["c_n"], countries, m)
            res[pname][f"{k}_D01"] = report(v["tp_d"], v["k_d"], n_true, v["c_tp"], v["c_n"], countries, m)
    rep["results"] = res
    if scripts is not None:
        gs, gp = gtc // SHIFT, gtc % SHIFT
        gsm = np.bincount(gs[is_cross_script(scripts[0], scripts[1], gs, gp)], minlength=n_rows)
        rep["cross_script"] = {}
        for pname, m in pops.items():
            rep["cross_script"][pname] = {"true_pairs_cross_script": int(gsm[m].sum())}
            for k, v in arms.items():
                rep["cross_script"][pname][k] = {x: int(v[x][m].sum()) for x in ("sm_c_n", "sm_c_tp", "sm_tp_u", "sm_tp_d", "sm_fp_d")}
    rep["competition_all_train"] = {k: {**v["competition"], "pairs": v["pairs"]} for k, v in arms.items()}
    rep["validation_false_merges_d01"] = {
        K0: fm_categories(paths, cfgs[K0]["scoring"]["tag"], arms[K0]["val_fm_codes"]["d01"], n_true, None),
        X: fm_categories(paths, cfgs[X]["scoring"]["tag"], arms[X]["val_fm_codes"]["d01"], n_true, arms[K0]["val_cand_codes"]),
    }
    del arms
    if not a.skip_test:
        tscr = script_arrays(paths, "test") if a.script_analysis else None
        rep["test_unlabeled_diagnostics"] = {K0: test_diag(paths, cfgs[K0]["scoring"]["tag"], ts[K0], tscr),
                                             X: test_diag(paths, cfgs[X]["scoring"]["tag"], ts[X], tscr)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(a.out), rep)
    for pname in pops:
        r = res[pname]
        LOG.info("%-18s %s+D01 %.4f (P %.4f R %.4f) | %s+D01 %.4f (P %.4f R %.4f) | delta %+.4f | %s unrestr %.4f", pname, K0,
                 r[f"{K0}_D01"]["f05"], r[f"{K0}_D01"]["macro_precision"], r[f"{K0}_D01"]["macro_recall"], X,
                 r[f"{X}_D01"]["f05"], r[f"{X}_D01"]["macro_precision"], r[f"{X}_D01"]["macro_recall"],
                 r[f"{X}_D01"]["f05"] - r[f"{K0}_D01"]["f05"], X, r[f"{X}_unrestricted"]["f05"])


if __name__ == "__main__":
    main()
