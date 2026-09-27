"""Scorer + test-side forensics for one run (READ-ONLY diagnostic; no production change).

VALIDATION (labels; stored features re-scored with the run's model, exactly as `evaluate`):
  * score distribution of true / false candidate pairs by country; pair precision per score bin
  * per-S1 top-1 / top-2 / margin for positive S1s vs singletons (label-free statistics, so they are
    directly comparable with the test side)
  * match-cardinality table (n_true bucket -> F0.5, P, R, predicted vs true count)
  * low-scored true pairs and high-scored false pairs split into feature-defined categories
  * which blocking pass produced accepted true / false pairs
TEST (no labels): per country S1s, candidates/S1, predicted/S1, empty rate, top-1/top-2/margin,
  share of S1s with >=1 prediction, pass composition, score histogram — France vs India/US and
  test vs validation (score-space distribution shift).

    python scripts/analysis/scorer_test_forensics.py --config configs/exp_b04_blocking_v2.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ber.blocking.blocker import PASS_ORDER
from ber.config import Paths, load_config
from ber.data.normalize_records import load_manifest, read_country
from ber.data.split import load_val_ids
from ber.evaluation.metrics import evaluate_arrays
from ber.models import make_scorer, scorer_features
from ber.pipeline.run_baseline import _read_parts, label_pairs, scored_dir
from ber.utils.runtime import LOG, configure_arrow, setup_logging, write_json

BINS = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.67, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001], np.float32)


def q(x):
    if len(x) == 0:
        return None
    p = np.percentile(x, [5, 25, 50, 75, 95])
    return {"n": int(len(x)), "mean": float(x.mean()), "p05": float(p[0]), "p25": float(p[1]), "p50": float(p[2]), "p75": float(p[3]), "p95": float(p[4])}


def top12(group, score, m):
    """Per-group top-1, top-2 score (−1 if absent). group: int per pair in [0, m)."""
    o = np.lexsort((-score, group))
    g, s = group[o], score[o]
    first = np.ones(len(g), bool)
    first[1:] = g[1:] != g[:-1]
    t1 = np.full(m, -1.0, np.float32)
    t1[g[first]] = s[first]
    second = np.zeros(len(g), bool)
    second[1:] = (~first[1:]) & first[:-1]
    t2 = np.full(m, -1.0, np.float32)
    t2[g[second]] = s[second]
    return t1, t2


def label_free(t1, t2, k, c_n, t):
    has = c_n > 0
    marg = np.where(t2 >= 0, t1 - t2, np.nan)
    return {"s1": int(len(t1)), "candidates_per_s1": float(c_n.mean()), "predicted_per_s1": float(k.mean()),
            "empty_prediction_rate": float((k == 0).mean()), "no_candidate_rate": float((~has).mean()),
            "top1": q(t1[has]), "top2": q(t2[t2 >= 0]),
            "margin_top1_top2_when_both_above_t": q(marg[(t2 >= t)]),
            "share_top1_ge_t": float((t1 >= t).mean()), "share_top2_ge_t": float((t2 >= t).mean()),
            "share_top1_in_[t-0.15,t)": float(((t1 >= t - 0.15) & (t1 < t)).mean()),
            "predicted_count_hist": {str(b): float((k == b).mean()) for b in range(0, 6)} | {"6-10": float(((k >= 6) & (k <= 10)).mean()), "11+": float((k >= 11).mean())}}


def pass_share(flags, sel):
    f = flags[sel]
    return {p: float(((f >> i) & 1).astype(bool).mean()) if len(f) else None for i, p in enumerate(PASS_ORDER)} | \
           {"only_new_v2_passes": float(((f & 0b00111111) == 0).mean()) if len(f) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_b04_blocking_v2.yaml")
    ap.add_argument("--out", default="experiments/runs/POST-0.91-FORENSIC/scorer_test_forensics.json")
    ap.add_argument("--skip-test", action="store_true")
    a = ap.parse_args()
    setup_logging()
    configure_arrow()
    cfg = load_config(a.config)
    paths = Paths(cfg)
    exp = cfg["experiment"]["id"]
    t = float(json.loads((paths.run_dir(exp) / "model.json").read_text(encoding="utf-8"))["meta"]["threshold"])
    scorer = make_scorer(cfg, paths)
    need = [c for c in scorer_features(scorer) if c != "cheap_rank"]
    rep = {"config": a.config, "threshold": t}

    # ======================= VALIDATION
    val = load_val_ids(paths)
    m = len(val)
    d = _read_parts(scored_dir(cfg, paths, "train", "eval_read"), list(dict.fromkeys(["s1_row", "pool_row", "cheap_rank", "pass_flags"] + need)))
    score = scorer.score(d).astype(np.float32)
    is_true, pos, n_true, countries = label_pairs(paths, val, d["s1_row"], d["pool_row"])
    countries = np.asarray(countries, dtype=object)
    pred = score >= np.float32(t)
    k = np.bincount(pos[pred], minlength=m)
    tp = np.bincount(pos[pred & is_true], minlength=m)
    c_n = np.bincount(pos, minlength=m)
    r = evaluate_arrays(tp, k, n_true, countries=countries)
    rep["validation_check"] = {"f05": r["f05"], "P": r["macro_precision"], "R": r["macro_recall"]}
    pc_ = countries[pos]
    V = {}
    for c in sorted(set(countries.tolist())) + ["all"]:
        sp = np.ones(len(pos), bool) if c == "all" else pc_ == c
        se = np.ones(m, bool) if c == "all" else countries == c
        tr, fa = score[sp & is_true], score[sp & ~is_true]
        h_t, _ = np.histogram(tr, BINS)
        h_f, _ = np.histogram(fa, BINS)
        t1, t2 = top12(pos[sp], score[sp], m)
        V[c] = {"true_pair_scores": q(tr), "false_pair_scores": q(fa),
                "pair_precision_by_score_bin": {f"[{BINS[i]:.2f},{BINS[i+1]:.2f})": (float(h_t[i] / (h_t[i] + h_f[i])) if h_t[i] + h_f[i] else None, int(h_t[i] + h_f[i])) for i in range(len(BINS) - 1)},
                "label_free_positive_s1": label_free(t1[se & (n_true > 0)], t2[se & (n_true > 0)], k[se & (n_true > 0)], c_n[se & (n_true > 0)], t),
                "label_free_singleton_s1": label_free(t1[se & (n_true == 0)], t2[se & (n_true == 0)], k[se & (n_true == 0)], c_n[se & (n_true == 0)], t),
                "label_free_all_s1": label_free(t1[se], t2[se], k[se], c_n[se], t),
                "singleton_share": float((n_true[se] == 0).mean()),
                "accepted_true_pass_share": pass_share(d["pass_flags"], sp & pred & is_true),
                "accepted_false_pass_share": pass_share(d["pass_flags"], sp & pred & ~is_true)}
    rep["validation"] = V

    # match cardinality (per n_true bucket; predicted vs true count)
    card = {}
    for name, lo, hi in (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6-10", 6, 10), ("11+", 11, 10**9)):
        for c in sorted(set(countries.tolist())) + ["all"]:
            se = (n_true >= lo) & (n_true <= hi) & ((countries == c) if c != "all" else True)
            if not se.any():
                continue
            rr = evaluate_arrays(tp[se], k[se], n_true[se])
            card[f"{name}|{c}"] = {"entities": int(se.sum()), "f05": rr["f05"], "P": rr["macro_precision"], "R": rr["macro_recall"],
                                   "mean_true": float(n_true[se].mean()), "mean_candidates_true": float(np.bincount(pos[is_true], minlength=m)[se].mean()),
                                   "mean_predicted": float(k[se].mean()), "under_predict_share": float((k[se] < n_true[se]).mean()),
                                   "over_predict_share": float((k[se] > n_true[se]).mean())}
    rep["match_cardinality"] = card

    # categories: low-scored true pairs / high-scored false pairs
    f = {c: d[c] for c in need}
    cats = {
        "name_core_eq": f["name_core_eq"] >= 1,
        "name_core_jaccard<0.34": f["name_core_jaccard"] < 0.34,
        "name_core_jaccard>=0.99": f["name_core_jaccard"] >= 0.99,
        "addr_any_empty": f["addr_any_empty"] >= 1,
        "addr_token_set<0.5": f["addr_token_set"] < 0.5,
        "addr_token_set>=0.9": f["addr_token_set"] >= 0.9,
        "num_conflict": f["num_conflict"] >= 1,
        "no_shared_number": f["addr_num_shared"] < 1,
        "legal_form_conflict": f["legal_form_conflict"] >= 1,
        "cheap_rank>=10": d["cheap_rank"] >= 10,
    }
    low_true = is_true & ~pred
    high_false = pred & ~is_true
    rep["categories"] = {}
    for label, sel in (("low_scored_true (missed, in candidates)", low_true), ("accepted_false (false merges)", high_false), ("accepted_true", pred & is_true)):
        rep["categories"][label] = {"pairs": int(sel.sum()), "by_country_pairs": {c: int((sel & (pc_ == c)).sum()) for c in sorted(set(countries.tolist()))},
                                    "share": {n: float(v[sel].mean()) for n, v in cats.items()},
                                    "score": q(score[sel])}
    # false merges: on positive S1 vs singleton S1; is the false pair outranking a true one?
    fm = high_false
    rep["false_merge_breakdown"] = {"on_singleton_s1": int((fm & (n_true[pos] == 0)).sum()), "on_positive_s1": int((fm & (n_true[pos] > 0)).sum())}
    # "near-duplicate" false merges: name core equal AND address token set >= 0.9 (likely label noise / branch ambiguity)
    rep["false_merge_breakdown"]["name_eq_and_addr_ge_0.9"] = int((fm & cats["name_core_eq"] & cats["addr_token_set>=0.9"]).sum())
    rep["false_merge_breakdown"]["name_eq_addr_lt_0.5 (chain/branch)"] = int((fm & cats["name_core_eq"] & cats["addr_token_set<0.5"]).sum())
    rep["false_merge_breakdown"]["name_jacc_lt_0.34 (address-driven)"] = int((fm & cats["name_core_jaccard<0.34"]).sum())
    del d, f, score

    # ======================= TEST
    if not a.skip_test:
        tdir = scored_dir(cfg, paths, "test", "eval")
        td = _read_parts(tdir, ["s1_row", "score", "pass_flags"])
        s1r = td["s1_row"].astype(np.int64)
        man = load_manifest(paths, "test")
        n1 = int(man["source1"]["rows"])
        ctry = np.empty(n1, dtype=object)
        for key in man["source1"]["countries"]:
            ctry[read_country(paths, "test", "source1", key, ["row"]).column("row").to_numpy()] = key
        sc = td["score"].astype(np.float32)
        tpred = sc >= np.float32(t)
        tk = np.bincount(s1r[tpred], minlength=n1)
        tcn = np.bincount(s1r, minlength=n1)
        t1, t2 = top12(s1r, sc, n1)
        T = {}
        pc_t = ctry[s1r]
        for c in sorted(set(ctry.tolist())):
            se = ctry == c
            sp = pc_t == c
            h, _ = np.histogram(sc[sp], BINS)
            T[c] = {"label_free_all_s1": label_free(t1[se], t2[se], tk[se], tcn[se], t),
                    "candidate_score_hist_share": {f"[{BINS[i]:.2f},{BINS[i+1]:.2f})": float(h[i] / max(h.sum(), 1)) for i in range(len(BINS) - 1)},
                    "accepted_pass_share": pass_share(td["pass_flags"], sp & tpred),
                    "candidate_pass_share": pass_share(td["pass_flags"], sp)}
        rep["test"] = T
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(a.out), rep)
    LOG.info("validation check %s", rep["validation_check"])
    for c, v in rep.get("test", {}).items():
        lf = v["label_free_all_s1"]
        LOG.info("TEST %-7s cand/S1 %.2f pred/S1 %.2f empty %.3f top1>=t %.3f top2>=t %.3f", c, lf["candidates_per_s1"], lf["predicted_per_s1"],
                 lf["empty_prediction_rate"], lf["share_top1_ge_t"], lf["share_top2_ge_t"])
    for c, v in rep["validation"].items():
        lf = v["label_free_all_s1"]
        LOG.info("VAL  %-7s cand/S1 %.2f pred/S1 %.2f empty %.3f top1>=t %.3f top2>=t %.3f singleton %.3f", c, lf["candidates_per_s1"], lf["predicted_per_s1"],
                 lf["empty_prediction_rate"], lf["share_top1_ge_t"], lf["share_top2_ge_t"], v["singleton_share"])


if __name__ == "__main__":
    sys.exit(main())
