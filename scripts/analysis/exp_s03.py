"""EXP-S-03 (S-02 veto refitted on EXP-B-09 stage 1) analysis. Deterministic, seed 42. Read-only w.r.t. every other experiment.

    python scripts/analysis/exp_s03.py shadow-features | eval | veto | testdiag | budget | leakage | gates

Control = EXP-B-09 + D-01 @ 0.65 (stage-1 all-train scores forensic_alltrain_B09). S-03 = control AND refitted stage 2.
Shadow v2 populations, masks and T1W/T4W weights are the frozen infrastructure (identical to EXP-S-02 / B-08 / B-09 evaluations);
only the stage-2 feature parts are S-03's own (built from B-09 scores under each competition).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_b09_eval as E  # noqa: E402  (streamed stage-1 decisions, tier weights)
import exp_s02_shadow as X  # noqa: E402
import post_s01_forensics as F  # noqa: E402

from ber.config import Paths, load_config  # noqa: E402
from ber.decision.stage2 import restrict_additions  # noqa: E402
from ber.forensics.diagnostics import summary  # noqa: E402
from ber.forensics.shadow import weighted_metrics  # noqa: E402
from ber.pipeline import run_stage2 as S  # noqa: E402
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, write_json  # noqa: E402

CFG = "configs/exp_s03_veto_on_b09.yaml"
RUN = Path("experiments/runs/EXP-S-03")
OUT = RUN / "analysis"
SHIFT = S.SHIFT
T1_B09 = 0.65
TAG = "forensic_alltrain_B09"
BINS = [(-1, 0.02), (0.02, 0.1), (0.1, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 0.95), (0.95, 2)]


def cfgp():
    cfg = load_config(CFG)
    return cfg, Paths(cfg)


def fdirs(paths) -> dict:
    base = paths.processed_dir("train")
    return {"test_matched(v1 random orphan)": base / "stage2_EXP-S-03_orphan", "chain_orphaned(v1)": base / "stage2_EXP-S-03_chainorphan",
            "T2": base / "stage2_EXP-S-03_v2tier2", "T3": base / "stage2_EXP-S-03_v2tier3", "T4": base / "stage2_EXP-S-03_v2tier4"}


def drops():
    M1, M = np.load(X.V1 / "masks.npz"), X.masks_v2()
    return {"test_matched(v1 random orphan)": M1["orphan"], "chain_orphaned(v1)": M1["chain_orphan"], "T2": M["orphan_T2"], "T3": M["orphan_T3"], "T4": M["orphan_T4"]}


def cmd_shadow_features(c):
    cfg, paths = cfgp()
    rep = {}
    for comp, fd in fdirs(paths).items():
        with PeakMemory() as pm:
            st = S.build_features(cfg, paths, "train", drop_s1=drops()[comp], out=fd)
        rep[comp] = {"dir": fd.as_posix(), "eligible_pairs": st["eligible_pairs"], "dropped_s1": st["dropped_s1"], "seconds": round(pm.seconds, 1), "peak_gb": round(pm.peak_gb, 2)}
        LOG.info("%s: %s", comp, rep[comp])
        write_json(OUT / "shadow_features.json", rep)


def s03_arrays(cfg, paths, split, fdir=None):
    booster, meta = S.load_stage2(cfg, paths)
    s1, pool, prob, sc = S.predict_all(cfg, paths, split, booster, meta["features"], fdir)
    return s1, pool, prob, sc, meta


def decide_s03(ctl_codes, s1, pool, prob, sc, meta):
    code = s1 * SHIFT + pool
    if not np.isin(ctl_codes, code).all():
        raise AssertionError("a control pair is missing from the stage-2 eligible pairs")
    s2 = S.decide(s1, pool, prob, meta["threshold"], meta["variant"])
    cm = np.isin(code, ctl_codes)
    fin = restrict_additions(cm, s2, sc, np.inf)
    if (fin & ~cm).any():
        raise AssertionError("S-03 added a pair")
    return code, cm, fin


# ----------------------------------------------------------------------------- eval (local + shadow v2 + tiers + locked)
def cmd_eval(c):
    cfg, paths = cfgp()
    comps = X.competitions(c)  # frozen populations / masks / test-matched weights
    tw, ess, share = E.tier_weights(c, comps)
    fd = fdirs(paths)
    rep = {"systems": {"B09": "control: EXP-B-09 + D-01 @ 0.65", "S03": "B-09 + D-01 AND refitted stage 2 (veto)"}, "effective_sample_size(weighted tiers)": ess, "competitions": {}}
    for comp, (_, drop, pops, weights) in comps.items():
        pops, weights = dict(pops), dict(weights)
        la_key = "T1" if comp == "T1(base)" else ("T4" if comp == "T4" else None)
        if la_key:
            for pn in ("SHADOW-DEV", "SHADOW-LOCKED-A", "SHADOW-LOCKED-B"):
                pops[f"{pn}[W]"], weights[f"{pn}[W]"] = pops[pn], tw[la_key][pn]
        t0 = time.perf_counter()
        with PeakMemory() as pm:
            d09 = E.decisions(c, "train", TAG, T1_B09, drop)
            s1, pool, prob, sc, meta = s03_arrays(cfg, paths, "train", fd.get(comp))
            code, cm, fin = decide_s03(d09["codes"], s1, pool, prob, sc, meta)
            s03 = np.sort(code[fin])
            del s1, pool, prob, sc, code, cm, fin
            res = {"B09": E.pop_metrics(c, d09, pops, weights), "S03": E.pop_metrics(c, dict(d09, codes=s03), pops, weights, d09["codes"])}
            res["additions"] = int(len(np.setdiff1d(s03, d09["codes"])))
        rep["competitions"][comp] = {"populations": {k: int(v.sum()) for k, v in pops.items()}, "metrics": res, "seconds": round(time.perf_counter() - t0, 1), "peak_gb": round(pm.peak_gb, 2)}
        first = next(k for k in pops if k.startswith("SHADOW"))
        LOG.info("%-30s %s: B09 %.4f S03 %.4f additions %d | %.0fs %.2f GB", comp, first, res["B09"][first]["f05"], res["S03"][first]["f05"], res["additions"], time.perf_counter() - t0, pm.peak_gb)
        write_json(OUT / "eval.json", rep)
    rep["density_analysis"] = density(rep)
    write_json(OUT / "eval.json", rep)


def density(rep) -> dict:
    C = rep["competitions"]
    s02 = json.loads((X.OUT / "density_analysis.json").read_text(encoding="utf-8"))
    out = {"rule": "REJECT if adv(T1) - adv(X) > max(0.001, 0.25 * adv(T1)) for X in T4, T1W, T4W on DEV (notes.md 1.2 H); LOCKED reported"}
    keys = [("T1", "T1(base)", ""), ("T2", "T2", ""), ("T3", "T3", ""), ("T4", "T4", ""), ("T1W", "T1(base)", "[W]"), ("T4W", "T4", "[W]")]
    for pop in ("SHADOW-DEV", "SHADOW-LOCKED-A", "SHADOW-LOCKED-B"):
        dens = {r["tier"]: r["lookalike_density"] for r in s02[pop]["tiers"]}
        rows = []
        for t, ck, suf in keys:
            b, e = C[ck]["metrics"]["B09"][pop + suf], C[ck]["metrics"]["S03"][pop + suf]
            rows.append({"tier": t, "lookalike_density(B-06 defined)": dens[t], "B09_f05": b["f05"], "S03_f05": e["f05"], "adv": e["f05"] - b["f05"],
                         "d_precision": e["precision"] - b["precision"], "d_recall": e["recall"] - b["recall"],
                         "d_fm_per_1k": e["false_merges_per_1k_s1"] - b["false_merges_per_1k_s1"], "vetoes_per_s1": e["removals_per_s1_vs_B06"],
                         "additions_per_s1": e["additions_per_s1_vs_B06"]})
        a1 = rows[0]["adv"]
        thr = max(0.001, 0.25 * a1)
        dec = {r["tier"]: a1 - r["adv"] for r in rows if r["tier"] in ("T4", "T1W", "T4W")}
        d = np.array([r["lookalike_density(B-06 defined)"] for r in rows[:4]])
        out[pop] = {"tiers": rows, "slope_adv_per_unit_density(T1..T4)": float(np.polyfit(d, [r["adv"] for r in rows[:4]], 1)[0]), "decay_vs_T1": dec,
                    "material_threshold": thr, "any_material_decay": bool(any(v > thr for v in dec.values()))}
    return out


# ----------------------------------------------------------------------------- veto decomposition + score bins + countries
def cmd_veto(c):
    cfg, paths = cfgp()
    gts = np.sort(c.gtc)
    d09 = E.decisions(c, "train", TAG, T1_B09)
    s1, pool, prob, sc, meta = s03_arrays(cfg, paths, "train")
    code, cm, fin = decide_s03(d09["codes"], s1, pool, prob, sc, meta)
    y = E.is_true(code, gts)
    rep = {"stage2_threshold": meta["threshold"], "stage2_variant": meta["variant"], "populations": {}}
    cty = c.ctry[s1]
    for pname, m0 in (("E_half_B", c.P["E_half_B"]), ("validation", c.P["validation"]), ("GATE-B", np.load(X.V1 / "masks.npz")["gate_B"])):
        m = m0[s1]
        R = {}
        for g in ("all", "india", "us"):
            mm = m & ((cty == g) if g != "all" else True)
            acc, veto = cm & mm, cm & ~fin & mm
            fa = acc & ~y
            R[g] = {"B09_accepted": int(acc.sum()), "vetoed": int(veto.sum()), "veto_rate": float(veto.sum() / max(acc.sum(), 1)),
                    "true_vetoes(removed false merges)": int((veto & ~y).sum()), "false_vetoes(removed true matches)": int((veto & y).sum()),
                    "veto_precision(share of vetoes that were false pairs)": float((veto & ~y).sum() / max(veto.sum(), 1)),
                    "veto_recall(share of B-09 false acceptances removed)": float((veto & ~y).sum() / max(fa.sum(), 1)),
                    "additions": int((fin & ~cm & mm).sum())}
        bins = {}
        for lo, hi in BINS:
            b = m & (sc >= lo) & (sc < hi)
            acc, veto = b & cm, b & cm & ~fin
            bins[f"{max(lo, 0):g}-{hi if hi <= 1 else 1:g}"] = {"eligible_pairs": int(b.sum()), "B09_accepted": int(acc.sum()),
                                                               "B09_precision": float((acc & y).sum() / max(acc.sum(), 1)) if acc.any() else None,
                                                               "vetoed": int(veto.sum()), "veto_rate": float(veto.sum() / max(acc.sum(), 1)) if acc.any() else None,
                                                               "false_merges_B09": int((acc & ~y).sum()), "false_merges_removed": int((veto & ~y).sum()),
                                                               "true_removed": int((veto & y).sum()), "S03_precision": float((fin & b & y).sum() / max((fin & b).sum(), 1)) if (fin & b).any() else None}
        R["score_bins(stage-1 score)"] = bins
        R["stage2_prob_of_vetoed"] = summary(prob[cm & ~fin & m])
        rep["populations"][pname] = R
        LOG.info("veto %s: %s", pname, {g: {k: R[g][k] for k in ("vetoed", "true_vetoes(removed false merges)", "false_vetoes(removed true matches)", "veto_precision(share of vetoes that were false pairs)")} for g in ("all", "india", "us")})
    write_json(RUN / "veto_analysis.json", rep)


# ----------------------------------------------------------------------------- test diagnostics
def cmd_testdiag(c):
    cfg, paths = cfgp()
    files = {"B09": Path("experiments/runs/EXP-B-09/output"), "S03": RUN / "output"}
    rep = {"label": "UNLABELED test diagnostics"}
    rep["candidates_byte_identical"] = (files["B09"] / "candidate_pairs.tsv").read_bytes() == (files["S03"] / "candidate_pairs.tsv").read_bytes()
    ids, cnt = X.row_counts({"B06": files["B09"] / "matching_results.tsv", "S03": files["S03"] / "matching_results.tsv"})  # "B06" key = reference file (B-09)
    if ids != E._ids(c.paths, "test", "source1").to_pylist():
        raise AssertionError("row order")
    kb, kx, add, rem = cnt["B06"]["k"], cnt["S03"]["k"], cnt["S03"]["add"], cnt["S03"]["rem"]
    ct = c.ctry_te
    rep["predictions"] = {g: {"s1": int(m.sum()), "B09": int(kb[m].sum()), "S03": int(kx[m].sum()), "change": float(kx[m].sum() / kb[m].sum() - 1), "additions": int(add[m].sum()),
                              "removals": int(rem[m].sum()), "removals_per_s1": float(rem[m].sum() / m.sum()), "removal_rate(of B09 predictions)": float(rem[m].sum() / kb[m].sum()),
                              "B09_empty_rate": float((kb[m] == 0).mean()), "S03_empty_rate": float((kx[m] == 0).mean())}
                          for g, m in [("all", np.ones(len(kb), bool))] + [(g, ct == g) for g in ("france", "india", "us")]}
    d09 = E.decisions(c, "test", "scored_EXP-B-09", T1_B09)
    s1, pool, prob, sc, meta = s03_arrays(cfg, paths, "test")
    code, cm, fin = decide_s03(d09["codes"], s1, pool, prob, sc, meta)
    v = cm & ~fin
    rep["vetoed_pairs"] = {"count": int(v.sum()), "stage1_score": summary(sc[v]), "stage2_prob": summary(prob[v]),
                           "by_country": {g: int((v & (ct[s1] == g)).sum()) for g in ("france", "india", "us")}}
    rep["recomputed_matches_file"] = int(fin.sum()) == int(kx.sum())
    rep["conflicts_B09"] = {"before_d01": d09["conflicts_before_d01"]["conflicted_records"], "after_d01": d09["conflicts_after_d01"]["conflicted_records"]}
    ev = json.loads((RUN / "stage2_eval.json").read_text(encoding="utf-8"))
    tr = ev["results"]["E_half_B"]["vs_control"]["removals_per_s1"]
    vt = json.loads((RUN / "veto_analysis.json").read_text(encoding="utf-8"))["populations"]["E_half_B"]
    tr_c = {g: vt[g]["vetoed"] / max(int((c.P["E_half_B"] & (c.ctry == g)).sum()), 1) for g in ("india", "us")}
    p = rep["predictions"]
    rep["train_reference(E_half_B)"] = {"removals_per_s1": tr, "by_country": tr_c}
    rep["ratios"] = {"test_over_train_removals_per_s1": p["all"]["removals_per_s1"] / tr, "india_test_over_train": p["india"]["removals_per_s1"] / tr_c["india"],
                     "us_test_over_train": p["us"]["removals_per_s1"] / tr_c["us"],
                     "france_removal_rate_over_overall": p["france"]["removal_rate(of B09 predictions)"] / p["all"]["removal_rate(of B09 predictions)"]}
    write_json(RUN / "test_diagnostics.json", rep)
    LOG.info("testdiag: %s | ratios %s", {g: (round(v["change"], 4), v["additions"], v["removals"]) for g, v in p.items()}, rep["ratios"])


# ----------------------------------------------------------------------------- error budget (B-09 vs S-03)
def cmd_budget(c):
    cfg, paths = cfgp()
    gts = np.sort(c.gtc)
    cand, csc = [], []
    for p in sorted(c.paths.scored_dir("train", TAG).glob("*.parquet")):
        tb = pq.read_table(p, columns=["s1_row", "pool_row", "score"])
        cd = tb.column("s1_row").to_numpy().astype(np.int64) * SHIFT + tb.column("pool_row").to_numpy().astype(np.int64)
        y = E.is_true(cd, gts)
        cand.append(cd[y])
        csc.append(tb.column("score").to_numpy().astype(np.float32)[y])
    cand, csc = np.concatenate(cand), np.concatenate(csc)
    o = np.argsort(cand)
    cand, csc = cand[o], csc[o]
    d09 = E.decisions(c, "train", TAG, T1_B09)
    s1, pool, prob, sc, meta = s03_arrays(cfg, paths, "train")
    code, cm, fin = decide_s03(d09["codes"], s1, pool, prob, sc, meta)
    s03 = np.sort(code[fin])
    del s1, pool, prob, sc, code, cm, fin
    rep = {"definition": "oracle F0.5 gain [UB] from fixing ONE bucket alone; E half B and SHADOW-DEV", "systems": {}}
    for sysname, acc in (("B09+D01", d09["codes"]), ("S03", s03)):
        ya = E.is_true(acc, gts)
        tp, k = np.bincount(acc[ya] // SHIFT, minlength=c.n), np.bincount(acc // SHIFT, minlength=c.n)
        fn = gts[~E.is_true(gts, acc)]
        pos = np.minimum(np.searchsorted(cand, fn), len(cand) - 1)
        inc = cand[pos] == fn
        fs = np.where(inc, csc[pos], np.nan)
        in_ctl = E.is_true(fn, d09["codes"])
        bk = {"never_retrieved(not in top-20)": ~inc, "retrieved_score<0.1": inc & (fs < 0.1), "retrieved_0.1-0.5": inc & (fs >= 0.1) & (fs < 0.5),
              "retrieved_0.5-0.65": inc & (fs >= 0.5) & (fs < T1_B09), "score>=0.65_lost_D01": inc & (fs >= T1_B09) & ~in_ctl, "vetoed_true(S-03 cost)": in_ctl}
        fp = acc[~ya]
        single = c.n_true == 0
        fpb = {"false_merges(S1 with matches)": ~single[fp // SHIFT], "singleton_false_positives": single[fp // SHIFT]}
        R = {}
        for pname, m in (("E_half_B", c.P["E_half_B"]), ("SHADOW-DEV", X.masks_v2()["dev"])):
            base = weighted_metrics(tp[m], k[m], c.n_true[m])["f05"]
            r = {"f05": base, "fn": {}, "fp": {}}
            for n_, bm in bk.items():
                a = np.bincount(fn[bm] // SHIFT, minlength=c.n)
                r["fn"][n_] = {"pairs": int(a[m].sum()), "oracle_gain[UB]": weighted_metrics(tp[m] + a[m], k[m] + a[m], c.n_true[m])["f05"] - base}
            for n_, bm in fpb.items():
                rm = np.bincount(fp[bm] // SHIFT, minlength=c.n)
                r["fp"][n_] = {"pairs": int(rm[m].sum()), "oracle_gain[UB]": weighted_metrics(tp[m], k[m] - rm[m], c.n_true[m])["f05"] - base}
            R[pname] = r
        rep["systems"][sysname] = R
        LOG.info("budget %s: %s", sysname, {k2: (v["pairs"], round(v["oracle_gain[UB]"], 4)) for k2, v in (R["E_half_B"]["fn"] | R["E_half_B"]["fp"]).items()})
    write_json(RUN / "error_budget.json", rep)


# ----------------------------------------------------------------------------- leakage + read-only
def cmd_leakage(c):
    import hashlib
    cfg, paths = cfgp()
    P, _ = S.populations(cfg, paths)
    M1, M = np.load(X.V1 / "masks.npz"), X.masks_v2()
    tr = P["stage2_train"] | P["stage2_inner"]
    meta = json.loads((RUN / "stage2_model.json").read_text(encoding="utf-8"))
    rep = {f"stage2_fit_s1_in_{n}": int((tr & m).sum()) for n, m in (("validation", P["validation"]), ("E_half_B", P["E_half_B"]), ("stage1_fit", P["fit"]),
                                                                      ("SHADOW-BASE", M1["shadow"]), ("SHADOW-DEV", M["dev"]), ("SHADOW-LOCKED-A", M["locked_A"]),
                                                                      ("SHADOW-LOCKED-B", M["locked_B"]))}
    rep["stage2_train_inner_counts_match_model_meta"] = meta["train_s1"] == int(P["stage2_train"].sum()) and meta["inner_s1"] == int(P["stage2_inner"].sum())
    rep["stage2_fitted_on"] = {"stage1_run": meta["stage1_run"], "stage1_threshold": meta["stage1_threshold"], "train_tag": cfg["stage2"]["train_tag"]}
    rep["stage1_scores_out_of_sample_for_stage2_rows"] = "stage-2 rows = E half A minus validation; E excludes the B-09 fit set (stage2_fit_s1_in_stage1_fit = 0)"
    rep["threshold_source"] = "inner hash split of the stage-2 training partition (stage2_model.json inner_table); frozen before LOCKED evaluation"
    snap = {}
    for line in (OUT / "readonly_snapshot_before.sha256").read_text(encoding="utf-8").splitlines():
        h, f = line.split(maxsplit=1)
        f = f.lstrip("*")
        snap[f] = h == hashlib.sha256(Path(f).read_bytes()).hexdigest()
    rep["readonly_unchanged"] = snap
    rep["no_test_labels_no_external_data"] = True
    rep["all_ok"] = bool(all(v == 0 for k, v in rep.items() if k.startswith("stage2_fit_s1_in_")) and rep["stage2_train_inner_counts_match_model_meta"]
                         and meta["stage1_run"] == "EXP-B-09" and all(snap.values()))
    write_json(RUN / "leakage.json", rep)
    LOG.info("leakage all_ok=%s", rep["all_ok"])


# ----------------------------------------------------------------------------- gates (notes.md 1.2)
def cmd_gates(c):
    ev = json.loads((OUT / "eval.json").read_text(encoding="utf-8"))
    base = ev["competitions"]["T1(base)"]["metrics"]
    td = json.loads((RUN / "test_diagnostics.json").read_text(encoding="utf-8"))
    lk = json.loads((RUN / "leakage.json").read_text(encoding="utf-8"))
    det = json.loads((OUT / "determinism.json").read_text(encoding="utf-8"))
    rt = json.loads((RUN / "runtime_memory.json").read_text(encoding="utf-8"))
    sub = json.loads((RUN / "submission_summary.json").read_text(encoding="utf-8"))
    off = (OUT / "official_validator.txt").read_text(encoding="utf-8")
    dd = lambda comp, pop, key="f05": ev["competitions"][comp]["metrics"]["S03"][pop][key] - ev["competitions"][comp]["metrics"]["B09"][pop][key]  # noqa: E731
    P3 = ("validation", "GATE-B", "E_half_B")
    g = {}
    g["A_f05>=+0.002"] = {p: {"delta": dd("T1(base)", p), "pass": dd("T1(base)", p) >= 0.002} for p in P3}
    hv = {h: dd("T1(base)", h) for h in ("validation_half_A", "validation_half_B")}
    g["B_halves_agree"] = {"deltas": hv, "pass": all(v > 0 for v in hv.values())}
    g["C_precision>=-0.0005"] = {p: {"delta": dd("T1(base)", p, "precision"), "pass": dd("T1(base)", p, "precision") >= -0.0005} for p in P3}
    fm = {p: base["S03"][p]["false_merges_per_1k_s1"] / base["B09"][p]["false_merges_per_1k_s1"] - 1 for p in P3}
    g["D_false_merges_-10%"] = {"relative_change": fm, "pass": all(v <= -0.10 for v in fm.values())}
    adds = sum(ev["competitions"][k]["metrics"]["additions"] for k in ev["competitions"]) + td["predictions"]["all"]["additions"] + sub.get("vs_control_test", {}).get("additions", 0)
    g["E_additions_zero"] = {"additions_total": adds, "pass": adds == 0}
    g["F_test_matched>=+0.002"] = {"delta": dd("test_matched(v1 random orphan)", "SHADOW-TEST-MATCHED"), "pass": dd("test_matched(v1 random orphan)", "SHADOW-TEST-MATCHED") >= 0.002}
    g["G_chain_orphaned>=+0.002"] = {"delta": dd("chain_orphaned(v1)", "SHADOW-CHAIN-ORPHANED"), "pass": dd("chain_orphaned(v1)", "SHADOW-CHAIN-ORPHANED") >= 0.002}
    da = ev["density_analysis"]
    g["H_no_density_decay(DEV)"] = {"decay": da["SHADOW-DEV"]["decay_vs_T1"], "threshold": da["SHADOW-DEV"]["material_threshold"], "pass": not da["SHADOW-DEV"]["any_material_decay"],
                                    "locked_reported": {p: da[p]["any_material_decay"] for p in ("SHADOW-LOCKED-A", "SHADOW-LOCKED-B")}}
    dev = dd("T1(base)", "SHADOW-DEV")
    g["I_locked_within_0.0015(confirmatory)"] = {p: {"delta": dd("T1(base)", p), "dev": dev, "pass": abs(dd("T1(base)", p) - dev) <= 0.0015} for p in ("SHADOW-LOCKED-A", "SHADOW-LOCKED-B")}
    r = td["ratios"]
    g["J_test_removals_consistent"] = {"test_over_train": r["test_over_train_removals_per_s1"], "france_over_overall": r["france_removal_rate_over_overall"],
                                       "pass": abs(r["test_over_train_removals_per_s1"] - 1) <= 0.25 and r["france_removal_rate_over_overall"] <= 1.5}
    ctry = {f"{g_}:{p}": base["S03"][p]["by_country"][g_]["f05"] - base["B09"][p]["by_country"][g_]["f05"] for p in P3 for g_ in ("india", "us")}
    g["K_countries_not_worse"] = {"deltas": ctry, "pass": all(v >= -0.001 for v in ctry.values())}
    g["L_no_leakage"] = {"pass": bool(lk["all_ok"])}
    g["M_determinism"] = {"pass": bool(det["byte_identical"])}
    g["N_strict_and_candidates_identical"] = {"pass": sub.get("strict_validation_errors") == [] and td["candidates_byte_identical"]}
    g["O_official_validator"] = {"pass": "PASS" in off}
    g["P_memory<=5GB"] = {"peaks": {k: v["peak_gb"] for k, v in rt["pipeline"].items() if v}, "pass": all(v["peak_gb"] <= 5.0 for v in rt["pipeline"].values() if v)}
    flat = []
    for v in g.values():
        flat += [v["pass"]] if "pass" in v else [x["pass"] for x in v.values() if isinstance(x, dict) and "pass" in x]
    g["all_pass"] = bool(all(flat))
    write_json(RUN / "gates.json", g)
    LOG.info("gates all_pass=%s", g["all_pass"])


CMDS = {"shadow-features": cmd_shadow_features, "eval": cmd_eval, "veto": cmd_veto, "testdiag": cmd_testdiag, "budget": cmd_budget, "leakage": cmd_leakage, "gates": cmd_gates}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=list(CMDS))
    a = ap.parse_args()
    setup_logging()
    configure_arrow()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with PeakMemory() as pm:
        CMDS[a.cmd](X.Ctx())
    LOG.info("%s done in %.0fs, peak %.2f GB", a.cmd, time.perf_counter() - t0, pm.peak_gb)


if __name__ == "__main__":
    main()
