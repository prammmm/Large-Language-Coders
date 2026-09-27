"""EXP-S-04 (B-10 blocker + fresh stage-1 fit + fresh S-03-style veto) evaluation. Deterministic; read-only w.r.t. every other
experiment (S-03 / B-09 artefacts are only read).

    python scripts/analysis/exp_s04.py eval-base | gates | shadow-features | eval-shadow | testdiag | budget | leakage

Systems: B09 (B-09 + D-01 @ 0.65), S03 (primary control), S04s1 (S-04 stage 1 + D-01 @ its inner threshold), S04 (S04s1 AND
refitted stage-2 veto). eval-base = the T1(base) competition (validation, halves, GATE-B, E half B, E, shadow base/dev/locked);
eval-shadow = test-matched, chain, T2-T4 (only after G1-G8 pass). Gates: notes.md 1.2 / 1.3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_b09_eval as E  # noqa: E402
import exp_s02_shadow as X  # noqa: E402
import exp_s03 as S3  # noqa: E402

from ber.config import Paths, load_config  # noqa: E402
from ber.forensics.diagnostics import summary  # noqa: E402
from ber.forensics.shadow import weighted_metrics  # noqa: E402
from ber.pipeline import run_stage2 as S  # noqa: E402
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, write_json  # noqa: E402

CFG4 = "configs/exp_s04_veto_on_b10.yaml"
RUN = Path("experiments/runs/EXP-S-04")
OUT = RUN / "analysis"
SHIFT = S.SHIFT
TAG09, TAG04 = "forensic_alltrain_B09", "forensic_alltrain_S04"
T09 = 0.65
P3 = ("validation", "GATE-B", "E_half_B")


def cfg4():
    c = load_config(CFG4)
    return c, Paths(c)


def t04() -> float:
    return float(json.loads((RUN / "model.json").read_text(encoding="utf-8"))["meta"]["threshold"])


def fdirs4(paths) -> dict:
    base = paths.processed_dir("train")
    return {"test_matched(v1 random orphan)": base / "stage2_EXP-S-04_orphan", "chain_orphaned(v1)": base / "stage2_EXP-S-04_chainorphan",
            "T2": base / "stage2_EXP-S-04_v2tier2", "T3": base / "stage2_EXP-S-04_v2tier3", "T4": base / "stage2_EXP-S-04_v2tier4"}


def veto(cfg, paths, ctl_codes, split, fdir=None):
    """Stage-2 veto of a system on its own stage-1 control: returns sorted final codes, additions, weak (<0.8) accepted codes."""
    booster, meta = S.load_stage2(cfg, paths)
    s1, pool, prob, sc = S.predict_all(cfg, paths, split, booster, meta["features"], fdir)
    code, cm, fin = S3.decide_s03(ctl_codes, s1, pool, prob, sc, meta)  # asserts zero additions
    out = np.sort(code[fin]), int((fin & ~cm).sum()), np.sort(code[fin & (sc < 0.8)]), (code[cm & ~fin], sc[cm & ~fin], prob[cm & ~fin])
    del s1, pool, prob, sc, code, cm, fin
    return out


def run_comp(c, comp, pops, weights, drop):
    c3, p3 = S3.cfgp()
    c4, p4 = cfg4()
    d09 = E.decisions(c, "train", TAG09, T09, drop)
    s03, add3, w03, _ = veto(c3, p3, d09["codes"], "train", S3.fdirs(p3).get(comp))
    d04 = E.decisions(c, "train", TAG04, t04(), drop)
    s04, add4, w04, vet = veto(c4, p4, d04["codes"], "train", fdirs4(p4).get(comp))
    res = {"B09": E.pop_metrics(c, d09, pops, weights), "S03": E.pop_metrics(c, dict(d09, codes=s03), pops, weights, d09["codes"]),
           "S04s1": E.pop_metrics(c, d04, pops, weights), "S04": E.pop_metrics(c, dict(d04, codes=s04), pops, weights, d04["codes"]),
           "additions": {"S03": add3, "S04": add4}}
    wk3, wk4 = np.bincount(w03 // SHIFT, minlength=c.n), np.bincount(w04 // SHIFT, minlength=c.n)
    res["weak_accepted(stage1<0.8)_per_s1"] = {p: {"S03": X.avg(wk3[m], weights.get(p)[m] if weights.get(p) is not None else None),
                                                   "S04": X.avg(wk4[m], weights.get(p)[m] if weights.get(p) is not None else None)} for p, m in pops.items()}
    return res, d09, s03, d04, s04, vet


def log_comp(comp, res, pops):
    for pn in [k for k in pops if not k.endswith("[W]")]:
        a, b, s1 = res["S03"][pn], res["S04"][pn], res["S04s1"][pn]
        LOG.info("%-22s %-20s S03 %.4f (P %.4f) | S04s1 %.4f | S04 %.4f (P %.4f) | S04-S03 %+.4f", comp, pn, a["f05"], a["precision"], s1["f05"], b["f05"], b["precision"], b["f05"] - a["f05"])


def pops_for(c, comp):
    comps = X.competitions(c)
    tw, ess, share = E.tier_weights(c, comps)
    _, drop, pops, weights = comps[comp]
    pops, weights = dict(pops), dict(weights)
    la_key = "T1" if comp == "T1(base)" else ("T4" if comp == "T4" else None)
    if la_key:
        for pn in ("SHADOW-DEV", "SHADOW-LOCKED-A", "SHADOW-LOCKED-B"):
            pops[f"{pn}[W]"], weights[f"{pn}[W]"] = pops[pn], tw[la_key][pn]
    return drop, pops, weights


def load_eval():
    f = OUT / "eval.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"competitions": {}}


def cand_codes(c, tag, m):
    out = []
    for p in sorted(c.paths.scored_dir("train", tag).glob("*.parquet")):
        tb = pq.read_table(p, columns=["s1_row", "pool_row"])
        s1 = tb.column("s1_row").to_numpy().astype(np.int64)
        k = m[s1]
        out.append(s1[k] * SHIFT + tb.column("pool_row").to_numpy().astype(np.int64)[k])
    return np.sort(np.concatenate(out))


def cmd_eval_base(c):
    comp = "T1(base)"
    drop, pops, weights = pops_for(c, comp)
    t0 = time.perf_counter()
    with PeakMemory() as pm:
        res, d09, s03, d04, s04, vet = run_comp(c, comp, pops, weights, drop)
        gts = np.sort(c.gtc)
        # candidate-list movement B-09 -> S-04 (full competition, top-20 lists) on validation and E half B
        mv = {}
        for pn in ("validation", "E_half_B"):
            m = pops[pn]
            a, b = cand_codes(c, TAG09, m), cand_codes(c, TAG04, m)
            new, lost = np.setdiff1d(b, a), np.setdiff1d(a, b)
            nt, lt = int(E.is_true(new, gts).sum()), int(E.is_true(lost, gts).sum())
            mv[pn] = {"B09_candidates": int(len(a)), "S04_candidates": int(len(b)), "true_added": nt, "false_added": int(len(new) - nt), "true_lost": lt,
                      "false_displaced": int(len(lost) - lt), "net_true": nt - lt}
        res["candidate_movement_vs_B09(top-20, full competition)"] = mv
        vc, vs, vp = vet
        res["S04_vetoed_all_train"] = {"count": int(len(vc)), "true_share": float(E.is_true(np.sort(vc), gts).mean()) if len(vc) else None,
                                       "stage1_score": summary(vs), "stage2_prob": summary(vp)}
    ev = load_eval()
    ev["systems"] = {"B09": "B-09 + D-01 @ 0.65", "S03": "primary control: B-09 + D-01 AND S-03 veto", "S04s1": f"S-04 stage 1 (B-10 blocker, refit) + D-01 @ {t04()}",
                     "S04": "S04s1 AND refitted stage-2 veto", "note": "removals_per_s1_vs_B06 = removals vs each system's own stage-1 control"}
    ev["competitions"][comp] = {"populations": {k: int(v.sum()) for k, v in pops.items()}, "metrics": res, "seconds": round(time.perf_counter() - t0, 1), "peak_gb": round(pm.peak_gb, 2)}
    write_json(OUT / "eval.json", ev)
    log_comp(comp, res, pops)
    LOG.info("additions %s | movement %s | %.0fs peak %.2f GB", res["additions"], mv, time.perf_counter() - t0, pm.peak_gb)


def cmd_gates(c):
    ev = load_eval()
    base = ev["competitions"]["T1(base)"]["metrics"]
    d = lambda p, k="f05": base["S04"][p][k] - base["S03"][p][k]  # noqa: E731
    g = {}
    for i, p in zip(("G1", "G2", "G3"), P3):
        g[i] = {"rule": f"{p}: S04 >= S03 + 0.0005", "S03": base["S03"][p]["f05"], "S04": base["S04"][p]["f05"], "delta": d(p), "pass": d(p) >= 0.0005}
    g["G4"] = {"rule": "precision drop <= 0.0005", "delta": {p: d(p, "precision") for p in P3}, "pass": all(d(p, "precision") >= -0.0005 for p in P3)}
    fm = {p: {"S03_per_1k": base["S03"][p]["false_merges_per_1k_s1"], "S04_per_1k": base["S04"][p]["false_merges_per_1k_s1"],
              "S03_count": round(base["S03"][p]["false_merges_per_1k_s1"] * ev["competitions"]["T1(base)"]["populations"][p] / 1000),
              "S04_count": round(base["S04"][p]["false_merges_per_1k_s1"] * ev["competitions"]["T1(base)"]["populations"][p] / 1000),
              "relative": base["S04"][p]["false_merges_per_1k_s1"] / base["S03"][p]["false_merges_per_1k_s1"] - 1} for p in P3}
    g["G5"] = {"rule": "false merges per 1k <= S03 x 1.05", "value": fm, "pass": all(v["relative"] <= 0.05 for v in fm.values())}
    for gi, cc in (("G6", "india"), ("G7", "us")):
        v = {p: base["S04"][p]["by_country"][cc]["f05"] - base["S03"][p]["by_country"][cc]["f05"] for p in P3}
        g[gi] = {"rule": f"{cc} F0.5 delta >= -0.001", "delta": v, "pass": all(x >= -0.001 for x in v.values())}
    adds = sum(ev["competitions"][k]["metrics"]["additions"]["S04"] for k in ev["competitions"])
    g["G8"] = {"rule": "stage-2 additions = 0", "additions": adds, "pass": adds == 0}
    g["G1_G8_pass"] = all(g[f"G{i}"]["pass"] for i in range(1, 9))
    sh = ev.get("density")
    if sh:
        g["SHADOW"] = sh
    write_json(RUN / "gates.json", g)
    LOG.info("gates: %s | G1-G8 pass=%s", {k: (round(v["delta"], 5) if isinstance(v.get("delta"), float) else v.get("delta"), v["pass"]) for k, v in g.items() if k.startswith("G") and k[1:].isdigit()}, g["G1_G8_pass"])


def cmd_shadow_features(c):
    cfg, paths = cfg4()
    rep = {}
    drops = S3.drops()
    for comp, fd in fdirs4(paths).items():
        with PeakMemory() as pm:
            st = S.build_features(cfg, paths, "train", drop_s1=drops[comp], out=fd)
        rep[comp] = {"dir": fd.as_posix(), "eligible_pairs": st["eligible_pairs"], "dropped_s1": st["dropped_s1"], "seconds": round(pm.seconds, 1), "peak_gb": round(pm.peak_gb, 2)}
        LOG.info("%s: %s", comp, rep[comp])
        write_json(OUT / "shadow_features.json", rep)


def cmd_eval_shadow(c):
    for comp in ("T4", "test_matched(v1 random orphan)", "chain_orphaned(v1)", "T2", "T3"):
        drop, pops, weights = pops_for(c, comp)
        t0 = time.perf_counter()
        with PeakMemory() as pm:
            res = run_comp(c, comp, pops, weights, drop)[0]
        ev = load_eval()
        ev["competitions"][comp] = {"populations": {k: int(v.sum()) for k, v in pops.items()}, "metrics": res, "seconds": round(time.perf_counter() - t0, 1), "peak_gb": round(pm.peak_gb, 2)}
        ev["density"] = density(ev)
        write_json(OUT / "eval.json", ev)
        log_comp(comp, res, pops)
    ev = load_eval()
    LOG.info("density %s", ev["density"])


def density(ev) -> dict:
    C = ev["competitions"]
    out = {}
    for pop in ("SHADOW-DEV", "SHADOW-LOCKED-A", "SHADOW-LOCKED-B"):
        rows = {}
        for t, ck, suf in (("T1", "T1(base)", ""), ("T2", "T2", ""), ("T3", "T3", ""), ("T4", "T4", ""), ("T1W", "T1(base)", "[W]"), ("T4W", "T4", "[W]")):
            if ck in C and pop + suf in C[ck]["metrics"]["S04"]:
                a, b = C[ck]["metrics"]["S03"][pop + suf], C[ck]["metrics"]["S04"][pop + suf]
                rows[t] = {"S03": a["f05"], "S04": b["f05"], "adv": b["f05"] - a["f05"], "d_precision": b["precision"] - a["precision"]}
        if "T1" in rows:
            a1 = rows["T1"]["adv"]
            thr = max(0.001, 0.25 * a1)
            dec = {t: a1 - rows[t]["adv"] for t in ("T4", "T1W", "T4W") if t in rows}
            rows["_rule"] = {"decay_vs_T1": dec, "threshold": thr, "no_material_decay": all(v <= thr for v in dec.values()),
                             "T1W_T4W_positive": all(rows[t]["adv"] > 0 for t in ("T1W", "T4W") if t in rows)}
        out[pop] = rows
    if "test_matched(v1 random orphan)" in C:
        m = C["test_matched(v1 random orphan)"]["metrics"]
        out["SHADOW-TEST-MATCHED_adv"] = m["S04"]["SHADOW-TEST-MATCHED"]["f05"] - m["S03"]["SHADOW-TEST-MATCHED"]["f05"]
    if "chain_orphaned(v1)" in C:
        m = C["chain_orphaned(v1)"]["metrics"]
        out["SHADOW-CHAIN_adv"] = {p: m["S04"][p]["f05"] - m["S03"][p]["f05"] for p in ("SHADOW-CHAIN-ORPHANED", "SHADOW-CHAIN-MEMBERS")}
    return out


def cmd_testdiag(c):
    c3, p3 = S3.cfgp()
    c4, p4 = cfg4()
    files = {"S03": Path("experiments/runs/EXP-S-03/output"), "S04": RUN / "output"}
    ids, cnt = X.row_counts({"B06": files["S03"] / "matching_results.tsv", "S04": files["S04"] / "matching_results.tsv"})  # "B06" = reference (S-03)
    if ids != E._ids(c.paths, "test", "source1").to_pylist():
        raise AssertionError("row order")
    ka, kb = cnt["B06"]["k"], cnt["S04"]["k"]
    ct = c.ctry_te
    d04 = E.decisions(c, "test", "scored_EXP-S-04", t04())
    d09 = E.decisions(c, "test", "scored_EXP-B-09", T09)
    s04, add4, w04, vet = veto(c4, p4, d04["codes"], "test")
    s03, add3, w03, _ = veto(c3, p3, d09["codes"], "test")
    n = len(ct)
    k4r = np.bincount(s04 // SHIFT, minlength=n)
    rep = {"label": "UNLABELED test diagnostics", "recomputed_S04_matches_file": bool(np.array_equal(k4r, kb)), "additions_S04": add4}
    rep["by_country"] = {}
    wk3, wk4 = np.bincount(w03 // SHIFT, minlength=n), np.bincount(w04 // SHIFT, minlength=n)
    vetn = np.bincount(vet[0] // SHIFT, minlength=n)
    for g, m in [("all", np.ones(n, bool))] + [(g, ct == g) for g in ("france", "india", "us")]:
        rep["by_country"][g] = {"s1": int(m.sum()), "S03_predictions_per_s1": float(ka[m].mean()), "S04_predictions_per_s1": float(kb[m].mean()),
                                "prediction_change": float(kb[m].sum() / ka[m].sum() - 1), "S03_candidates_per_s1": float(d09["ncand"][m].mean()),
                                "S04_candidates_per_s1": float(d04["ncand"][m].mean()), "S03_weak_accepted_per_s1": float(wk3[m].mean()),
                                "S04_weak_accepted_per_s1": float(wk4[m].mean()), "S04_veto_removals_per_s1": float(vetn[m].mean()),
                                "S03_empty_rate": float((ka[m] == 0).mean()), "S04_empty_rate": float((kb[m] == 0).mean())}
    ev = load_eval()
    b = ev["competitions"]["T1(base)"]["metrics"]
    rep["train_reference(E_half_B)"] = {"S04_removals_per_s1": b["S04"]["E_half_B"]["removals_per_s1_vs_B06"], "S03_removals_per_s1": b["S03"]["E_half_B"]["removals_per_s1_vs_B06"],
                                        "weak_accepted": ev["competitions"]["T1(base)"]["metrics"]["weak_accepted(stage1<0.8)_per_s1"]["E_half_B"]}
    rep["conflicts_S04"] = {"before_d01": d04["conflicts_before_d01"]["conflicted_records"], "after_d01": d04["conflicts_after_d01"]["conflicted_records"]}
    write_json(RUN / "test_diagnostics.json", rep)
    LOG.info("testdiag: %s", rep["by_country"])


def budget_for(c, tag, t1, acc, ctl, gts):
    cand, csc = [], []
    for p in sorted(c.paths.scored_dir("train", tag).glob("*.parquet")):
        tb = pq.read_table(p, columns=["s1_row", "pool_row", "score"])
        cd = tb.column("s1_row").to_numpy().astype(np.int64) * SHIFT + tb.column("pool_row").to_numpy().astype(np.int64)
        y = E.is_true(cd, gts)
        cand.append(cd[y])
        csc.append(tb.column("score").to_numpy().astype(np.float32)[y])
    cand, csc = np.concatenate(cand), np.concatenate(csc)
    o = np.argsort(cand)
    cand, csc = cand[o], csc[o]
    ya = E.is_true(acc, gts)
    tp, k = np.bincount(acc[ya] // SHIFT, minlength=c.n), np.bincount(acc // SHIFT, minlength=c.n)
    fn = gts[~E.is_true(gts, acc)]
    pos = np.minimum(np.searchsorted(cand, fn), len(cand) - 1)
    inc = cand[pos] == fn
    fs = np.where(inc, csc[pos], np.nan)
    in_ctl = E.is_true(fn, ctl)
    bk = {"A_never_retrieved(not in top-20)": ~inc, "B_retrieved_weak(<0.5)": inc & (fs < 0.5), "C_threshold_region(0.5-t1)": inc & (fs >= 0.5) & (fs < t1),
          "F_score>=t1_lost_D01": inc & (fs >= t1) & ~in_ctl, "G_vetoed_true(stage-2 cost)": in_ctl}
    fp = acc[~ya]
    single = c.n_true == 0
    fpb = {"D_false_merges(S1 with matches)": ~single[fp // SHIFT], "E_singleton_false_positives": single[fp // SHIFT]}
    m = c.P["E_half_B"]
    base = weighted_metrics(tp[m], k[m], c.n_true[m])["f05"]
    r = {"f05": base, "buckets": {}}
    for n_, bm in bk.items():
        a = np.bincount(fn[bm] // SHIFT, minlength=c.n)
        r["buckets"][n_] = {"pairs": int(a[m].sum()), "oracle_gain[UB]": weighted_metrics(tp[m] + a[m], k[m] + a[m], c.n_true[m])["f05"] - base}
    for n_, bm in fpb.items():
        rm = np.bincount(fp[bm] // SHIFT, minlength=c.n)
        r["buckets"][n_] = {"pairs": int(rm[m].sum()), "oracle_gain[UB]": weighted_metrics(tp[m], k[m] - rm[m], c.n_true[m])["f05"] - base}
    return r


def cmd_budget(c):
    c3, p3 = S3.cfgp()
    c4, p4 = cfg4()
    gts = np.sort(c.gtc)
    d09 = E.decisions(c, "train", TAG09, T09)
    s03 = veto(c3, p3, d09["codes"], "train")[0]
    r3 = budget_for(c, TAG09, T09, s03, d09["codes"], gts)
    del s03
    d04 = E.decisions(c, "train", TAG04, t04())
    s04 = veto(c4, p4, d04["codes"], "train")[0]
    r4 = budget_for(c, TAG04, t04(), s04, d04["codes"], gts)
    rep = {"definition": "E half B; oracle F0.5 gain [UB] from fixing ONE bucket alone; B/C bands use each system's own stage-1 score and threshold",
           "S03": r3, "S04": r4, "delta_pairs(S04-S03)": {k: r4["buckets"][k]["pairs"] - r3["buckets"][k]["pairs"] for k in r3["buckets"]}}
    write_json(RUN / "error_budget.json", rep)
    LOG.info("budget delta pairs: %s", rep["delta_pairs(S04-S03)"])


def cmd_leakage(c):
    cfg, paths = cfg4()
    P, _ = S.populations(cfg, paths)
    M1, M = np.load(X.V1 / "masks.npz"), X.masks_v2()
    tr = P["stage2_train"] | P["stage2_inner"]
    meta = json.loads((RUN / "stage2_model.json").read_text(encoding="utf-8"))
    m1 = json.loads((RUN / "model.json").read_text(encoding="utf-8"))["meta"]
    rep = {f"stage2_fit_s1_in_{n}": int((tr & m).sum()) for n, m in (("validation", P["validation"]), ("E_half_B", P["E_half_B"]), ("stage1_fit", P["fit"]),
                                                                      ("SHADOW-BASE", M1["shadow"]), ("SHADOW-DEV", M["dev"]), ("SHADOW-LOCKED-A", M["locked_A"]),
                                                                      ("SHADOW-LOCKED-B", M["locked_B"]))}
    rep["stage1_fit_s1_in_validation"] = int((P["fit"] & P["validation"]).sum())
    rep["stage2_counts_match_meta"] = meta["train_s1"] == int(P["stage2_train"].sum()) and meta["inner_s1"] == int(P["stage2_inner"].sum())
    rep["stage2_fitted_on"] = {"stage1_run": meta["stage1_run"], "stage1_threshold": meta["stage1_threshold"], "train_tag": cfg["stage2"]["train_tag"]}
    rep["stage1_model"] = {"training_parts": m1["training_parts"], "config": m1["config"], "threshold": m1["threshold"], "fit_s1": m1["fit_s1"]}
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()  # noqa: E731
    rep["hashes"] = {f: sha(f) for f in ("configs/exp_s04_stage1_b10.yaml", "configs/exp_s04_veto_on_b10.yaml", str(RUN / "model.json"), str(RUN / "stage2_model.lgb.txt"),
                                         "output/matching_results.tsv", "output/candidate_pairs.tsv")}
    rep["stage2_weights_differ_from_S03"] = sha(RUN / "stage2_model.lgb.txt") != sha("experiments/runs/EXP-S-03/stage2_model.lgb.txt")
    rep["root_output_is_S03"] = rep["hashes"]["output/matching_results.tsv"].startswith("46cb8fb5") and rep["hashes"]["output/candidate_pairs.tsv"].startswith("e6fee476")
    rep["no_test_labels_no_external_data_no_country_rules"] = True
    rep["all_ok"] = bool(all(v == 0 for k, v in rep.items() if k.startswith("stage2_fit_s1_in_")) and rep["stage1_fit_s1_in_validation"] == 0 and rep["stage2_counts_match_meta"]
                         and meta["stage1_run"] == "EXP-S-04" and rep["stage2_weights_differ_from_S03"] and rep["root_output_is_S03"])
    write_json(RUN / "leakage.json", rep)
    LOG.info("leakage all_ok=%s", rep["all_ok"])


CMDS = {"eval-base": cmd_eval_base, "gates": cmd_gates, "shadow-features": cmd_shadow_features, "eval-shadow": cmd_eval_shadow, "testdiag": cmd_testdiag,
        "budget": cmd_budget, "leakage": cmd_leakage}


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
