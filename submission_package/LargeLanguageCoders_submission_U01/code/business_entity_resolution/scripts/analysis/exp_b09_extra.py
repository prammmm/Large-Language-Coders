"""EXP-B-09 specific analyses (list-membership decomposition, displacement, India structure, test diagnostics, gates A-P).
Built on exp_b09_eval (streamed evaluator shared with EXP-B-08). Deterministic; read-only w.r.t. all other experiments.

    python scripts/analysis/exp_b09_extra.py decompose | testdiag | india | gates
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
import exp_b09_eval as E  # noqa: E402
import exp_s02_shadow as X  # noqa: E402
import final_audit as FA  # noqa: E402

from ber.forensics.audit import pair_profile  # noqa: E402
from ber.forensics.diagnostics import summary  # noqa: E402
from ber.forensics.shadow import weighted_metrics  # noqa: E402
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, write_json  # noqa: E402

SHIFT = E.SHIFT
OUT, RUN = E.OUT, E.RUN
NN_BIT = 1 << 8  # PASS_ORDER index of name_num


def list_codes(c, split, tag, m=None):
    """Candidate-list codes (sorted) of a scored split, optionally restricted to S1 in mask m."""
    out = []
    for p in sorted(c.paths.scored_dir(split, tag).glob("*.parquet")):
        tb = pq.read_table(p, columns=["s1_row", "pool_row"])
        s1 = tb.column("s1_row").to_numpy().astype(np.int64)
        keep = m[s1] if m is not None else slice(None)
        out.append(s1[keep] * SHIFT + tb.column("pool_row").to_numpy().astype(np.int64)[keep])
    return np.sort(np.concatenate(out))


def member(x, sorted_ref):
    return E.is_true(x, sorted_ref)  # searchsorted membership


# ----------------------------------------------------------------------------- decomposition + displacement
def cmd_decompose(c):
    gts = np.sort(c.gtc)
    d06 = E.decisions(c, "train", E.TAG["B06"], E.threshold("B06"))
    d09 = E.decisions(c, "train", E.TAG["B09"], E.threshold("B09"))
    M = X.masks_v2()
    pops = {"E_half_B": c.P["E_half_B"], "validation": c.P["validation"], "GATE-B": np.load(X.V1 / "masks.npz")["gate_B"], "SHADOW-DEV": M["dev"]}
    dec, disp_rep = {}, {}
    for pname, m in pops.items():
        L06, L09 = list_codes(c, "train", E.TAG["B06"], m), list_codes(c, "train", E.TAG["B09"], m)
        new = L09[~member(L09, L06)]
        gone = L06[~member(L06, L09)]
        a06 = d06["codes"][m[d06["codes"] // SHIFT]]
        a09 = d09["codes"][m[d09["codes"] // SHIFT]]
        acc_new = a09[member(a09, new)]  # accepted pairs that are new candidates
        lost_disp = a06[member(a06, gone)]  # B-06 acceptances whose candidate was displaced from the B-09 list
        common06 = a06[~member(a06, gone)]
        common09 = a09[~member(a09, new)]
        resc_add = common09[~member(common09, common06)]
        resc_rem = common06[~member(common06, common09)]
        t = lambda x: int(member(x, gts).sum())  # noqa: E731
        bc = lambda x: np.bincount(x // SHIFT, minlength=c.n)  # noqa: E731
        tp06, k06 = bc(a06[member(a06, gts)]), bc(a06)
        tp09, k09 = bc(a09[member(a09, gts)]), bc(a09)
        # hybrid = B-06 decisions with ONLY the list change applied (new acceptances in, displaced acceptances out)
        tp_h = tp06 + bc(acc_new[member(acc_new, gts)]) - bc(lost_disp[member(lost_disp, gts)])
        k_h = k06 + bc(acc_new) - bc(lost_disp)
        f = lambda tp, k: weighted_metrics(tp[m], k[m], c.n_true[m])  # noqa: E731
        f06, f09, fh = f(tp06, k06), f(tp09, k09), f(tp_h, k_h)
        dec[pname] = {
            "s1": int(m.sum()), "true_pairs": int(c.n_true[m].sum()),
            "A_new_candidates": int(len(new)), "B_new_true": t(new), "C_new_false": int(len(new) - t(new)),
            "D_newly_accepted_true(from new candidates)": t(acc_new), "E_newly_accepted_false(from new candidates)": int(len(acc_new) - t(acc_new)),
            "precision_of_accepted_new_candidates": t(acc_new) / max(len(acc_new), 1),
            "F_existing_candidates_displaced": int(len(gone)), "G_existing_true_displaced": t(gone),
            "G2_B06_acceptances_lost_to_displacement": {"pairs": int(len(lost_disp)), "true": t(lost_disp)},
            "H_rescoring_changes(common candidates)": {"added": int(len(resc_add)), "added_true": t(resc_add), "removed": int(len(resc_rem)), "removed_true": t(resc_rem)},
            "f05": {"B06": f06["f05"], "hybrid(list change only)": fh["f05"], "B09": f09["f05"], "delta_total": f09["f05"] - f06["f05"],
                    "delta_from_list_change": fh["f05"] - f06["f05"], "delta_from_rescoring_and_d01": f09["f05"] - fh["f05"]},
            "precision": {"B06": f06["precision"], "B09": f09["precision"]}, "recall": {"B06": f06["recall"], "B09": f09["recall"]}}
        rows = {}
        for g in ("india", "us"):
            mg = c.ctry == g
            sel = lambda x: x[mg[x // SHIFT]]  # noqa: E731
            rows[g] = {"new_candidates": int(len(sel(new))), "new_true": t(sel(new)), "displaced": int(len(sel(gone))), "displaced_true": t(sel(gone)),
                       "displacement_rate(displaced/list)": float(len(sel(gone)) / max(len(sel(L06)), 1)),
                       "accepted_new_true": t(sel(acc_new)), "accepted_new_false": int(len(sel(acc_new)) - t(sel(acc_new)))}
        disp_rep[pname] = {"displacement_rate(displaced/B06 list)": float(len(gone) / max(len(L06), 1)), "true_displacement_rate(displaced true/B06 true in list)":
                           t(gone) / max(t(L06), 1), "by_country": rows}
        LOG.info("decompose %s: %s", pname, dec[pname]["f05"])
        del L06, L09, new, gone
    write_json(OUT / "decomposition.json", dec)
    write_json(RUN / "displacement_analysis.json", disp_rep)


# ----------------------------------------------------------------------------- test diagnostics
def cand_changes(f06: Path, f09: Path) -> dict:
    rows = new = gone = n06 = n09 = changed = 0
    with open(f06, "rb") as a, open(f09, "rb") as b:
        if next(a) != next(b):
            raise AssertionError("header")
        for la, lb in zip(a, b):
            pa_, pb_ = la.rstrip(b"\n").partition(b"\t"), lb.rstrip(b"\n").partition(b"\t")
            if pa_[0] != pb_[0]:
                raise AssertionError("row order")
            xa = {x for x in pa_[2].split(b",") if x}
            xb = {x for x in pb_[2].split(b",") if x}
            rows += 1
            n06 += len(xa)
            n09 += len(xb)
            d1, d2 = len(xb - xa), len(xa - xb)
            new += d1
            gone += d2
            changed += bool(d1 or d2)
    return {"rows": rows, "B06_candidates": n06, "B09_candidates": n09, "candidate_change": n09 / n06 - 1, "new_candidates": new, "displaced_candidates": gone,
            "rows_changed": changed, "candidates_per_s1": {"B06": n06 / rows, "B09": n09 / rows}}


def cmd_testdiag(c):
    rep = {"label": "UNLABELED test diagnostics (distribution only)"}
    rep["candidates"] = cand_changes(E.OUTFILES["B06"] / "candidate_pairs.tsv", E.OUTFILES["B09"] / "candidate_pairs.tsv")
    ids, cnt = X.row_counts({"B06": E.OUTFILES["B06"] / "matching_results.tsv", "B09": E.OUTFILES["B09"] / "matching_results.tsv"})
    if ids != E._ids(c.paths, "test", "source1").to_pylist():
        raise AssertionError("row order differs from test S1 order")
    kb, kx, add, rem = cnt["B06"]["k"], cnt["B09"]["k"], cnt["B09"]["add"], cnt["B09"]["rem"]
    rep["predictions"] = {g: {"s1": int(m.sum()), "B06": int(kb[m].sum()), "B09": int(kx[m].sum()), "change": float(kx[m].sum() / kb[m].sum() - 1),
                              "added_vs_B06": int(add[m].sum()), "removed_vs_B06": int(rem[m].sum()), "B06_empty_rate": float((kb[m] == 0).mean()),
                              "B09_empty_rate": float((kx[m] == 0).mean())}
                          for g, m in [("all", np.ones(len(kb), bool))] + [(g, c.ctry_te == g) for g in ("france", "india", "us")]}
    d06 = E.decisions(c, "test", E.TEST_TAG["B06"], E.threshold("B06"))
    d09 = E.decisions(c, "test", E.TEST_TAG["B09"], E.threshold("B09"))
    L06 = list_codes(c, "test", E.TEST_TAG["B06"])
    acc_new = ~member(d09["codes"], L06)
    del L06
    ct = c.ctry_te
    rep["accepted_from_new_candidates"] = {"all": int(acc_new.sum()), **{g: int((acc_new & (ct[d09["codes"] // SHIFT] == g)).sum()) for g in ("france", "india", "us")},
                                           "score_distribution": summary(d09["scores"][acc_new])}
    rep["files_match_recomputed_decisions"] = {"B06": int(kb.sum()) == len(d06["codes"]), "B09": int(kx.sum()) == len(d09["codes"])}
    rep["conflicts"] = {s: {"before_d01": d["conflicts_before_d01"]["conflicted_records"], "after_d01": d["conflicts_after_d01"]["conflicted_records"]}
                        for s, d in (("B06", d06), ("B09", d09))}
    weak = {}
    for s, d in (("B06", d06), ("B09", d09)):
        w = d["scores"] < np.float32(0.8)
        weak[s] = {"all": float(w.sum() / len(ct)), **{g: float((w & (ct[d["codes"] // SHIFT] == g)).sum() / (ct == g).sum()) for g in ("france", "india", "us")}}
    rep["weak_accepted(score<0.8)_per_s1"] = weak | {"relative_change": {g: weak["B09"][g] / weak["B06"][g] - 1 for g in weak["B06"]}}
    rep["accepted_score_distribution"] = {"B06": summary(d06["scores"]), "B09": summary(d09["scores"])}
    dens = {}
    for s in ("B06", "B09"):
        v = np.concatenate([pq.read_table(p, columns=["score"]).column("score").to_numpy() for p in sorted(c.paths.scored_dir("test", E.TEST_TAG[s]).glob("*.parquet"))])
        s1 = np.concatenate([pq.read_table(p, columns=["s1_row"]).column("s1_row").to_numpy() for p in sorted(c.paths.scored_dir("test", E.TEST_TAG[s]).glob("*.parquet"))]).astype(np.int64)
        dens[s] = {"candidate_score": summary(v), "mid(0.1-0.67)_per_s1": {g: float(((v >= 0.1) & (v < 0.67) & (ct[s1] == g)).sum() / (ct == g).sum()) for g in ("france", "india", "us")},
                   "weak(<0.5)_per_s1": float((v < 0.5).sum() / len(ct))}
        del v, s1
    rep["stage1_candidate_score_density"] = dens
    write_json(RUN / "test_diagnostics.json", rep)
    LOG.info("testdiag: %s | weak %s | conflicts %s", rep["predictions"]["all"], rep["weak_accepted(score<0.8)_per_s1"]["relative_change"], rep["conflicts"])


# ----------------------------------------------------------------------------- India / structure of recovered pairs
def cmd_india(c):
    f06 = np.load("experiments/runs/FINAL_RESEARCH_AUDIT/fates_B06_k1000.npz")
    f09 = np.load(OUT / "fates_B09_k1000.npz")
    gcode, r06, r09, fl9, ctry = f06["gcode"].astype(np.int64), f06["g_rank"], f09["g_rank"], f09["g_flags"], f06["g_country"]
    if not np.array_equal(gcode, f09["gcode"].astype(np.int64)):
        raise AssertionError("fates not aligned")
    d09 = E.decisions(c, "train", E.TAG["B09"], E.threshold("B09"))
    d06 = E.decisions(c, "train", E.TAG["B06"], E.threshold("B06"))
    acc09, acc06 = member(gcode, d09["codes"]), member(gcode, d06["codes"])
    rec = (r06 >= 20) & (r09 < 20)  # entered the top-20 lists
    lost = (r06 < 20) & (r09 >= 20)  # displaced true pairs
    idx = np.flatnonzero(rec | lost)
    xmap = FA.xmap_b06(c)
    name_df, word_df = FA.pool_df_fast(c, xmap)
    s1v, pv = FA.load_views(c, "train", gcode[idx] // SHIFT, gcode[idx] % SHIFT, xmap)
    prof = {}
    for i in idx.tolist():
        a, b = s1v.get(int(gcode[i] // SHIFT)), pv.get(int(gcode[i] % SHIFT))
        if a is not None and b is not None:
            prof[i] = pair_profile(a, b, name_df.get(a["country"], {}), word_df.get(a["country"], {}))
    def cats(sel):
        rows = [prof[i] for i in np.flatnonzero(sel).tolist() if i in prof]
        n = max(len(rows), 1)
        share = lambda f: float(sum(bool(f(r)) for r in rows) / n)  # noqa: E731
        return {"pairs": len(rows), "shared_house_number": share(lambda r: r["shared_number"]), "name+number(shared token and number)": share(lambda r: r["xlit_shared"] > 0 and r["shared_number"]),
                "name_only": share(lambda r: r["name_only"]), "address_only": share(lambda r: r["address_only"]), "cross_script": share(lambda r: r["script_mismatch"]),
                "record_address_missing": share(lambda r: r["rec_addr_empty"])}
    rep = {"population": "frozen validation true pairs (diagnostic)", "labels": "[M]"}
    for g in ("india", "us"):
        m = ctry == g
        rep[g] = {"true_pairs": int(m.sum()), "no_key_B06": int((m & (r06 >= 1000)).sum()), "no_key_recovered_any_rank": int((m & (r06 >= 1000) & (r09 < 1000)).sum()),
                  "entered_top20": int((m & rec).sum()), "entered_top20_via_name_num_flag": int((m & rec & ((fl9 & NN_BIT) > 0)).sum()),
                  "displaced_from_top20": int((m & lost).sum()), "entered_top20_and_accepted_by_B09": int((m & rec & acc09).sum()),
                  "displaced_and_accepted_by_B06": int((m & lost & acc06).sum()), "recall@20": {"B06": float((r06[m] < 20).mean()), "B09": float((r09[m] < 20).mean())},
                  "decision_recall": {"B06": float(acc06[m].mean()), "B09": float(acc09[m].mean())},
                  "structure_of_entered_top20": cats(m & rec), "structure_of_entered_and_accepted": cats(m & rec & acc09), "structure_of_displaced": cats(m & lost)}
    write_json(RUN / "india_analysis.json", rep)
    LOG.info("india: %s", {g: {k: rep[g][k] for k in ("entered_top20", "entered_top20_and_accepted_by_B09", "displaced_from_top20", "decision_recall")} for g in ("india", "us")})


# ----------------------------------------------------------------------------- gates A-P (notes.md 1.3)
def cmd_gates(c):
    ev = json.loads((OUT / "eval.json").read_text(encoding="utf-8"))
    base = ev["competitions"]["T1(base)"]["metrics"]
    bf = json.loads((OUT / "blocking_forensics.json").read_text(encoding="utf-8"))
    td = json.loads((RUN / "test_diagnostics.json").read_text(encoding="utf-8"))
    lk = json.loads((OUT / "leakage.json").read_text(encoding="utf-8"))
    det = json.loads((OUT / "determinism.json").read_text(encoding="utf-8"))
    rt = json.loads((OUT / "runtime_memory.json").read_text(encoding="utf-8"))
    sub = json.loads((RUN / "submission_summary.json").read_text(encoding="utf-8"))
    off = (OUT / "official_validator.txt").read_text(encoding="utf-8")
    dd = lambda comp, pop, key="f05": ev["competitions"][comp]["metrics"]["B09"][pop][key] - ev["competitions"][comp]["metrics"]["B06"][pop][key]  # noqa: E731
    g = {}
    g["A_blocking_recall_gain>=0.004"] = {"value": bf["recall_all"]["delta@20"], "pass": bf["recall_all"]["delta@20"] >= 0.004}
    g["B_f05>=+0.002"] = {p: {"delta": dd("T1(base)", p), "pass": dd("T1(base)", p) >= 0.002} for p in ("validation", "GATE-B", "E_half_B")}
    g["C_precision_drop<=0.001"] = {p: {"delta": dd("T1(base)", p, "precision"), "pass": dd("T1(base)", p, "precision") >= -0.001} for p in ("validation", "GATE-B", "E_half_B")}
    hv = {h: dd("T1(base)", h) for h in ("validation_half_A", "validation_half_B")}
    g["D_validation_halves_agree"] = {"deltas": hv, "pass": all(v > 0 for v in hv.values())}
    ind = {p: base["B09"][p]["by_country"]["india"]["f05"] - base["B06"][p]["by_country"]["india"]["f05"] for p in ("validation", "GATE-B", "E_half_B")}
    g["E_india_not_worse(>=-0.001)"] = {"deltas": ind, "pass": all(v >= -0.001 for v in ind.values())}
    g["F_shadow_test_matched>=+0.002"] = {"delta": dd("test_matched(v1 random orphan)", "SHADOW-TEST-MATCHED"), "pass": dd("test_matched(v1 random orphan)", "SHADOW-TEST-MATCHED") >= 0.002}
    g["G_shadow_chain_orphaned>=+0.002"] = {"delta": dd("chain_orphaned(v1)", "SHADOW-CHAIN-ORPHANED"), "pass": dd("chain_orphaned(v1)", "SHADOW-CHAIN-ORPHANED") >= 0.002}
    da = ev["density_analysis"]
    g["H_no_material_density_decay(DEV)"] = {"decay": da["SHADOW-DEV"]["decay_vs_T1"], "threshold": da["SHADOW-DEV"]["material_threshold"], "pass": not da["SHADOW-DEV"]["any_material_decay"],
                                             "locked_reported": {p: da[p]["any_material_decay"] for p in ("SHADOW-LOCKED-A", "SHADOW-LOCKED-B")}}
    dev = dd("T1(base)", "SHADOW-DEV")
    g["I_locked_within_0.0015_of_dev(confirmatory)"] = {p: {"delta": dd("T1(base)", p), "dev": dev, "pass": abs(dd("T1(base)", p) - dev) <= 0.0015} for p in ("SHADOW-LOCKED-A", "SHADOW-LOCKED-B")}
    wk = td["weak_accepted(score<0.8)_per_s1"]["relative_change"]["all"]
    pc_all, pc_fr = td["predictions"]["all"]["change"], td["predictions"]["france"]["change"]
    g["J_weak_acceptance_and_prediction_change"] = {"weak_accepted_relative_change": wk, "prediction_change": pc_all, "france_change": pc_fr,
                                                    "pass": wk <= 0.10 and pc_all <= 0.02 and pc_fr <= 0.03}
    g["K_no_leakage"] = {"pass": bool(lk["all_ok"])}
    g["L_determinism"] = {"pass": bool(det["byte_identical"])}
    g["M_strict_validator"] = {"pass": sub.get("strict_validation_errors") == []}
    g["N_official_validator"] = {"pass": "PASS" in off}
    g["O_memory<=5GB(pipeline)"] = {"peaks": {k: v["peak_gb"] for k, v in rt["pipeline"].items()}, "pass": all(v["peak_gb"] <= 5.0 for v in rt["pipeline"].values())}
    cb, ca = td["conflicts"]["B06"]["before_d01"], td["conflicts"]["B09"]["before_d01"]
    er = td["predictions"]["all"]["B09_empty_rate"] - td["predictions"]["all"]["B06_empty_rate"]
    g["P_test_side_plausible"] = {"conflicts_before_d01_change": ca / cb - 1, "empty_rate_change": er, "pass": (ca / cb - 1) <= 0.25 and abs(er) <= 0.01}
    flat = []
    for k, v in g.items():
        if "pass" in v:
            flat.append(v["pass"])
        else:
            flat += [x["pass"] for x in v.values() if isinstance(x, dict) and "pass" in x]
    g["all_pass"] = bool(all(flat))
    write_json(RUN / "gates.json", g)
    LOG.info("gates all_pass=%s", g["all_pass"])


CMDS = {"decompose": cmd_decompose, "testdiag": cmd_testdiag, "india": cmd_india, "gates": cmd_gates}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=list(CMDS))
    a = ap.parse_args()
    setup_logging()
    configure_arrow()
    t0 = time.perf_counter()
    with PeakMemory() as pm:
        CMDS[a.cmd](X.Ctx())
    LOG.info("%s done in %.0fs, peak %.2f GB", a.cmd, time.perf_counter() - t0, pm.peak_gb)


if __name__ == "__main__":
    main()
