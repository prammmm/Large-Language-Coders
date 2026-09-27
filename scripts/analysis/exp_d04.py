"""EXP-D-04 (removal-only per-S1 expected-F0.5 pruning on S-03) evaluation. Deterministic; read-only w.r.t. every other
experiment (S-03 model / features / shadow feature parts re-used).

    python scripts/analysis/exp_d04.py eval | testdiag | gates

eval      every frozen competition (T1(base) first: E half B, validation + halves, GATE-B, shadow; then test-matched,
          chain, T2-T4): S-03 vs D-04 metrics, removals vs S-03, density (T1 / T1W / T4W).
testdiag  label-free test comparison of the S-03 and D-04 output files (predictions, removals per S1, by country).
gates     G1-G8 of notes.md 1.2.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.csv as pv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_b09_eval as E  # noqa: E402
import exp_s02_shadow as X  # noqa: E402
import exp_s03 as S3  # noqa: E402

from ber.decision.ef_prune import expected_f_prune  # noqa: E402
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, write_json  # noqa: E402

RUN = Path("experiments/runs/EXP-D-04")
OUT = RUN / "analysis"
ORDER = ["T1(base)", "T4", "test_matched(v1 random orphan)", "chain_orphaned(v1)", "T2", "T3"]


def cmd_eval(c):
    cfg, paths = S3.cfgp()
    comps = X.competitions(c)
    tw, ess, share = E.tier_weights(c, comps)
    fd = S3.fdirs(paths)
    rep = {"systems": {"S03": "control: B-09 + D-01 AND stage-2 veto @ 0.69", "D04": "S-03 + removal-only per-S1 expected-F0.5 pruning"},
           "note": "removals_per_s1_vs_B06 of D04 = removals vs S-03", "competitions": {}}
    for comp in ORDER:
        _, drop, pops, weights = comps[comp]
        pops, weights = dict(pops), dict(weights)
        la_key = "T1" if comp == "T1(base)" else ("T4" if comp == "T4" else None)
        if la_key:
            for pn in ("SHADOW-DEV", "SHADOW-LOCKED-A", "SHADOW-LOCKED-B"):
                pops[f"{pn}[W]"], weights[f"{pn}[W]"] = pops[pn], tw[la_key][pn]
        t0 = time.perf_counter()
        with PeakMemory() as pm:
            d09 = E.decisions(c, "train", S3.TAG, S3.T1_B09, drop)
            s1, pool, prob, sc, meta = S3.s03_arrays(cfg, paths, "train", fd.get(comp))
            code, cm, fin = S3.decide_s03(d09["codes"], s1, pool, prob, sc, meta)
            keep = expected_f_prune(s1, pool, prob, fin)
            if (keep & ~fin).any():
                raise AssertionError("D-04 added a pair")
            s03, d04 = np.sort(code[fin]), np.sort(code[keep])
            del s1, pool, prob, sc, code, cm, fin, keep
            res = {"S03": E.pop_metrics(c, dict(d09, codes=s03), pops, weights, d09["codes"]),
                   "D04": E.pop_metrics(c, dict(d09, codes=d04), pops, weights, s03)}
            res["removed_vs_S03"] = int(len(s03) - len(d04))
            gts = np.sort(c.gtc)
            rem = np.setdiff1d(s03, d04)
            res["removed_true_share"] = float(E.is_true(rem, gts).mean()) if len(rem) else None
        rep["competitions"][comp] = {"populations": {k: int(v.sum()) for k, v in pops.items()}, "metrics": res,
                                     "seconds": round(time.perf_counter() - t0, 1), "peak_gb": round(pm.peak_gb, 2)}
        for pn in [k for k in pops if not k.endswith("[W]")][:8]:
            a, b = res["S03"][pn], res["D04"][pn]
            LOG.info("%-24s %-22s S03 %.4f (P %.4f) D04 %.4f (P %.4f) delta %+.4f", comp, pn, a["f05"], a["precision"], b["f05"], b["precision"], b["f05"] - a["f05"])
        LOG.info("%s: removed %d (true share %s) | %.0fs peak %.2f GB", comp, res["removed_vs_S03"], res["removed_true_share"], time.perf_counter() - t0, pm.peak_gb)
        write_json(OUT / "eval.json", rep)
        if comp == "T4":
            rep["density"] = density(rep)
            write_json(OUT / "eval.json", rep)
    rep["density"] = density(rep)
    write_json(OUT / "eval.json", rep)


def density(rep) -> dict:
    C = rep["competitions"]
    out = {}
    for pop in ("SHADOW-DEV", "SHADOW-LOCKED-A", "SHADOW-LOCKED-B"):
        rows = {}
        for t, ck, suf in (("T1", "T1(base)", ""), ("T4", "T4", ""), ("T1W", "T1(base)", "[W]"), ("T4W", "T4", "[W]")):
            if ck in C:
                a, b = C[ck]["metrics"]["S03"][pop + suf], C[ck]["metrics"]["D04"][pop + suf]
                rows[t] = {"S03_f05": a["f05"], "D04_f05": b["f05"], "adv": b["f05"] - a["f05"], "d_precision": b["precision"] - a["precision"]}
        out[pop] = rows
    return out


def read_pairs(path):
    t = pv.read_csv(path, parse_options=pv.ParseOptions(delimiter="\t"), convert_options=pv.ConvertOptions(column_types={"matched_entity_ids": "string"}))
    ids = t.column(0).to_pylist()
    lists = t.column(1).to_pylist()
    return ids, np.array([0 if not v else v.count(",") + 1 for v in lists], np.int64)


def cmd_testdiag(c):
    ids_a, ka = read_pairs(Path("experiments/runs/EXP-S-03/output/matching_results.tsv"))
    ids_b, kb = read_pairs(RUN / "output" / "matching_results.tsv")
    if ids_a != ids_b:
        raise AssertionError("S1 rows differ")
    cty = np.asarray(c.ctry_te)
    ev = json.loads((OUT / "eval.json").read_text(encoding="utf-8"))
    eb = ev["competitions"]["T1(base)"]["metrics"]["D04"]["E_half_B"]["removals_per_s1_vs_B06"]
    rep = {"label": "UNLABELED test diagnostics (S-03 vs D-04 files)", "E_half_B_removals_per_s1": eb}
    for g in ("all", "france", "india", "us"):
        m = np.ones(len(ka), bool) if g == "all" else cty == g
        rep[g] = {"s1": int(m.sum()), "S03_pairs": int(ka[m].sum()), "D04_pairs": int(kb[m].sum()), "removals_per_s1": float((ka[m] - kb[m]).sum() / max(m.sum(), 1)),
                  "S03_empty_rate": float((ka[m] == 0).mean()), "D04_empty_rate": float((kb[m] == 0).mean()), "prediction_change": float(kb[m].sum() / max(ka[m].sum(), 1) - 1)}
    rep["test_over_E_half_B_removals"] = rep["all"]["removals_per_s1"] / max(eb, 1e-12)
    rep["france_over_overall_removals"] = rep["france"]["removals_per_s1"] / max(rep["all"]["removals_per_s1"], 1e-12)
    write_json(RUN / "test_diagnostics.json", rep)
    LOG.info("testdiag: %s", {g: rep[g] for g in ("all", "france", "india", "us")})


def cmd_gates(c):
    ev = json.loads((OUT / "eval.json").read_text(encoding="utf-8"))
    base = ev["competitions"]["T1(base)"]["metrics"]
    d = lambda pn, k="f05": base["D04"][pn][k] - base["S03"][pn][k]  # noqa: E731
    three = ("validation", "GATE-B", "E_half_B")
    td = json.loads((RUN / "test_diagnostics.json").read_text(encoding="utf-8")) if (RUN / "test_diagnostics.json").exists() else None
    dens = ev.get("density", {}).get("SHADOW-DEV", {})
    g = {"G1": {"rule": ">= +0.001 on validation, GATE-B, E half B", "value": {p: d(p) for p in three}, "pass": all(d(p) >= 0.001 for p in three)},
         "G2": {"rule": "both validation halves > 0", "value": {p: d(p) for p in ("validation_half_A", "validation_half_B")},
                "pass": all(d(p) > 0 for p in ("validation_half_A", "validation_half_B"))},
         "G3": {"rule": "precision >= S-03 on the three", "value": {p: d(p, "precision") for p in three}, "pass": all(d(p, "precision") >= 0 for p in three)},
         "G4": {"rule": "India / US delta >= -0.0005 on the three",
                "value": {p: {cc: base["D04"][p]["by_country"][cc]["f05"] - base["S03"][p]["by_country"][cc]["f05"] for cc in ("india", "us")} for p in three}}}
    g["G4"]["pass"] = all(v >= -0.0005 for p in three for v in g["G4"]["value"][p].values())
    if all(k in dens for k in ("T1", "T1W", "T4W")):
        g["G5"] = {"rule": "adv(T1W), adv(T4W) >= adv(T1) - 0.0005 (SHADOW-DEV)", "value": {k: dens[k]["adv"] for k in ("T1", "T4", "T1W", "T4W") if k in dens},
                   "pass": all(dens[k]["adv"] >= dens["T1"]["adv"] - 0.0005 for k in ("T1W", "T4W"))}
    if td:
        r = td["test_over_E_half_B_removals"]
        fr = td["france_over_overall_removals"]
        g["G6"] = {"rule": "test removals/S1 in 0.5x-1.5x of E half B; France within +-50% of overall", "value": {"test/E_B": r, "france/overall": fr},
                   "pass": bool(0.5 <= r <= 1.5 and 0.5 <= fr <= 1.5)}
    g["evaluated"] = sorted(k for k in g if k.startswith("G"))
    g["all_evaluated_pass"] = all(g[k]["pass"] for k in g["evaluated"])
    write_json(RUN / "gates.json", g)
    LOG.info("gates: %s", {k: (g[k]["pass"], g[k]["value"]) for k in g["evaluated"]})


CMDS = {"eval": cmd_eval, "testdiag": cmd_testdiag, "gates": cmd_gates}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=list(CMDS))
    a = ap.parse_args()
    setup_logging()
    configure_arrow()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    CMDS[a.cmd](X.Ctx())
    LOG.info("%s done in %.0fs", a.cmd, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
