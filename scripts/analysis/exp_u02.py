"""EXP-U-02: V1 (remaining S-04 acceptances) and V2 (B-08 top-30 acceptances on candidates new to both B-09 and S-04 lists) on top of
U-01. Read-only w.r.t. all models / scores. Gates: experiments/runs/EXP-U-02/notes.md.

    python scripts/analysis/exp_u02.py eval
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_b09_eval as E  # noqa: E402
import exp_s02_shadow as X  # noqa: E402
import exp_s04 as S4  # noqa: E402
import exp_u01 as U1  # noqa: E402

from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, write_json  # noqa: E402

RUN = Path("experiments/runs/EXP-U-02")
SHIFT = S4.SHIFT
TAG08, T08 = "forensic_alltrain_B08", 0.68


def add_unmatched(base, cand):
    cand = cand[~E.is_true(cand, base)]
    cand = cand[~E.is_true(cand % SHIFT, np.unique(base % SHIFT))]
    return np.sort(np.concatenate([base, cand])), cand


def main():
    setup_logging()
    configure_arrow()
    RUN.mkdir(parents=True, exist_ok=True)
    c = X.Ctx()
    drop, pops, weights = S4.pops_for(c, "T1(base)")
    t0 = time.perf_counter()
    with PeakMemory() as pm:
        res, d09, s03, d04, s04, _ = S4.run_comp(c, "T1(base)", pops, weights, drop)
        allm = np.ones(c.n, bool)
        c09 = S4.cand_codes(c, S4.TAG09, allm)
        u01, _ = U1.combine(s03, s04, c09)
        del s03
        v1, add1 = add_unmatched(u01, s04)
        c04 = S4.cand_codes(c, S4.TAG04, allm)
        d08 = E.decisions(c, "train", TAG08, T08)
        b08 = d08["codes"]
        b08 = b08[~E.is_true(b08, c09) & ~E.is_true(b08, c04)]
        del c09, c04
        v2, add2 = add_unmatched(u01, b08)
        v12, add12 = add_unmatched(v1, b08)
        gts = np.sort(c.gtc)
        out = {"U01": E.pop_metrics(c, dict(d09, codes=u01), pops, weights)}
        for name, codes, add in (("V1", v1, add1), ("V2", v2, add2), ("V1+V2", v12, np.concatenate([add1, add12[~E.is_true(add12, np.sort(add1))]]))):
            out[name] = E.pop_metrics(c, dict(d09, codes=codes), pops, weights, u01)
            yt = E.is_true(np.sort(add), gts)
            na = np.bincount(add // SHIFT, minlength=c.n)
            out[name + "_added"] = {p: {"added_per_s1": float(na[m].mean()), "true_share": float(yt[pops[p][np.sort(add) // SHIFT]].mean()) if na[m].sum() else None}
                                    for p, m in pops.items() if not p.endswith("[W]")}
    P3 = ("validation", "GATE-B", "E_half_B")
    gates = {}
    for name in ("V1", "V2", "V1+V2"):
        d = lambda p, k="f05": out[name][p][k] - out["U01"][p][k]  # noqa: E731
        g = {"A": {p: d(p) for p in P3}, "B": {p: d(p) for p in ("validation_half_A", "validation_half_B")}, "C": {p: d(p, "precision") for p in P3},
             "D": {p: {cc: out[name][p]["by_country"][cc]["f05"] - out["U01"][p]["by_country"][cc]["f05"] for cc in ("india", "us")} for p in P3},
             "E": d("SHADOW-DEV[W]"), "F": out[name + "_added"]["E_half_B"]["true_share"], "added_per_s1_E_half_B": out[name + "_added"]["E_half_B"]["added_per_s1"]}
        g["pass"] = {"A": all(v >= 0.0003 for v in g["A"].values()), "B": all(v > 0 for v in g["B"].values()), "C": all(v >= -0.0005 for v in g["C"].values()),
                     "D": all(v >= -0.0003 for p in P3 for v in g["D"][p].values()), "E": g["E"] > 0, "F": (g["F"] or 0) >= 0.80}
        g["offline_pass"] = all(g["pass"].values())
        gates[name] = g
        LOG.info("%s: dF0.5 %s | halves %s | dP %s | DEV[W] %+.5f | added/S1 %.4f true %.3f | pass %s", name, {k: round(v, 5) for k, v in g["A"].items()},
                 {k: round(v, 5) for k, v in g["B"].items()}, {k: round(v, 5) for k, v in g["C"].items()}, g["E"], g["added_per_s1_E_half_B"], g["F"] or 0, g["offline_pass"])
    write_json(RUN / "eval.json", {"metrics": out, "seconds": round(time.perf_counter() - t0, 1), "peak_gb": round(pm.peak_gb, 2)})
    write_json(RUN / "gates_offline.json", gates)
    LOG.info("done %.0fs peak %.2f GB", time.perf_counter() - t0, pm.peak_gb)


if __name__ == "__main__":
    main()
