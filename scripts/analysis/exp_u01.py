"""EXP-U-01: S-03 + (S-04 final pairs on candidates new to the B-10 blocker, record not matched by S-03). Read-only w.r.t. all
models / scores (S-03, S-04). Pre-declared gates: experiments/runs/EXP-U-01/notes.md.

    python scripts/analysis/exp_u01.py eval | submit
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
import exp_s04 as S4  # noqa: E402

from ber.pipeline.run_baseline import _ids, pool_ids  # noqa: E402
from ber.submission.writer import CANDIDATE_HEADER, MATCHING_HEADER, write_id_lists  # noqa: E402
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, write_json  # noqa: E402

RUN = Path("experiments/runs/EXP-U-01")
SHIFT = S4.SHIFT


def combine(s03, s04, b09cand):
    """S-03 plus S-04 pairs outside the B-09 candidate lists whose record S-03 does not match."""
    new = s04[~E.is_true(s04, b09cand)]
    taken = np.unique(s03 % SHIFT)
    new = new[~E.is_true(new % SHIFT, taken)]
    return np.sort(np.concatenate([s03, new])), new


def cmd_eval(c):
    drop, pops, weights = S4.pops_for(c, "T1(base)")
    t0 = time.perf_counter()
    with PeakMemory() as pm:
        res, d09, s03, d04, s04, _ = S4.run_comp(c, "T1(base)", pops, weights, drop)
        b09cand = S4.cand_codes(c, S4.TAG09, np.ones(c.n, bool))
        u, new = combine(s03, s04, b09cand)
        del b09cand
        gts = np.sort(c.gtc)
        res["U01"] = E.pop_metrics(c, dict(d09, codes=u), pops, weights, s03)
        yt = E.is_true(new, gts)
        n_add = np.bincount(new // SHIFT, minlength=c.n)
        n_add_t = np.bincount(new[yt] // SHIFT, minlength=c.n)
        res["U01_added"] = {p: {"added_per_s1": float(n_add[m].mean()), "added_true_share": float(n_add_t[m].sum() / max(n_add[m].sum(), 1))} for p, m in pops.items() if not p.endswith("[W]")}
    out = {"populations": {k: int(v.sum()) for k, v in pops.items()}, "metrics": res, "seconds": round(time.perf_counter() - t0, 1), "peak_gb": round(pm.peak_gb, 2)}
    write_json(RUN / "eval.json", out)
    b = res
    d = lambda p, k="f05": b["U01"][p][k] - b["S03"][p][k]  # noqa: E731
    P3 = ("validation", "GATE-B", "E_half_B")
    g = {"U1": {p: d(p) for p in P3}, "U2": {p: d(p) for p in ("validation_half_A", "validation_half_B")}, "U3": {p: d(p, "precision") for p in P3},
         "U4": {p: {cc: b["U01"][p]["by_country"][cc]["f05"] - b["S03"][p]["by_country"][cc]["f05"] for cc in ("india", "us")} for p in P3},
         "U5": {"SHADOW-DEV[W]": d("SHADOW-DEV[W]"), "SHADOW-DEV": d("SHADOW-DEV")}}
    g["pass"] = {"U1": all(v >= 0.0005 for v in g["U1"].values()), "U2": all(v > 0 for v in g["U2"].values()), "U3": all(v >= -0.0005 for v in g["U3"].values()),
                 "U4": all(v >= -0.0005 for p in P3 for v in g["U4"][p].values()), "U5": g["U5"]["SHADOW-DEV[W]"] > 0}
    g["offline_pass"] = all(g["pass"].values())
    g["added"] = res["U01_added"]
    write_json(RUN / "gates_offline.json", g)
    for p in P3 + ("validation_half_A", "validation_half_B", "SHADOW-DEV[W]"):
        LOG.info("%-18s S03 %.4f (P %.4f R %.4f) U01 %.4f (P %.4f R %.4f) delta %+.5f", p, b["S03"][p]["f05"], b["S03"][p]["precision"], b["S03"][p]["recall"],
                 b["U01"][p]["f05"], b["U01"][p]["precision"], b["U01"][p]["recall"], d(p))
    LOG.info("added %s | gates %s offline_pass=%s | %.0fs peak %.2f GB", res["U01_added"]["E_half_B"], g["pass"], g["offline_pass"], time.perf_counter() - t0, pm.peak_gb)


def test_cands(c, tag):
    s1s, pools, rks = [], [], []
    for p in sorted(c.paths.scored_dir("test", tag).glob("*.parquet")):
        tb = pq.read_table(p, columns=["s1_row", "pool_row", "cheap_rank"])
        s1s.append(tb.column("s1_row").to_numpy().astype(np.int64))
        pools.append(tb.column("pool_row").to_numpy().astype(np.int64))
        rks.append(tb.column("cheap_rank").to_numpy().astype(np.int64))
    return np.concatenate(s1s), np.concatenate(pools), np.concatenate(rks)


def cmd_submit(c):
    c3, p3 = S3.cfgp()
    c4, p4 = S4.cfg4()
    out_dir = RUN / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    with PeakMemory() as pm:
        d09 = E.decisions(c, "test", "scored_EXP-B-09", S4.T09)
        s03 = S4.veto(c3, p3, d09["codes"], "test")[0]
        d04 = E.decisions(c, "test", "scored_EXP-S-04", S4.t04())
        s04 = S4.veto(c4, p4, d04["codes"], "test")[0]
        del d09, d04
        a1, a2, a3 = test_cands(c, "scored_EXP-B-09")
        b1, b2, b3 = test_cands(c, "scored_EXP-S-04")
        ca, cb = a1 * SHIFT + a2, b1 * SHIFT + b2
        ca_sorted = np.sort(ca)
        only_b = ~E.is_true(cb, ca_sorted)
        u, new = combine(s03, s04, ca_sorted)
        # union candidate list: B-09 list (cheap-rank order) then the S-04-only candidates (cheap-rank order), per S1
        cs1 = np.concatenate([a1, b1[only_b]])
        cpool = np.concatenate([a2, b2[only_b]])
        grp = np.concatenate([np.zeros(len(a1), np.int64), np.ones(int(only_b.sum()), np.int64)])
        rk = np.concatenate([a3, b3[only_b]])
        o = np.lexsort((rk, grp, cs1))
        cs1, cpool = cs1[o], cpool[o]
        del a1, a2, a3, b1, b2, b3, ca, cb, grp, rk, o
        code = cs1 * SHIFT + cpool
        if len(np.unique(code)) != len(code):
            raise AssertionError("duplicate candidate")
        match = E.is_true(code, u)
        if int(match.sum()) != len(u):
            raise AssertionError("a match is outside the union candidate set")
        n = len(c.ctry_te)
        ct = c.ctry_te
        k03, kn = np.bincount(s03 // SHIFT, minlength=n), np.bincount(new // SHIFT, minlength=n)
        diag = {"label": "UNLABELED test diagnostics", "S03_pairs": int(len(s03)), "added_pairs": int(len(new)), "U01_pairs": int(len(u)),
                "candidates_B09": int(len(ca_sorted)), "candidates_union": int(len(code)),
                "by_country": {g: {"s1": int(m.sum()), "added_per_s1": float(kn[m].mean()), "S03_pred_per_s1": float(k03[m].mean()),
                                   "prediction_change": float(kn[m].sum() / max(k03[m].sum(), 1))}
                               for g, m in [("all", np.ones(n, bool))] + [(g, ct == g) for g in ("france", "india", "us")]}}
        s1_ids = _ids(c.paths, "test", "source1").to_pylist()
        ids = pool_ids(c.paths, "test")
        cst = write_id_lists(out_dir / "candidate_pairs.tsv", CANDIDATE_HEADER, s1_ids, cs1, (cpool, ids))
        mst = write_id_lists(out_dir / "matching_results.tsv", MATCHING_HEADER, s1_ids, cs1[match], (cpool[match], ids))
    ev = json.loads((RUN / "eval.json").read_text(encoding="utf-8"))
    eb = ev["metrics"]["U01_added"]["E_half_B"]["added_per_s1"]
    diag["train_E_half_B_added_per_s1"] = eb
    diag["test_over_train_added"] = diag["by_country"]["all"]["added_per_s1"] / max(eb, 1e-12)
    diag["france_over_overall_added"] = diag["by_country"]["france"]["added_per_s1"] / max(diag["by_country"]["all"]["added_per_s1"], 1e-12)
    diag["U6_pass"] = bool(diag["test_over_train_added"] <= 1.5 and diag["france_over_overall_added"] <= 1.5)
    diag["files"] = {"candidate": cst, "matching": mst}
    diag["sha256"] = {f: hashlib.sha256((out_dir / f).read_bytes()).hexdigest() for f in ("matching_results.tsv", "candidate_pairs.tsv")}
    diag["peak_gb"] = round(pm.peak_gb, 2)
    write_json(RUN / "test_diagnostics.json", diag)
    LOG.info("submit: %s", {k: v for k, v in diag.items() if k not in ("files",)})


CMDS = {"eval": cmd_eval, "submit": cmd_submit}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=list(CMDS))
    a = ap.parse_args()
    setup_logging()
    configure_arrow()
    RUN.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    CMDS[a.cmd](X.Ctx())
    LOG.info("%s done in %.0fs", a.cmd, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
