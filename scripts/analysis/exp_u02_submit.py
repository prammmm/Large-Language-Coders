"""EXP-U-02 test output: U-01 (S-03 + S-04 acceptances on candidates new to the B-10 blocker) + V2 (B-08 top-30 stage-1 + D-01 @ 0.68
acceptances on candidates in neither the B-09 nor the S-04 list, record not matched). Candidate file = union of the three scored lists
(B-09 top-20, S-04 top-20, B-08 top-30). Read-only w.r.t. all models / scores.

    python scripts/analysis/exp_u02_submit.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_b09_eval as E  # noqa: E402
import exp_s02_shadow as X  # noqa: E402
import exp_s03 as S3  # noqa: E402
import exp_s04 as S4  # noqa: E402
import exp_u01 as U1  # noqa: E402
import exp_u02 as U2  # noqa: E402

from ber.pipeline.run_baseline import _ids, pool_ids  # noqa: E402
from ber.submission.writer import CANDIDATE_HEADER, MATCHING_HEADER, write_id_lists  # noqa: E402
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, write_json  # noqa: E402

RUN = Path("experiments/runs/EXP-U-02")
SHIFT = S4.SHIFT


def main():
    setup_logging()
    configure_arrow()
    c = X.Ctx()
    c3, p3 = S3.cfgp()
    c4, p4 = S4.cfg4()
    out_dir = RUN / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    with PeakMemory() as pm:
        d09 = E.decisions(c, "test", "scored_EXP-B-09", S4.T09)
        s03 = S4.veto(c3, p3, d09["codes"], "test")[0]
        del d09
        d04 = E.decisions(c, "test", "scored_EXP-S-04", S4.t04())
        s04 = S4.veto(c4, p4, d04["codes"], "test")[0]
        del d04
        b08 = E.decisions(c, "test", "scored_EXP-B-08", U2.T08)["codes"]
        lists = [U1.test_cands(c, t) for t in ("scored_EXP-B-09", "scored_EXP-S-04", "scored_EXP-B-08")]
        codes = [a * SHIFT + b for a, b, _ in lists]
        c09, c04 = np.sort(codes[0]), np.sort(codes[1])
        u01, _ = U1.combine(s03, s04, c09)
        b08 = b08[~E.is_true(b08, c09) & ~E.is_true(b08, c04)]
        u02, add2 = U2.add_unmatched(u01, b08)
        # union candidate list: B-09 list, then S-04-only, then B-08-only candidates (each in its own cheap-rank order), per S1
        seen = c09
        parts = []
        for g, ((a1, a2, a3), cd) in enumerate(zip(lists, codes)):
            keep = np.ones(len(cd), bool) if g == 0 else ~E.is_true(cd, seen)
            parts.append((a1[keep], a2[keep], np.full(int(keep.sum()), g, np.int64), a3[keep]))
            if g:
                seen = np.sort(np.concatenate([seen, cd[keep]]))
        cs1 = np.concatenate([p[0] for p in parts]); cpool = np.concatenate([p[1] for p in parts])
        grp = np.concatenate([p[2] for p in parts]); rk = np.concatenate([p[3] for p in parts])
        o = np.lexsort((rk, grp, cs1))
        cs1, cpool = cs1[o], cpool[o]
        del lists, codes, parts, grp, rk, o, seen
        code = cs1 * SHIFT + cpool
        if len(np.unique(code)) != len(code):
            raise AssertionError("duplicate candidate")
        match = E.is_true(code, u02)
        if int(match.sum()) != len(u02):
            raise AssertionError("a match is outside the union candidate set")
        n = len(c.ctry_te)
        ct = c.ctry_te
        k01, ka = np.bincount(u01 // SHIFT, minlength=n), np.bincount(add2 // SHIFT, minlength=n)
        diag = {"label": "UNLABELED test diagnostics", "U01_pairs": int(len(u01)), "V2_added_pairs": int(len(add2)), "U02_pairs": int(len(u02)),
                "candidates_union": int(len(code)),
                "by_country": {g: {"s1": int(m.sum()), "added_per_s1": float(ka[m].mean()), "prediction_change_vs_U01": float(ka[m].sum() / max(k01[m].sum(), 1))}
                               for g, m in [("all", np.ones(n, bool))] + [(g, ct == g) for g in ("france", "india", "us")]}}
        s1_ids = _ids(c.paths, "test", "source1").to_pylist()
        ids = pool_ids(c.paths, "test")
        cst = write_id_lists(out_dir / "candidate_pairs.tsv", CANDIDATE_HEADER, s1_ids, cs1, (cpool, ids))
        mst = write_id_lists(out_dir / "matching_results.tsv", MATCHING_HEADER, s1_ids, cs1[match], (cpool[match], ids))
    g = json.loads((RUN / "gates_offline.json").read_text(encoding="utf-8"))["V2"]
    eb = g["added_per_s1_E_half_B"]
    diag["train_E_half_B_added_per_s1"] = eb
    diag["test_over_train_added"] = diag["by_country"]["all"]["added_per_s1"] / max(eb, 1e-12)
    diag["france_over_overall_added"] = diag["by_country"]["france"]["added_per_s1"] / max(diag["by_country"]["all"]["added_per_s1"], 1e-12)
    diag["G_pass"] = bool(diag["test_over_train_added"] <= 1.5 and diag["france_over_overall_added"] <= 1.5)
    diag["files"] = {"candidate": cst, "matching": mst}
    diag["sha256"] = {f: hashlib.sha256((out_dir / f).read_bytes()).hexdigest() for f in ("matching_results.tsv", "candidate_pairs.tsv")}
    diag["peak_gb"] = round(pm.peak_gb, 2)
    write_json(RUN / "test_diagnostics.json", diag)
    LOG.info("submit: %s", {k: v for k, v in diag.items() if k != "files"})


if __name__ == "__main__":
    main()
