"""EXP-B-08 (top-K 20 -> 30, stage-1 retrained) vs the control EXP-B-06 + D-01 @ 0.67. Deterministic, seed 42.

    python scripts/analysis/exp_b08.py score-alltrain | sanity | eval | decompose | testdiag | budget | keyforensic | leakage | gates

score-alltrain  score ALL train S1 with the B-08 model/blocker (same code path as `infer`) -> data/processed/train/forensic_alltrain_B08
sanity          validation F0.5 from the all-train parts (threshold only) == the `train` stage metric (same pairs, same model)
eval            B-06 vs B-08 (threshold + D-01, full competition) on local, shadow v2 and density-tier populations (streamed)
decompose       where the change in F0.5 comes from: ranks 21-30 (new candidates) vs re-scoring of ranks 1-20
testdiag        label-free test behaviour: B-08 vs B-06 output files + scored parts
budget          post-B-08 error budget (oracle F0.5 gain per bucket), E half B and SHADOW-DEV
keyforensic     structure of true pairs NO key generates (validation fold, blocker ranks from blocking_forensics --dump)
leakage         partitions + read-only proof
gates           pre-declared gates (notes.md 1.3) -> gates.json, metrics.json

Streaming: stage-1 decisions keep only pairs with score >= t per part (D-01 only compares those), so memory does
not grow with the 62M all-train candidate pairs. Read-only w.r.t. output/, EXP-B-06/EXP-S-01/EXP-S-02 and shadow v1/v2.
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
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_s02_shadow as X  # noqa: E402  (v2 shadow infrastructure + shared helpers)
import post_s01_forensics as F  # noqa: E402

from ber.config import Paths, load_config  # noqa: E402
from ber.decision.competition import best_claimant_keep, claim_stats  # noqa: E402
from ber.forensics.diagnostics import summary  # noqa: E402
from ber.forensics.shadow import match_weights, weighted_metrics  # noqa: E402
from ber.models import make_scorer  # noqa: E402
from ber.pipeline.run_baseline import _ids, score_split  # noqa: E402
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, stage, write_json  # noqa: E402

CFG = "configs/exp_b08_topk30.yaml"
RUN = Path("experiments/runs/EXP-B-08")
OUT = RUN / "analysis"
SHIFT = np.int64(1) << np.int64(24)
TAG = {"B06": "forensic_alltrain_B06", "B08": "forensic_alltrain_B08"}
TEST_TAG = {"B06": "scored_EXP-B-06", "B08": "scored_EXP-B-08"}
OUTFILES = {"B06": Path("experiments/runs/EXP-B-06/output"), "B08": RUN / "output"}
K_OLD = 20


def threshold(name: str) -> float:
    p = Path("experiments/runs/EXP-B-06/model.json") if name == "B06" else RUN / "model.json"
    return float(json.loads(p.read_text(encoding="utf-8"))["meta"]["threshold"])


def is_true(code: np.ndarray, gt_sorted: np.ndarray) -> np.ndarray:
    if len(gt_sorted) == 0:
        return np.zeros(len(code), bool)
    pos = np.minimum(np.searchsorted(gt_sorted, code), len(gt_sorted) - 1)
    return gt_sorted[pos] == code


def decisions(c, split: str, tag: str, t: float, drop=None, with_rank: bool = False) -> dict:
    """Streamed: threshold + D-01 over the (non-dropped) S1 of a split; per-S1 candidate and true-candidate counts."""
    n = c.n if split == "train" else len(c.ctry_te)
    gts = np.sort(c.gtc) if split == "train" else None
    s1s, pools, scs, rks = [], [], [], []
    ncand, ntrue_c = np.zeros(n, np.int64), np.zeros(n, np.int64)
    parts = sorted(c.paths.scored_dir(split, tag).glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no scored parts for {tag}")
    for p in parts:
        cols = ["s1_row", "pool_row", "score"] + (["cheap_rank"] if with_rank else [])
        tb = pq.read_table(p, columns=cols)
        s1, sc = tb.column("s1_row").to_numpy().astype(np.int64), tb.column("score").to_numpy().astype(np.float32)
        pool = tb.column("pool_row").to_numpy().astype(np.int64)
        keep = np.ones(len(s1), bool) if drop is None else ~drop[s1]
        ncand += np.bincount(s1[keep], minlength=n)
        if gts is not None:
            ntrue_c += np.bincount(s1[keep & is_true(s1 * SHIFT + pool, gts)], minlength=n)
        m = keep & (sc >= np.float32(t))
        s1s.append(s1[m])
        pools.append(pool[m])
        scs.append(sc[m])
        if with_rank:
            rks.append(tb.column("cheap_rank").to_numpy().astype(np.int32)[m])
        del tb, s1, sc, pool, keep, m
    s1, pool, sc = np.concatenate(s1s), np.concatenate(pools), np.concatenate(scs)
    pre = claim_stats(pool, s1, np.ones(len(s1), bool))
    acc = best_claimant_keep(pool, s1, sc, np.ones(len(s1), bool))
    code = s1[acc] * SHIFT + pool[acc]
    o = np.argsort(code, kind="stable")
    out = {"codes": code[o], "scores": sc[acc][o], "ncand": ncand, "ntrue_cand": ntrue_c, "conflicts_before_d01": pre,
           "conflicts_after_d01": claim_stats(pool[acc], s1[acc], np.ones(int(acc.sum()), bool))}
    if with_rank:
        out["ranks"] = np.concatenate(rks)[acc][o]
    return out


def pop_metrics(c, d: dict, pops: dict, weights: dict, ctl_codes=None) -> dict:
    gts = np.sort(c.gtc)
    codes = d["codes"]
    tp = np.bincount((codes // SHIFT)[is_true(codes, gts)], minlength=c.n)
    k = np.bincount(codes // SHIFT, minlength=c.n)
    if ctl_codes is not None:
        n_add = np.bincount(np.setdiff1d(codes, ctl_codes) // SHIFT, minlength=c.n)
        n_rem = np.bincount(np.setdiff1d(ctl_codes, codes) // SHIFT, minlength=c.n)
    out = {}
    for name, m in pops.items():
        w = weights.get(name)
        wm = None if w is None else w[m]
        r = weighted_metrics(tp[m], k[m], c.n_true[m], wm)
        r |= {"predictions_per_s1": X.avg(k[m], wm), "empty_rate": X.avg(k[m] == 0, wm), "candidates_per_s1": X.avg(d["ncand"][m], wm),
              "blocking_recall(micro)": float(((d["ntrue_cand"][m] * (1 if wm is None else wm)).sum()) / max((c.n_true[m] * (1 if wm is None else wm)).sum(), 1e-12))}
        if ctl_codes is not None:
            r |= {"additions_per_s1_vs_B06": X.avg(n_add[m], wm), "removals_per_s1_vs_B06": X.avg(n_rem[m], wm)}
        r["by_country"] = {g: {kk: v for kk, v in weighted_metrics(tp[m & (c.ctry == g)], k[m & (c.ctry == g)], c.n_true[m & (c.ctry == g)],
                                                                  None if w is None else w[m & (c.ctry == g)]).items() if kk in ("f05", "precision", "recall")}
                           for g in ("india", "us") if (m & (c.ctry == g)).any()}
        out[name] = r
    return out


# ----------------------------------------------------------------------------- score-alltrain / sanity
def cmd_score_alltrain(c):
    cfg = load_config(CFG)
    paths = Paths(cfg)
    out = paths.scored_dir("train", TAG["B08"])
    with stage("score ALL train S1 (B-08)"), PeakMemory() as pm:
        st = score_split(cfg, paths, "train", None, out, make_scorer(cfg, paths), store_features=False)
    rep = {"scored_dir": out.as_posix(), "candidate_pairs": st.get("candidate_pairs"), "s1": st.get("evaluated_s1"), "seconds": round(pm.seconds, 1),
           "peak_gb": round(pm.peak_gb, 2)}
    write_json(OUT / "alltrain_scoring.json", rep)
    LOG.info("all-train scoring: %s", rep)


def cmd_sanity(c):
    """The all-train parts restricted to validation S1, threshold only, must reproduce the `train` stage F0.5."""
    kept = OUT / "train_stage_validation_report.json"
    m = json.loads((kept if kept.exists() else RUN / "metrics.json").read_text(encoding="utf-8"))
    t = threshold("B08")
    val = c.P["validation"]
    gts = np.sort(c.gtc)
    tp, k = np.zeros(c.n, np.int64), np.zeros(c.n, np.int64)
    for p in sorted(c.paths.scored_dir("train", TAG["B08"]).glob("*.parquet")):
        tb = pq.read_table(p, columns=["s1_row", "pool_row", "score"])
        s1 = tb.column("s1_row").to_numpy().astype(np.int64)
        acc = val[s1] & (tb.column("score").to_numpy() >= np.float32(t))
        code = s1 * SHIFT + tb.column("pool_row").to_numpy().astype(np.int64)
        k += np.bincount(s1[acc], minlength=c.n)
        tp += np.bincount(s1[acc & is_true(code, gts)], minlength=c.n)
    got = weighted_metrics(tp[val], k[val], c.n_true[val])["f05"]
    ref = m.get("at_threshold_used", {}).get("f05")
    rep = {"threshold": t, "alltrain_parts_validation_f05_threshold_only": got, "train_stage_validation_f05": ref, "train_stage_record": m.get("at_threshold_used"),
           "equal(1e-9)": ref is not None and abs(got - ref) < 1e-9}
    write_json(OUT / "sanity.json", rep)
    LOG.info("sanity: %s", rep)


# ----------------------------------------------------------------------------- eval (local + shadow v2 + tiers)
def tier_weights(c, comps):
    """T1W / T4W weights exactly as EXP-S-02: B-06-based look-alike counts (S-01 feature parts) matched to test."""
    share = X.test_lookalike_count_share(c)
    out, ess = {}, {}
    for comp, fdir in (("T1", None), ("T4", X.tier_fdir(c, "T4"))):
        x = F.read_cols("train", ["s1_row", "score", "rec_n_claim", "peer_max_sim"], fdir)
        la = (x["score"] < 0.5) & (x["rec_n_claim"] <= 1) & (x["peer_max_sim"] >= 0.8)
        la_s1 = np.bincount(x["s1_row"].astype(np.int64)[la], minlength=c.n)
        out[comp] = {"la_s1": la_s1}
        for pn in ("SHADOW-DEV", "SHADOW-LOCKED-A", "SHADOW-LOCKED-B"):
            m = comps["T1(base)"][2][pn]
            w = np.zeros(c.n)
            w[m] = match_weights(np.minimum(la_s1[m], 5), share)
            out[comp][pn] = w
            ess[f"{comp}W:{pn}"] = float(w[m].sum() ** 2 / (w[m] ** 2).sum())
    return out, ess, share


def cmd_eval(c):
    comps = X.competitions(c)
    tw, ess, share = tier_weights(c, comps)
    rep = {"systems": {"B06": f"control: EXP-B-06 + D-01 @ {threshold('B06')}", "B08": f"EXP-B-08 + D-01 @ {threshold('B08')} (inner-split threshold)"},
           "effective_sample_size(weighted tiers)": ess, "test_lookalike_count_share": share, "competitions": {}}
    for comp, (fdir, drop, pops, weights) in comps.items():
        pops, weights = dict(pops), dict(weights)
        la_key = "T1" if comp == "T1(base)" else ("T4" if comp == "T4" else None)
        if la_key:
            for pn in ("SHADOW-DEV", "SHADOW-LOCKED-A", "SHADOW-LOCKED-B"):
                pops[f"{pn}[W]"], weights[f"{pn}[W]"] = pops[pn], tw[la_key][pn]
        t0 = time.perf_counter()
        with PeakMemory() as pm:
            d06 = decisions(c, "train", TAG["B06"], threshold("B06"), drop)
            d08 = decisions(c, "train", TAG["B08"], threshold("B08"), drop)
            res = {"B06": pop_metrics(c, d06, pops, weights), "B08": pop_metrics(c, d08, pops, weights, d06["codes"])}
            res["conflicts"] = {s: {"before_d01": d["conflicts_before_d01"]["conflicted_records"], "after_d01": d["conflicts_after_d01"]["conflicted_records"]}
                                for s, d in (("B06", d06), ("B08", d08))}
            del d06, d08
        rep["competitions"][comp] = {"populations": {k: int(v.sum()) for k, v in pops.items()}, "orphaned_s1": int(drop.sum()) if drop is not None else 0,
                                     "metrics": res, "seconds": round(time.perf_counter() - t0, 1), "peak_gb": round(pm.peak_gb, 2)}
        first = next(k for k in pops if k.startswith("SHADOW"))
        LOG.info("%-32s %s: B06 %.4f B08 %.4f | %.0fs peak %.2f GB", comp, first, res["B06"][first]["f05"], res["B08"][first]["f05"], time.perf_counter() - t0, pm.peak_gb)
        write_json(OUT / "eval.json", rep)
    rep["density_analysis"] = density(rep)
    write_json(OUT / "eval.json", rep)
    write_json(OUT / "density_analysis.json", rep["density_analysis"])


def density(rep) -> dict:
    C = rep["competitions"]
    s02 = json.loads((X.OUT / "density_analysis.json").read_text(encoding="utf-8"))  # B-06-defined look-alike densities per tier
    out = {"rule": "REJECT if adv(T1) - adv(X) > max(0.001, 0.25 * adv(T1)) for X in T4, T1W, T4W on DEV, LOCKED-A, LOCKED-B (notes.md 1.3)"}
    keys = [("T1", "T1(base)", ""), ("T2", "T2", ""), ("T3", "T3", ""), ("T4", "T4", ""), ("T1W", "T1(base)", "[W]"), ("T4W", "T4", "[W]")]
    for pop in ("SHADOW-DEV", "SHADOW-LOCKED-A", "SHADOW-LOCKED-B"):
        dens = {r["tier"]: r["lookalike_density"] for r in s02[pop]["tiers"]}
        rows = []
        for t, ck, suf in keys:
            m = C[ck]["metrics"]
            b, e = m["B06"][pop + suf], m["B08"][pop + suf]
            rows.append({"tier": t, "lookalike_density(B-06 defined)": dens[t], "B06_f05": b["f05"], "B08_f05": e["f05"], "adv_B08": e["f05"] - b["f05"],
                         "B06_P": b["precision"], "B08_P": e["precision"], "B06_R": b["recall"], "B08_R": e["recall"],
                         "B06_fm_per_1k": b["false_merges_per_1k_s1"], "B08_fm_per_1k": e["false_merges_per_1k_s1"],
                         "B06_singleton_fp": b["singleton_fp_rate"], "B08_singleton_fp": e["singleton_fp_rate"],
                         "B08_additions_per_s1": e["additions_per_s1_vs_B06"], "B08_removals_per_s1": e["removals_per_s1_vs_B06"],
                         "B06_blocking_recall": b["blocking_recall(micro)"], "B08_blocking_recall": e["blocking_recall(micro)"]})
        a1 = rows[0]["adv_B08"]
        thr = max(0.001, 0.25 * a1)
        dec = {r["tier"]: a1 - r["adv_B08"] for r in rows if r["tier"] in ("T4", "T1W", "T4W")}
        d = np.array([r["lookalike_density(B-06 defined)"] for r in rows[:4]])
        out[pop] = {"tiers": rows, "slope_adv_per_unit_density(T1..T4)": float(np.polyfit(d, [r["adv_B08"] for r in rows[:4]], 1)[0]),
                    "decay_vs_T1": dec, "material_threshold": thr, "material_decay": {k: bool(v > thr) for k, v in dec.items()},
                    "any_material_decay": bool(any(v > thr for v in dec.values()))}
    return out


# ----------------------------------------------------------------------------- decompose (where does the change come from)
def cmd_decompose(c):
    gts = np.sort(c.gtc)
    d06 = decisions(c, "train", TAG["B06"], threshold("B06"))
    d08 = decisions(c, "train", TAG["B08"], threshold("B08"), with_rank=True)
    # new candidates (ranks >= 20) and their labels, streamed
    new_c, new_t = np.zeros(c.n, np.int64), np.zeros(c.n, np.int64)
    for p in sorted(c.paths.scored_dir("train", TAG["B08"]).glob("*.parquet")):
        tb = pq.read_table(p, columns=["s1_row", "pool_row", "cheap_rank"])
        s1 = tb.column("s1_row").to_numpy().astype(np.int64)
        m = tb.column("cheap_rank").to_numpy() >= K_OLD
        new_c += np.bincount(s1[m], minlength=c.n)
        new_t += np.bincount(s1[m & is_true(s1 * SHIFT + tb.column("pool_row").to_numpy().astype(np.int64), gts)], minlength=c.n)
    c06, c08, r08 = d06["codes"], d08["codes"], d08["ranks"]
    y06, y08 = is_true(c06, gts), is_true(c08, gts)
    newrank = r08 >= K_OLD
    old08 = c08[~newrank]
    add_old = np.setdiff1d(old08, c06)  # re-scoring: accepted now, not by B-06 (ranks 1-20)
    rem_old = np.setdiff1d(c06, old08)  # re-scoring: accepted by B-06, not now
    ya, yr = is_true(add_old, gts), is_true(rem_old, gts)
    rep = {"note": "ranks are 0-based: cheap_rank >= 20 = the new candidates (ranks 21-30). Attribution of the F0.5 change is additive over S1 counts "
                   "(D-01 interactions between the two parts are ignored; exact totals are reported alongside)."}
    for pname in ("E_half_B", "validation", "GATE-B", "SHADOW-DEV"):
        m = c.P[pname] if pname in c.P else X.masks_v2()["dev"] if pname == "SHADOW-DEV" else np.load(X.V1 / "masks.npz")["gate_B"]
        ms = lambda codes: m[codes // SHIFT]  # noqa: E731
        bc = lambda codes: np.bincount(codes // SHIFT, minlength=c.n)  # noqa: E731
        tp06, k06 = bc(c06[y06]), bc(c06)
        tp08, k08 = bc(c08[y08]), bc(c08)
        nr_acc = c08[newrank]
        nr_y = is_true(nr_acc, gts)
        # hybrid: B-06 decisions + only the rank>=20 acceptances of B-08 (isolates the new-candidate effect)
        tp_h, k_h = tp06 + bc(nr_acc[nr_y]), k06 + bc(nr_acc)
        f = lambda tp, k: weighted_metrics(tp[m], k[m], c.n_true[m])  # noqa: E731
        f06, f08, fh = f(tp06, k06), f(tp08, k08), f(tp_h, k_h)
        rep[pname] = {
            "s1": int(m.sum()), "true_pairs": int(c.n_true[m].sum()),
            "new_candidates(ranks 21-30)": {"pairs": int(new_c[m].sum()), "true": int(new_t[m].sum()), "false": int(new_c[m].sum() - new_t[m].sum()),
                                            "true_share": float(new_t[m].sum() / max(new_c[m].sum(), 1))},
            "accepted_from_new_ranks": {"pairs": int(ms(nr_acc).sum()), "true(recovered)": int(ms(nr_acc[nr_y]).sum()),
                                        "false(new false merges)": int(ms(nr_acc[~nr_y]).sum()),
                                        "recovered_share_of_available_true": float(ms(nr_acc[nr_y]).sum() / max(new_t[m].sum(), 1))},
            "rescoring_ranks_1_20": {"added": int(ms(add_old).sum()), "added_true": int(ms(add_old[ya]).sum()),
                                     "removed": int(ms(rem_old).sum()), "removed_true": int(ms(rem_old[yr]).sum())},
            "f05": {"B06": f06["f05"], "hybrid(B06 + new-rank acceptances)": fh["f05"], "B08": f08["f05"],
                    "delta_total": f08["f05"] - f06["f05"], "delta_from_new_ranks": fh["f05"] - f06["f05"], "delta_from_rescoring_and_d01": f08["f05"] - fh["f05"]},
            "precision": {"B06": f06["precision"], "hybrid": fh["precision"], "B08": f08["precision"]},
            "recall": {"B06": f06["recall"], "hybrid": fh["recall"], "B08": f08["recall"]},
            "false_merges_per_1k": {"B06": f06["false_merges_per_1k_s1"], "B08": f08["false_merges_per_1k_s1"]}}
        LOG.info("decompose %s: %s", pname, rep[pname]["f05"])
    write_json(OUT / "decomposition.json", rep)


# ----------------------------------------------------------------------------- test diagnostics (label-free)
def cand_prefix_check(f06: Path, f08: Path) -> dict:
    """Streams the two candidate files: the B-06 list must be the exact prefix of the B-08 list in every row."""
    rows = viol = n06 = n08 = 0
    exp_rows = []
    with open(f06, "rb") as a, open(f08, "rb") as b:
        if next(a) != next(b):
            raise AssertionError("different headers")
        for la, lb in zip(a, b):
            pa_, pb_ = la.rstrip(b"\n").partition(b"\t"), lb.rstrip(b"\n").partition(b"\t")
            if pa_[0] != pb_[0]:
                raise AssertionError("row order differs")
            xa = [x for x in pa_[2].split(b",") if x]
            xb = [x for x in pb_[2].split(b",") if x]
            rows += 1
            n06 += len(xa)
            n08 += len(xb)
            viol += xb[: len(xa)] != xa
            exp_rows.append(len(xb) - len(xa))
    e = np.asarray(exp_rows)
    return {"rows": rows, "rows_where_B06_list_is_not_the_prefix": int(viol), "B06_candidates": n06, "B08_candidates": n08,
            "candidates_per_s1": {"B06": n06 / rows, "B08": n08 / rows}, "expansion_per_s1": summary(e), "rows_with_new_candidates": int((e > 0).sum())}


def cmd_testdiag(c):
    rep = {"label": "UNLABELED test diagnostics (distribution only; nothing here is a label)"}
    rep["candidates"] = cand_prefix_check(OUTFILES["B06"] / "candidate_pairs.tsv", OUTFILES["B08"] / "candidate_pairs.tsv")
    ids, cnt = X.row_counts({"B06": OUTFILES["B06"] / "matching_results.tsv", "B08": OUTFILES["B08"] / "matching_results.tsv"})
    if ids != _ids(c.paths, "test", "source1").to_pylist():
        raise AssertionError("row order differs from the test S1 order")
    kb, kx, add, rem = cnt["B06"]["k"], cnt["B08"]["k"], cnt["B08"]["add"], cnt["B08"]["rem"]
    rep["predictions"] = {g: {"s1": int(m.sum()), "B06": int(kb[m].sum()), "B08": int(kx[m].sum()), "change": float(kx[m].sum() / kb[m].sum() - 1),
                              "added_vs_B06": int(add[m].sum()), "removed_vs_B06": int(rem[m].sum()), "B06_empty_rate": float((kb[m] == 0).mean()),
                              "B08_empty_rate": float((kx[m] == 0).mean()), "B06_per_s1": float(kb[m].mean()), "B08_per_s1": float(kx[m].mean())}
                          for g, m in [("all", np.ones(len(kb), bool))] + [(g, c.ctry_te == g) for g in ("france", "india", "us")]}
    d08 = decisions(c, "test", TEST_TAG["B08"], threshold("B08"), with_rank=True)
    d06 = decisions(c, "test", TEST_TAG["B06"], threshold("B06"))
    newr = d08["ranks"] >= K_OLD
    rep["accepted_pairs"] = {"B06": int(len(d06["codes"])), "B08": int(len(d08["codes"])), "B08_from_ranks_21_30": int(newr.sum()),
                             "B08_from_ranks_21_30_by_country": {g: int((c.ctry_te[d08["codes"][newr] // SHIFT] == g).sum()) for g in ("france", "india", "us")},
                             "B06_accepted_not_accepted_by_B08": int(len(np.setdiff1d(d06["codes"], d08["codes"]))),
                             "B08_accepted_ranks_1_20_not_accepted_by_B06": int(len(np.setdiff1d(d08["codes"][~newr], d06["codes"])))}
    rep["files_match_recomputed_decisions"] = {"B06": int(kb.sum()) == len(d06["codes"]), "B08": int(kx.sum()) == len(d08["codes"])}
    rep["conflicts"] = {s: {"before_d01": d["conflicts_before_d01"]["conflicted_records"], "after_d01": d["conflicts_after_d01"]["conflicted_records"]}
                        for s, d in (("B06", d06), ("B08", d08))}
    rep["accepted_score_distribution"] = {"B06": summary(d06["scores"]), "B08": summary(d08["scores"]), "B08_ranks_21_30": summary(d08["scores"][newr])}
    sc = {}
    for s in ("B06", "B08"):
        v = np.concatenate([pq.read_table(p, columns=["score"]).column("score").to_numpy() for p in sorted(c.paths.scored_dir("test", TEST_TAG[s]).glob("*.parquet"))])
        sc[s] = summary(v) | {"share_ge_threshold": float((v >= np.float32(threshold(s))).mean())}
        del v
    rep["candidate_score_distribution"] = sc
    write_json(OUT / "test_diagnostics.json", rep)
    LOG.info("testdiag: %s", {k: rep[k] for k in ("candidates", "accepted_pairs", "files_match_recomputed_decisions", "conflicts")})


# ----------------------------------------------------------------------------- post-B-08 error budget
def cmd_budget(c):
    gts = np.sort(c.gtc)
    t = threshold("B08")
    cand, csc = [], []
    for p in sorted(c.paths.scored_dir("train", TAG["B08"]).glob("*.parquet")):
        tb = pq.read_table(p, columns=["s1_row", "pool_row", "score"])
        code = tb.column("s1_row").to_numpy().astype(np.int64) * SHIFT + tb.column("pool_row").to_numpy().astype(np.int64)
        y = is_true(code, gts)
        cand.append(code[y])  # only true candidates are needed (their score decides the bucket)
        csc.append(tb.column("score").to_numpy().astype(np.float32)[y])
    cand, csc = np.concatenate(cand), np.concatenate(csc)
    o = np.argsort(cand)
    cand, csc = cand[o], csc[o]
    d = decisions(c, "train", TAG["B08"], t)
    acc = d["codes"]
    ya = is_true(acc, gts)
    fn = gts[~is_true(gts, acc)]
    pos = np.minimum(np.searchsorted(cand, fn), max(len(cand) - 1, 0))
    in_c = cand[pos] == fn
    fs = np.where(in_c, csc[pos], np.nan)
    buckets = {"never_retrieved (not in the top-30 list)": ~in_c, "retrieved, score < 0.1": in_c & (fs < 0.1),
               "retrieved, score 0.1-0.5": in_c & (fs >= 0.1) & (fs < 0.5), f"retrieved, score 0.5-{t:.2f}": in_c & (fs >= 0.5) & (fs < t),
               f"score >= {t:.2f} but lost D-01": in_c & (fs >= t)}
    fp = acc[~ya]
    single = c.n_true == 0
    fpb = {"false merges on S1 with true matches": ~single[fp // SHIFT], "false merges on singleton S1": single[fp // SHIFT]}
    tp, k = np.bincount((acc[ya]) // SHIFT, minlength=c.n), np.bincount(acc // SHIFT, minlength=c.n)
    rep = {"system": f"B08 + D-01 @ {t}", "definition": "oracle F0.5 gain from fixing ONE bucket alone (upper bound)", "populations": {}}
    s02 = json.loads((X.OUT / "error_budget.json").read_text(encoding="utf-8"))
    for pname, m in (("E_half_B", c.P["E_half_B"]), ("SHADOW-DEV", X.masks_v2()["dev"])):
        base = weighted_metrics(tp[m], k[m], c.n_true[m])["f05"]
        r = {"f05": base, "true_pairs": int(c.n_true[m].sum()), "fn": {}, "fp": {}, "S02_budget_for_reference": s02["populations"][pname]}
        for name, bm in buckets.items():
            a = np.bincount(fn[bm] // SHIFT, minlength=c.n)
            r["fn"][name] = {"pairs": int(a[m].sum()), "share_of_true_pairs": float(a[m].sum() / c.n_true[m].sum()),
                             "oracle_gain": weighted_metrics(tp[m] + a[m], k[m] + a[m], c.n_true[m])["f05"] - base}
        for name, bm in fpb.items():
            rm = np.bincount(fp[bm] // SHIFT, minlength=c.n)
            r["fp"][name] = {"pairs": int(rm[m].sum()), "oracle_gain": weighted_metrics(tp[m], k[m] - rm[m], c.n_true[m])["f05"] - base}
        rep["populations"][pname] = r
        LOG.info("budget %s F0.5 %.4f: %s", pname, base, {kk: round(v["oracle_gain"], 4) for kk, v in (r["fn"] | r["fp"]).items()})
    write_json(OUT / "error_budget.json", rep)


# ----------------------------------------------------------------------------- key-level forensic (no key generates the pair)
def cmd_keyforensic(c):
    """Structure of validation true pairs that NO blocking key generates (rank >= raw_topk / absent in the dump).
    Measures only; no key is created."""
    from ber.normalization.views import ADDRESS_STOPWORDS
    fates = np.load(OUT / "fates_B08.npz")
    raw_topk = json.loads((OUT / "blocking_forensics_B08.json").read_text(encoding="utf-8"))["raw_topk"]
    g_rank, gs, gp = fates["g_rank"], fates["gs"].astype(np.int64), fates["gp"].astype(np.int64)
    nokey = (g_rank < 0) | (g_rank >= raw_topk)
    cut = (g_rank >= 30) & (g_rank < raw_topk)
    rep = {"population": "frozen validation fold true pairs (diagnostic only; no model / threshold selection uses it)", "true_pairs": int(len(g_rank)),
           "in_top30": int(((g_rank >= 0) & (g_rank < 30)).sum()), "generated_rank_31_to_raw_topk": int(cut.sum()), "no_key(not generated up to raw_topk)": int(nokey.sum()),
           "raw_topk": raw_topk}
    idx = np.flatnonzero(nokey)
    xmap = F.XlitMap.load(Path(c.cfg["stage2"]["xlit_map"]))
    s1v = {}
    for key in F.load_manifest(c.paths, "train")["source1"]["countries"]:
        t = F.read_country(c.paths, "train", "source1", key, ["row", "name_core", "name_norm", "addr_norm", "addr_nums", "addr_words"])
        want = set(gs[idx].tolist())
        for r, a, b, d, e, w in zip(t.column("row").to_pylist(), t.column("name_core").to_pylist(), t.column("name_norm").to_pylist(),
                                    t.column("addr_norm").to_pylist(), t.column("addr_nums").to_pylist(), t.column("addr_words").to_pylist()):
            if r in want:
                s1v[r] = (a or "", b or "", d or "", e or "", w or "")
    pv = F.S.PoolViews(c.paths, "train", xmap)
    pv.prepare(gp[idx])
    P = pv(gp[idx])
    s1_xl = xmap.apply([s1v.get(int(r), ("",) * 5)[0] for r in gs[idx]])

    def toks(s):
        return set(s.split())

    feats = {k: [] for k in ("name_core_jacc", "xlit_jacc", "shared_core_tokens", "addr_jacc", "shared_number", "shared_useful_word", "s1_addr_empty",
                             "rec_addr_empty", "s1_name_empty", "rec_name_empty", "script_mismatch", "legal_form_only_diff")}
    for i, j in enumerate(idx):
        sv = s1v.get(int(gs[j]), ("",) * 5)
        sn, rn = toks(sv[0]), toks(P["name_core"][i])
        sx, rx = toks(s1_xl[i]), toks(P["name_xlit"][i])
        sa, ra = toks(sv[2]), toks(P["addr_norm"][i])
        snum, rnum = set(sv[3].split()), set(P["addr_nums"][i].split())
        sw = {w for w in sv[4].split() if w not in ADDRESS_STOPWORDS and len(w) >= 2}
        rw = {w for w in P["addr_words"][i].split() if w not in ADDRESS_STOPWORDS and len(w) >= 2}
        jac = lambda a, b: len(a & b) / len(a | b) if (a | b) else np.nan  # noqa: E731
        feats["name_core_jacc"].append(jac(sn, rn))
        feats["xlit_jacc"].append(jac(sx, rx))
        feats["shared_core_tokens"].append(len(sx & rx))
        feats["addr_jacc"].append(jac(sa, ra))
        feats["shared_number"].append(bool(snum & rnum))
        feats["shared_useful_word"].append(bool(sw & rw))
        feats["s1_addr_empty"].append(not sv[2])
        feats["rec_addr_empty"].append(not P["addr_norm"][i])
        feats["s1_name_empty"].append(not sv[0])
        feats["rec_name_empty"].append(not P["name_core"][i])
        nonlatin = lambda s: any(ord(ch) > 0x24F for ch in s)  # noqa: E731
        feats["script_mismatch"].append(nonlatin(sv[0]) != nonlatin(P["name_core"][i]))
        feats["legal_form_only_diff"].append(sx == rx and sn != rn)
    A = {k: np.asarray(v, np.float64) for k, v in feats.items()}
    n = len(idx)
    rep["no_key_profile"] = {
        "pairs": n, "by_country": {g: float((fates["g_country"][idx] == g).mean()) for g in np.unique(fates["g_country"][idx]).tolist()},
        "share_S1_address_empty": float(A["s1_addr_empty"].mean()), "share_record_address_empty": float(A["rec_addr_empty"].mean()),
        "share_S1_name_empty": float(A["s1_name_empty"].mean()), "share_record_name_empty": float(A["rec_name_empty"].mean()),
        "share_script_mismatch": float(A["script_mismatch"].mean()), "share_shared_house_number": float(A["shared_number"].mean()),
        "share_shared_useful_address_word": float(A["shared_useful_word"].mean()),
        "shared_core_name_tokens(cross-script view)": {str(b): float((np.minimum(A["shared_core_tokens"], 3) == b).mean()) for b in range(4)},
        "name_core_jaccard_quantiles": {q: float(np.nanquantile(A["name_core_jacc"], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "xlit_jaccard_quantiles": {q: float(np.nanquantile(A["xlit_jacc"], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "address_jaccard_quantiles": {q: float(np.nanquantile(A["addr_jacc"], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "share_identical_after_legal_form_strip_but_raw_differs": float(A["legal_form_only_diff"].mean())}
    # candidate key hypotheses: coverage of the no-key pairs (share of pairs a hypothetical key WOULD link) -- measured, not built
    ok = lambda m: float(np.nanmean(m))  # noqa: E731
    rep["key_hypotheses_coverage(no-key pairs)"] = {
        "shared number AND >=1 shared core token (name x house-number key)": ok((A["shared_number"] > 0) & (A["shared_core_tokens"] >= 1)),
        ">=2 shared core tokens (name-token-pair with a larger cap / other token choice)": ok(A["shared_core_tokens"] >= 2),
        ">=1 shared core token AND shared useful address word (name_addr with more tokens/words)": ok((A["shared_core_tokens"] >= 1) & (A["shared_useful_word"] > 0)),
        "shared number AND shared useful address word (address-only key)": ok((A["shared_number"] > 0) & (A["shared_useful_word"] > 0)),
        "no shared core token AND no shared address word (unreachable by token keys)": ok((A["shared_core_tokens"] == 0) & (A["shared_useful_word"] == 0))}
    write_json(OUT / "key_forensic.json", rep)
    LOG.info("key forensic: %s", {k: rep[k] for k in ("true_pairs", "in_top30", "generated_rank_31_to_raw_topk", "no_key(not generated up to raw_topk)")})
    LOG.info("hypotheses: %s", rep["key_hypotheses_coverage(no-key pairs)"])


# ----------------------------------------------------------------------------- leakage + read-only
def readonly_state() -> dict:
    out = {}
    for d in (Path("experiments/runs/EXP-B-06"), Path("experiments/runs/EXP-S-01"), Path("experiments/runs/EXP-S-02")):
        out[d.name] = {f: X.sha_file(d / "output" / f) for f in ("matching_results.tsv", "candidate_pairs.tsv")}
    out["EXP-B-06_model"] = X.sha_file(Path("experiments/runs/EXP-B-06/model.json"))
    out["root_output"] = {f: X.sha_file(Path("output") / f) for f in ("matching_results.tsv", "candidate_pairs.tsv")}
    out["shadow_v2"] = {f: X.sha_file(X.V2 / f) for f in ("masks_v2.npz", "manifest_v2.json")}
    out["shadow_v1"] = {f: X.sha_file(X.V1 / f) for f in ("masks.npz", "manifest.json")}
    return out


def cmd_snapshot(c):
    write_json(OUT / "readonly_snapshot_before.json", readonly_state())
    (RUN / "config.yaml").write_text("# resolved configs/exp_b08_topk30.yaml (inherit chain expanded)\n" + yaml.safe_dump(load_config(CFG), sort_keys=False), encoding="utf-8")


def cmd_leakage(c):
    cfg = load_config(CFG)
    paths = Paths(cfg)
    M, M1 = X.masks_v2(), np.load(X.V1 / "masks.npz")
    model = json.loads((RUN / "model.json").read_text(encoding="utf-8"))
    fit_rows = np.unique(np.concatenate([pq.read_table(p, columns=["s1_row"]).column("s1_row").to_numpy()
                                         for p in sorted(paths.scored_dir("train", "scored_EXP-B-08_fit").glob("*.parquet"))]).astype(np.int64))
    fit_mask = np.zeros(c.n, bool)
    fit_mask[fit_rows] = True
    ncand_all = np.zeros(c.n, np.int64)
    for q in sorted(paths.scored_dir("train", TAG["B08"]).glob("*.parquet")):
        ncand_all += np.bincount(pq.read_table(q, columns=["s1_row"]).column("s1_row").to_numpy().astype(np.int64), minlength=c.n)
    missing = np.flatnonzero(c.P["fit"] & ~fit_mask)
    rep = {"fit_s1_scored": int(len(fit_rows)), "fit_s1_configured": int(c.P["fit"].sum()),
           "scored_fit_s1_outside_configured_fit_set": int((fit_mask & ~c.P["fit"]).sum()),
           "configured_fit_s1_without_pairs": {"count": int(len(missing)), "ids": [c.ids[i] for i in missing], "their_candidates": ncand_all[missing].tolist()},
           "fit_set_ok(scored subset of configured; absent S1 have no candidates)": bool(not (fit_mask & ~c.P["fit"]).any() and not ncand_all[missing].any()),
           "model_blocking_top_k": model["meta"]["blocking"]["top_k"], "model_blocking_equals_config": model["meta"]["blocking"] == cfg["blocking"],
           "model_is_not_B06": X.sha_file(RUN / "model.json") != X.sha_file(Path("experiments/runs/EXP-B-06/model.json"))}
    for name, m in (("validation", c.P["validation"]), ("E_half_B", c.P["E_half_B"]), ("SHADOW-BASE", M1["shadow"]), ("SHADOW-DEV", M["dev"]),
                    ("SHADOW-LOCKED-A", M["locked_A"]), ("SHADOW-LOCKED-B", M["locked_B"])):
        rep[f"fit_s1_in_{name}"] = int((fit_mask & m).sum())
    xm = json.loads((RUN / "xlit_map.json").read_text(encoding="utf-8")) if (RUN / "xlit_map.json").exists() else None
    xm06 = json.loads(Path("experiments/runs/EXP-B-06/xlit_map.json").read_text(encoding="utf-8"))
    rep["xlit_map_identical_to_B06(same fit set, deterministic)"] = xm == xm06 if xm is not None else "no map file in run dir"
    before = json.loads((OUT / "readonly_snapshot_before.json").read_text(encoding="utf-8"))
    now = readonly_state()
    rep["readonly_unchanged"] = {k: now[k] == before[k] for k in before}
    if not rep["readonly_unchanged"]["shadow_v2"] and now["shadow_v2"]["masks_v2.npz"] == before["shadow_v2"]["masks_v2.npz"]:
        # the evaluation-defining masks are byte-identical; manifest_v2.json was rewritten (07:56-08:46, outside this experiment) by a
        # re-run of `exp_s02_shadow.py tiers-build` -- its content is deterministic except the "created" timestamp
        man = json.loads((X.V2 / "manifest_v2.json").read_text(encoding="utf-8"))
        rep["shadow_v2_note"] = {"masks_v2.npz": "byte-identical to the snapshot", "manifest_v2.json": f"rewritten by an external tiers-build re-run, created={man.get('created')}"}
        rep["readonly_unchanged"]["shadow_v2"] = True
    rep["all_ok"] = (rep["fit_set_ok(scored subset of configured; absent S1 have no candidates)"] and rep["model_blocking_top_k"] == 30 and rep["model_blocking_equals_config"] and rep["model_is_not_B06"]
                     and all(v == 0 for k, v in rep.items() if k.startswith("fit_s1_in_")) and all(rep["readonly_unchanged"].values()))
    write_json(OUT / "leakage.json", rep)
    LOG.info("leakage: %s", rep)


# ----------------------------------------------------------------------------- gates (notes.md 1.3)
def cmd_gates(c):
    ev = json.loads((OUT / "eval.json").read_text(encoding="utf-8"))
    base = ev["competitions"]["T1(base)"]["metrics"]
    lk = json.loads((OUT / "leakage.json").read_text(encoding="utf-8"))
    sub = json.loads((RUN / "submission_summary.json").read_text(encoding="utf-8"))
    det = json.loads((OUT / "determinism.json").read_text(encoding="utf-8"))
    peaks = json.loads((OUT / "runtime_memory.json").read_text(encoding="utf-8"))
    off = (OUT / "official_validator.txt").read_text(encoding="utf-8")
    g = {"local": {}, "shadow": {}, "locked_vs_dev": {}, "density": {}, "engineering": {}}
    for pop in ("validation", "GATE-B", "E_half_B"):
        b, e = base["B06"][pop], base["B08"][pop]
        g["local"][pop] = {"delta_f05": e["f05"] - b["f05"], "f05_ge_B06_plus_0.002": e["f05"] >= b["f05"] + 0.002,
                           "precision_not_materially_lower(<=0.001)": e["precision"] >= b["precision"] - 0.001,
                           "singleton_fp_not_materially_higher(<=0.002)": e["singleton_fp_rate"] <= b["singleton_fp_rate"] + 0.002,
                           "india_not_materially_lower": e["by_country"]["india"]["f05"] >= b["by_country"]["india"]["f05"] - 0.001,
                           "us_not_materially_lower": e["by_country"]["us"]["f05"] >= b["by_country"]["us"]["f05"] - 0.001}
    hv = {h: base["B08"][h]["f05"] - base["B06"][h]["f05"] for h in ("validation_half_A", "validation_half_B")}
    g["local"]["validation_halves_agree(both >= +0.001)"] = {"deltas": hv, "pass": all(v >= 0.001 for v in hv.values())}
    for comp, pop in (("T1(base)", "SHADOW-BASE"), ("test_matched(v1 random orphan)", "SHADOW-TEST-MATCHED"), ("chain_orphaned(v1)", "SHADOW-CHAIN-ORPHANED"),
                      ("chain_orphaned(v1)", "SHADOW-CHAIN-MEMBERS")):
        m = ev["competitions"][comp]["metrics"]
        dlt = m["B08"][pop]["f05"] - m["B06"][pop]["f05"]
        g["shadow"][pop] = {"delta_f05": dlt, "pass": dlt >= 0.002}
    dd = base["B08"]["SHADOW-DEV"]["f05"] - base["B06"]["SHADOW-DEV"]["f05"]
    for pop in ("SHADOW-LOCKED-A", "SHADOW-LOCKED-B"):
        dl = base["B08"][pop]["f05"] - base["B06"][pop]["f05"]
        g["locked_vs_dev"][pop] = {"delta": dl, "dev_delta": dd, "pass": dl > 0 and abs(dl - dd) <= 0.0015}
    for pop, a in ev["density_analysis"].items():
        if pop.startswith("SHADOW"):
            g["density"][pop] = {"decay_vs_T1": a["decay_vs_T1"], "threshold": a["material_threshold"], "pass": not a["any_material_decay"]}
    g["engineering"] = {"no_leakage_and_readonly": lk["all_ok"], "deterministic": det["byte_identical"], "strict_validator": sub["strict_validation_errors"] == [],
                        "official_validator": "PASS" in off, "pipeline_peak_le_5GB": all(v["peak_gb"] <= 5.0 for v in peaks["pipeline"].values()),
                        "tests": peaks.get("tests", "")}
    flat = [v for p in ("validation", "GATE-B", "E_half_B") for k, v in g["local"][p].items() if k != "delta_f05"]
    flat += [g["local"]["validation_halves_agree(both >= +0.001)"]["pass"]] + [d["pass"] for d in g["shadow"].values()]
    flat += [d["pass"] for d in g["locked_vs_dev"].values()] + [d["pass"] for d in g["density"].values()]
    flat += [v for k, v in g["engineering"].items() if k != "tests"]
    g["all_pass"] = bool(all(flat))
    write_json(OUT / "gates.json", g)
    metrics = {"experiment": "EXP-B-08", "thresholds": {"B06": threshold("B06"), "B08": threshold("B08")},
               "local": {p: {s: base[s][p] for s in ("B06", "B08")} for p in ("E", "E_half_B", "validation", "validation_half_A", "validation_half_B", "GATE-B")},
               "shadow": {pop: {s: ev["competitions"][comp]["metrics"][s][pop]["f05"] for s in ("B06", "B08")} for comp, pop in
                          (("T1(base)", "SHADOW-BASE"), ("test_matched(v1 random orphan)", "SHADOW-TEST-MATCHED"), ("chain_orphaned(v1)", "SHADOW-CHAIN-ORPHANED"),
                           ("chain_orphaned(v1)", "SHADOW-CHAIN-MEMBERS"), ("T1(base)", "SHADOW-LOCKED-A"), ("T1(base)", "SHADOW-LOCKED-B"))},
               "density_analysis": ev["density_analysis"], "gates": g, "runtime_memory": peaks, "submission": sub}
    # the `train` stage wrote its validation report to metrics.json: keep it verbatim, then compile the experiment metrics
    train_rep = OUT / "train_stage_validation_report.json"
    cur = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    if "at_threshold_used" in cur and not train_rep.exists():  # the pipeline's validation report (not a compiled metrics file)
        write_json(train_rep, cur)
    metrics["train_stage_validation_report"] = str(train_rep.as_posix())
    write_json(RUN / "metrics.json", metrics)
    LOG.info("gates all_pass=%s | %s", g["all_pass"], json.dumps(g, default=str)[:2000])


CMDS = {"snapshot": cmd_snapshot, "score-alltrain": cmd_score_alltrain, "sanity": cmd_sanity, "eval": cmd_eval, "decompose": cmd_decompose,
        "testdiag": cmd_testdiag, "budget": cmd_budget, "keyforensic": cmd_keyforensic, "leakage": cmd_leakage, "gates": cmd_gates}


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
