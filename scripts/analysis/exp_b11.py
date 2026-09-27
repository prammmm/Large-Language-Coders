"""EXP-B-11 (name_token.pair weight 1.0 -> 2.0 on the B-10 3-token key) blocking-only kill gate + rank-repair analysis.
Deterministic, seed 42. Read-only w.r.t. EXP-B-09 / EXP-B-10 artefacts and root output/.

    python scripts/analysis/exp_b11.py blockdiag

blockdiag   B-10 and B-11 blockers run 1000 deep on the frozen validation fold (B-10 re-run gives cheap scores and a
            determinism check against the stored EXP-B-10 fates). References: B-09 stored top-20 validation lists
            (scored_EXP-B-09) + EXP-B-09/analysis/fates_B09_k1000.npz. Writes blocking_forensics.json,
            rank_repair_analysis.json, displacement_analysis.json, india_analysis.json, us_analysis.json (blocking part).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_b10 as B10  # noqa: E402
import exp_s02_shadow as X  # noqa: E402
import final_audit as FA  # noqa: E402

from ber.blocking.blocker import PASS_ORDER, CountryBlocker  # noqa: E402
from ber.config import Paths, load_config  # noqa: E402
from ber.data.normalize_records import load_manifest, read_country  # noqa: E402
from ber.data.split import load_val_ids  # noqa: E402
from ber.forensics.audit import pair_profile  # noqa: E402
from ber.pipeline.run_baseline import VIEW_COLS, LazyPool, resolve_xmap, with_xlit  # noqa: E402
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, write_json  # noqa: E402

RUN = Path("experiments/runs/EXP-B-11")
OUT = RUN / "analysis"
CFG = {"B10": "configs/exp_b10_name_pair3.yaml", "B11": "configs/exp_b11_name_pair3_w2.yaml"}
SHIFT = B10.SHIFT
NT_BIT = 1 << PASS_ORDER.index("name_token")
BIG = 10**6


def run_blocker(cfg, paths, val_ids, top_k=1000):
    """Validation-fold blocker run: codes, cheap ranks, flags, cheap scores + S1 row -> country."""
    bcfg = copy.deepcopy(cfg["blocking"])
    bcfg["top_k"] = top_k
    xmap = resolve_xmap(cfg, paths)
    man = load_manifest(paths, "train")
    n_s2 = int(man["source2"]["rows"])
    val_arr = pa.array(val_ids, pa.string())
    codes, ranks, flags, scores, s1c = [], [], [], [], {}
    for key in sorted(man["source1"]["countries"]):
        s1 = read_country(paths, "train", "source1", key, VIEW_COLS)
        s1 = s1.filter(pc.is_in(s1.column("entity_id"), value_set=val_arr))
        prow = np.concatenate([read_country(paths, "train", src, key, ["row"]).column("row").to_numpy().astype(np.int64) + (n_s2 if src == "source3" else 0)
                               for src in ("source2", "source3")])
        srow = s1.column("row").to_numpy().astype(np.int64)
        s1c[key] = srow
        b = CountryBlocker(with_xlit(s1, xmap), LazyPool(paths, "train", key, xmap), bcfg)
        for lo, hi in b.chunk_bounds():
            cc = b.candidates(lo, hi)
            codes.append(srow[cc["s1_local"]] * SHIFT + prow[cc["pool_local"]])
            ranks.append(cc["cheap_rank"].astype(np.int32))
            flags.append(cc["pass_flags"].astype(np.uint16))
            scores.append(cc["cheap_score"].astype(np.float32))
        del b, s1
    return np.concatenate(codes), np.concatenate(ranks), np.concatenate(flags), np.concatenate(scores), s1c


def fates(code, rank, flag, score, gcode):
    o = np.argsort(code, kind="stable")
    cs_ = code[o]
    pos = np.minimum(np.searchsorted(cs_, gcode), len(cs_) - 1)
    found = cs_[pos] == gcode
    return (np.where(found, rank[o][pos], BIG), np.where(found, flag[o][pos], 0), np.where(found, score[o][pos], np.nan))


def reduce_run(name, gcode, val_ids, paths):
    cfg = load_config(CFG[name])
    t0 = time.perf_counter()
    with PeakMemory() as pm:
        code, rank, flag, score, s1c = run_blocker(cfg, paths, val_ids)
    secs = time.perf_counter() - t0
    r, f, s = fates(code, rank, flag, score, gcode)
    top20 = np.sort(code[rank < 20])
    n30 = int((rank < 30).sum())
    n_all = len(code)
    # per-S1 rank-20 score (the entry bar) for rank-repair context
    del code, rank, flag, score
    LOG.info("%s blocker: %.0fs peak %.2f GB | top20 %d", name, secs, pm.peak_gb, len(top20))
    return {"rank": r, "flag": f, "score": s, "top20": top20, "n30": n30, "n_all": n_all, "s1c": s1c, "secs": secs, "peak": pm.peak_gb}


def in_sorted(x, ref):
    if len(ref) == 0:
        return np.zeros(len(x), bool)
    pos = np.minimum(np.searchsorted(ref, x), len(ref) - 1)
    return ref[pos] == x


def cmd_blockdiag(c):
    paths = Paths(load_config(CFG["B11"]))
    val_ids = load_val_ids(paths)
    f09 = np.load("experiments/runs/EXP-B-09/analysis/fates_B09_k1000.npz")
    f10s = np.load("experiments/runs/EXP-B-10/analysis/fates_B10_k1000.npz")
    gcode, r09, gctry = f09["gcode"].astype(np.int64), f09["g_rank"], f09["g_country"]
    if not np.array_equal(gcode, f10s["gcode"].astype(np.int64)):
        raise AssertionError("B-09 / B-10 fates not aligned")
    gts = np.sort(gcode)
    R = {n: reduce_run(n, gcode, val_ids, paths) for n in ("B10", "B11")}
    r10, r11, s10, s11, fl11 = R["B10"]["rank"], R["B11"]["rank"], R["B10"]["score"], R["B11"]["score"], R["B11"]["flag"]
    det = {"B10_rerun_ranks_equal_stored_fates": bool(np.array_equal(r10, f10s["g_rank"])), "B10_rerun_flags_equal_stored": bool(np.array_equal(R["B10"]["flag"], f10s["g_flags"])),
           "rank_mismatches": int((r10 != f10s["g_rank"]).sum())}
    c09 = B10.load_parts("scored_EXP-B-09").column("_code").to_numpy()
    s1c = R["B11"]["s1c"]
    ctry_of = {k: np.sort(v) for k, v in s1c.items()}
    nval = sum(len(v) for v in s1c.values())

    def ctry_mask(codes, k):
        return in_sorted(codes // SHIFT, ctry_of[k]) if len(codes) else np.zeros(0, bool)

    rep = {"population": f"frozen validation fold ({nval:,} S1; {len(gcode):,} true pairs)", "top_k": 20,
           "runtime_s": {n: round(R[n]["secs"], 1) for n in R}, "peak_gb": {n: round(R[n]["peak"], 2) for n in R}, "determinism_B10_rerun": det}
    for lab, m in (("all", np.ones(len(gcode), bool)), ("india", gctry == "india"), ("us", gctry == "us")):
        rep[f"recall_{lab}"] = {f"{n}@{k}": float((r[m] < k).mean()) for n, r in (("B09", r09), ("B10", r10), ("B11", r11)) for k in (20, 30)}
        rep[f"recall_{lab}"].update({f"{n}_uncapped(rank<1000)": float((r[m] < 1000).mean()) for n, r in (("B09", r09), ("B10", r10), ("B11", r11))})
        rep[f"recall_{lab}"]["delta@20_B11_vs_B09"] = rep[f"recall_{lab}"]["B11@20"] - rep[f"recall_{lab}"]["B09@20"]
        rep[f"recall_{lab}"]["delta@20_B11_vs_B10"] = rep[f"recall_{lab}"]["B11@20"] - rep[f"recall_{lab}"]["B10@20"]

    lists = {}
    for lab in ("all", "india", "us"):
        m09 = np.ones(len(c09), bool) if lab == "all" else ctry_mask(c09, lab)
        top10, top11 = R["B10"]["top20"], R["B11"]["top20"]
        m10 = np.ones(len(top10), bool) if lab == "all" else ctry_mask(top10, lab)
        m11 = np.ones(len(top11), bool) if lab == "all" else ctry_mask(top11, lab)
        a09, a10, a11 = c09[m09], top10[m10], top11[m11]
        d = {"B09_candidates": int(len(a09)), "B10_candidates": int(len(a10)), "B11_candidates": int(len(a11)),
             "candidate_growth_vs_B09": len(a11) / max(len(a09), 1) - 1}
        for ref_lab, ref in (("vs_B09", a09), ("vs_B10", a10)):
            new, disp = np.setdiff1d(a11, ref), np.setdiff1d(ref, a11)
            nt, dt = int(in_sorted(new, gts).sum()), int(in_sorted(disp, gts).sum())
            d[ref_lab] = {"new_candidates": int(len(new)), "new_true": nt, "displaced_candidates": int(len(disp)), "displaced_true": dt,
                          "net_true": nt - dt, "displaced_true/new_true": dt / max(nt, 1)}
        lists[lab] = d
    rep["top20_lists"] = lists
    n_s1_c = {k: len(v) for k, v in s1c.items()}
    rep["candidates_per_s1_top20"] = {"B09": len(c09) / nval, "B10": len(R["B10"]["top20"]) / nval, "B11": len(R["B11"]["top20"]) / nval,
                                      "B11_top30": R["B11"]["n30"] / nval, "B11_uncapped(rank<1000)": R["B11"]["n_all"] / nval,
                                      "B10_uncapped(rank<1000)": R["B10"]["n_all"] / nval, "s1_per_country": n_s1_c}
    nk = r09 >= 1000
    rep["no_key_pairs_under_B09"] = {"pairs": int(nk.sum()), "B10_generated": int((r10[nk] < 1000).sum()), "B10_top20": int((r10[nk] < 20).sum()),
                                     "B11_generated": int((r11[nk] < 1000).sum()), "B11_top20": int((r11[nk] < 20).sum())}

    # ---- rank repair: B-10 true pairs generated but ranked beyond 20 (not generated by B-09 at any rank)
    tgt = (r09 >= 1000) & (r10 >= 20) & (r10 < 1000)
    tgt_any = (r10 >= 20) & (r10 < 1000)  # every true pair B-10 ranked beyond 20
    entered = tgt & (r11 < 20)
    rk = r11[tgt].astype(np.int64)
    dscore = (s11 - s10)[tgt]
    # did an entering pair's S1 lose a true pair that B-10 had in its top-20?
    lost_b10 = (r10 < 20) & (r11 >= 20)
    s1_lost = np.unique(gcode[lost_b10] // SHIFT)
    s1_entered = gcode[entered] // SHIFT
    rr = {"definition": "validation true pairs with B-09 rank >= 1000 (not generated) and 20 <= B-10 rank < 1000 (the B-10 report's 1,441)",
          "pairs": int(tgt.sum()), "entered_top20_B11": int(entered.sum()), "entered_top30_B11": int((tgt & (r11 < 30)).sum()),
          "remain_beyond_20": int((tgt & (r11 >= 20)).sum()),
          "B10_rank_quantiles(p10,p50,p90)": np.quantile(r10[tgt], [0.1, 0.5, 0.9]).tolist() if tgt.any() else None,
          "B11_rank_quantiles(p10,p50,p90)": np.quantile(rk, [0.1, 0.5, 0.9]).tolist() if tgt.any() else None,
          "rank_change_quantiles(B11-B10; p10,p50,p90)": np.quantile(rk - r10[tgt], [0.1, 0.5, 0.9]).tolist() if tgt.any() else None,
          "rank_improved": int((rk < r10[tgt]).sum()), "rank_worse": int((rk > r10[tgt]).sum()),
          "cheap_score_change_quantiles(p10,p50,p90)": np.nanquantile(dscore, [0.1, 0.5, 0.9]).tolist() if tgt.any() else None,
          "entered_pairs_whose_S1_lost_a_B10_top20_true_pair": int(in_sorted(s1_entered, s1_lost).sum()),
          "all_true_pairs_B10_rank_beyond_20": {"pairs": int(tgt_any.sum()), "entered_top20_B11": int((tgt_any & (r11 < 20)).sum())},
          "B10_top20_true_pairs": {"pairs": int((r10 < 20).sum()), "stay_in_B11_top20": int(((r10 < 20) & (r11 < 20)).sum()), "pushed_out": int(lost_b10.sum()),
                                   "pushed_out_rank_quantiles(p50,p90)": np.quantile(r11[lost_b10], [0.5, 0.9]).tolist() if lost_b10.any() else None,
                                   "pushed_out_have_name_token_flag": int(((R["B10"]["flag"] & NT_BIT) > 0)[lost_b10].sum())},
          "B09_top20_true_pairs": {"pairs": int((r09 < 20).sum()), "stay_in_B11_top20": int(((r09 < 20) & (r11 < 20)).sum()), "pushed_out": int(((r09 < 20) & (r11 >= 20)).sum())},
          "per_country": {g: {"pairs": int((tgt & (gctry == g)).sum()), "entered_top20_B11": int((entered & (gctry == g)).sum())} for g in ("india", "us")}}
    # per-pair table (compact) for the targeted pairs
    idx = np.flatnonzero(tgt)
    np.savez_compressed(OUT / "rank_repair_pairs.npz", gcode=gcode[idx], b10_rank=r10[idx], b11_rank=r11[idx], b10_score=s10[idx], b11_score=s11[idx],
                        entered=entered[idx], country=gctry[idx], s1_lost_b10_true=in_sorted(gcode[idx] // SHIFT, s1_lost))

    # ---- structure (pair profiles) of the moving sets
    ent09 = (r09 >= 20) & (r11 < 20)
    dis09 = (r09 < 20) & (r11 >= 20)
    stay_out = tgt & (r11 >= 20)
    sets = {"entered_top20_vs_B09": ent09, "displaced_from_top20_vs_B09": dis09, "rank_repaired(entered, of the B10 beyond-20 set)": entered,
            "still_beyond_20(of the B10 beyond-20 set)": stay_out, "pushed_out_vs_B10": lost_b10}
    union = np.flatnonzero(np.logical_or.reduce(list(sets.values())))
    xmap = FA.xmap_b06(c)
    name_df, word_df = FA.pool_df_fast(c, xmap)
    s1v, pv = FA.load_views(c, "train", gcode[union] // SHIFT, gcode[union] % SHIFT, xmap)
    prof = {}
    for i in union.tolist():
        a, b = s1v.get(int(gcode[i] // SHIFT)), pv.get(int(gcode[i] % SHIFT))
        if a is not None and b is not None:
            prof[i] = pair_profile(a, b, name_df.get(a["country"], {}), word_df.get(a["country"], {}))

    def cats(sel):
        rows = [prof[i] for i in np.flatnonzero(sel).tolist() if i in prof]
        n = max(len(rows), 1)
        share = lambda f: round(float(sum(bool(f(r)) for r in rows) / n), 3)  # noqa: E731
        return {"pairs": int(sel.sum()), "name_only": share(lambda r: r["name_only"]), "record_address_empty": share(lambda r: r["rec_addr_empty"]),
                "address_missing(either side)": share(lambda r: r["rec_addr_empty"] or r["s1_addr_empty"]),
                "shares_>=2_name_tokens": share(lambda r: r["xlit_shared"] >= 2), "shared_house_number": share(lambda r: r["shared_number"]),
                "name+number": share(lambda r: r["xlit_shared"] > 0 and r["shared_number"]), "address_only": share(lambda r: r["address_only"]),
                "cross_script": share(lambda r: r["script_mismatch"])}

    struct = {lab: {k: cats(m & (np.ones(len(gcode), bool) if lab == "all" else gctry == lab)) for k, m in sets.items()} for lab in ("all", "india", "us")}
    rr["structure"] = struct["all"]

    # ---- kill gate (notes.md 1.2)
    la = lists["all"]
    ka = rep["recall_all"]["delta@20_B11_vs_B09"]
    kb = la["vs_B09"]["displaced_true/new_true"]
    kc = la["candidate_growth_vs_B09"]
    kd = {g: {"recall@20_delta": rep[f"recall_{g}"]["delta@20_B11_vs_B09"], "displaced/new_true": lists[g]["vs_B09"]["displaced_true/new_true"],
              "candidate_growth": lists[g]["candidate_growth_vs_B09"]} for g in ("india", "us")}
    kd_ok = all(v["recall@20_delta"] >= 0 and v["displaced/new_true"] <= 0.5 and v["candidate_growth"] <= 0.05 for v in kd.values())
    gate = {"KA": {"rule": "recall@20 gain vs B-09 >= +0.004", "value": ka, "pass": bool(ka >= 0.004)},
            "KB": {"rule": "displaced true / new true (top-20 vs B-09) <= 0.25", "value": kb, "pass": bool(kb <= 0.25)},
            "KC": {"rule": "top-20 candidate growth vs B-09 <= +3%", "value": kc, "pass": bool(kc <= 0.03)},
            "KD": {"rule": "each country: recall@20 delta >= 0, displaced/new true <= 0.5, candidate growth <= +5%", "value": kd, "pass": bool(kd_ok)}}
    gate["all_pass"] = all(v["pass"] for k, v in gate.items() if k.startswith("K"))
    rep["kill_gate"] = gate
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(RUN / "blocking_forensics.json", rep)
    write_json(RUN / "rank_repair_analysis.json", rr)
    write_json(RUN / "displacement_analysis.json", {"population": "frozen validation fold, top-20 lists", "lists": lists,
                                                    "B10_top20_true_pairs": rr["B10_top20_true_pairs"], "B09_top20_true_pairs": rr["B09_top20_true_pairs"],
                                                    "structure_displaced_vs_B09": struct["all"]["displaced_from_top20_vs_B09"],
                                                    "structure_pushed_out_vs_B10": struct["all"]["pushed_out_vs_B10"]})
    for g in ("india", "us"):
        write_json(RUN / f"{g}_analysis.json", {"stage": "blocking-only (kill gate)", "recall": rep[f"recall_{g}"], "top20_lists": lists[g],
                                                 "rank_repair": rr["per_country"][g], "structure": struct[g],
                                                 "no_key_recovered": {"B10_generated": int(((r09 >= 1000) & (r10 < 1000) & (gctry == g)).sum()),
                                                                      "B11_top20": int(((r09 >= 1000) & (r11 < 20) & (gctry == g)).sum())}})
    np.savez_compressed(OUT / "fates_B11_k1000.npz", gcode=gcode, g_rank=r11, g_flags=fl11, g_score=s11, g_country=gctry)
    LOG.info("blockdiag: recall@20 B09 %.4f B10 %.4f B11 %.4f (delta vs B09 %+.4f) | new true %d displaced true %d | growth %+.4f | repaired %d/%d | gate %s",
             rep["recall_all"]["B09@20"], rep["recall_all"]["B10@20"], rep["recall_all"]["B11@20"], ka, la["vs_B09"]["new_true"], la["vs_B09"]["displaced_true"],
             kc, rr["entered_top20_B11"], rr["pairs"], {k: v["pass"] for k, v in gate.items() if k.startswith("K")})


CMDS = {"blockdiag": cmd_blockdiag}


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
