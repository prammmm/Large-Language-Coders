"""FINAL RESEARCH AUDIT (diagnostic only; read-only w.r.t. every production artefact). Seed 42, deterministic.

    python scripts/analysis/final_audit.py budget | nokey | keys | testdist | crossscript

budget       current error budget of the controls (B-06 + D-01 and S-02), E half B, by country and cardinality [M]
nokey        structure of validation true pairs by blocker fate (top-20 / 21-30 / 31-1000 / no key) [M]
keys         key-family upper bounds on the validation fold: recovered no-key / never-retrieved pairs, candidate growth [M/UB]
testdist     train (E) vs test distributions, label-free [M train / D test]
crossscript  cross-script true pairs: share, retrieval and decision recall by script combination [M]

Inputs: experiments/runs/FINAL_RESEARCH_AUDIT/fates_B06_k1000.npz (scripts/analysis/blocking_forensics.py --raw-topk 1000 --dump).
Outputs: experiments/runs/FINAL_RESEARCH_AUDIT/*.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_b08 as B  # noqa: E402
import exp_s02_shadow as X  # noqa: E402
import post_s01_forensics as F  # noqa: E402

from ber.decision.stage2 import restrict_additions  # noqa: E402
from ber.forensics.audit import (cardinality_bin, emit_pairs, encode_views, family_keys, oracle_gain, pair_profile,  # noqa: E402
                                 script_class)
from ber.forensics.diagnostics import summary  # noqa: E402
from ber.forensics.shadow import weighted_metrics  # noqa: E402
from ber.pipeline import run_stage2 as S  # noqa: E402
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, write_json  # noqa: E402

OUT = Path("experiments/runs/FINAL_RESEARCH_AUDIT")
SHIFT = np.int64(1) << np.int64(24)
FATES = OUT / "fates_B06_k1000.npz"
VIEW = ["name_norm", "name_core", "addr_norm", "addr_nums", "addr_words"]
NOT_GEN = 10**6


def fate_class(rank: np.ndarray) -> np.ndarray:
    return np.select([rank < 20, rank < 30, rank < 1000], ["top20", "rank21_30", "rank31_1000"], "no_key")


# ----------------------------------------------------------------------------- views
def load_views(c, split: str, s1_rows=None, pool_rows=None, xmap=None) -> tuple[dict, dict]:
    """row -> {view: str} for the requested S1 rows and GLOBAL pool rows (S2 rows, then S3 rows + n_s2)."""
    man = F.load_manifest(c.paths, split)
    n_s2 = int(man["source2"]["rows"])
    out_s1, out_pool = {}, {}
    todo = [("source1", 0, set(np.asarray(s1_rows).tolist()) if s1_rows is not None else set(), out_s1)]
    pr = np.asarray(pool_rows) if pool_rows is not None else np.zeros(0, np.int64)
    todo += [("source2", 0, set(pr[pr < n_s2].tolist()), out_pool), ("source3", n_s2, set((pr[pr >= n_s2] - n_s2).tolist()), out_pool)]
    for source, off, want, dst in todo:
        if not want:
            continue
        wa = pa.array(sorted(want), pa.int32())
        for key in man[source]["countries"]:
            t = F.read_country(c.paths, split, source, key, ["row"] + VIEW)
            t = t.filter(pc.is_in(t.column("row"), value_set=wa))
            if t.num_rows == 0:
                continue
            core = [x or "" for x in t.column("name_core").to_pylist()]
            xl = xmap.apply(core) if xmap is not None else core
            cols = {v: t.column(v).to_pylist() for v in VIEW}
            for i, r in enumerate(t.column("row").to_pylist()):
                d = {v: (cols[v][i] or "") for v in VIEW}
                d["name_xlit"] = xl[i] or ""
                d["country"] = key
                dst[r + off] = d
    return out_s1, out_pool


def country_pool(c, key: str, xmap, split: str = "train") -> pa.Table:
    """Pool of one country (S2 rows then S3 rows) with name_xlit, addr_words, addr_nums and the global pool row."""
    man = F.load_manifest(c.paths, split)
    n_s2 = int(man["source2"]["rows"])
    parts = []
    for source, off in (("source2", 0), ("source3", n_s2)):
        t = F.read_country(c.paths, split, source, key, ["row", "name_core", "addr_words", "addr_nums"])
        core = pc.fill_null(t.column("name_core"), "")
        xl = xmap.apply([x or "" for x in core.to_pylist()]) if xmap is not None else core
        parts.append(pa.table({"grow": pc.add(t.column("row").cast(pa.int64()), off), "name_xlit": pa.array(xl, pa.string()),
                               "addr_words": pc.fill_null(t.column("addr_words"), ""), "addr_nums": pc.fill_null(t.column("addr_nums"), "")}))
    return pa.concat_tables(parts)


def country_s1(c, key: str, xmap, rows_mask: np.ndarray, split: str = "train") -> pa.Table:
    t = F.read_country(c.paths, split, "source1", key, ["row", "name_core", "addr_words", "addr_nums"])
    t = t.filter(pa.array(rows_mask[t.column("row").to_numpy()]))
    core = pc.fill_null(t.column("name_core"), "")
    xl = xmap.apply([x or "" for x in core.to_pylist()]) if xmap is not None else core
    return pa.table({"grow": t.column("row").cast(pa.int64()), "name_xlit": pa.array(xl, pa.string()),
                     "addr_words": pc.fill_null(t.column("addr_words"), ""), "addr_nums": pc.fill_null(t.column("addr_nums"), "")})


def xmap_b06(c):
    return F.XlitMap.load(Path(c.cfg["stage2"]["xlit_map"]))


# ----------------------------------------------------------------------------- budget
def cmd_budget(c):
    gts = np.sort(c.gtc)
    n = c.n
    d06 = B.decisions(c, "train", B.TAG["B06"], 0.67)
    # S-02 = B-06 + D-01 AND S-01
    s1, pool, prob, sc, meta = F.stage2_arrays(F.CFG_S01, "train")
    code = s1 * SHIFT + pool
    s01 = S.decide(s1, pool, prob, meta["threshold"], meta["variant"])
    s02c = np.sort(code[restrict_additions(np.isin(code, d06["codes"]), s01, sc, np.inf)])
    del s1, pool, prob, sc, code, s01
    # true candidates of the B-06 lists (top-20) with their stage-1 score
    tc, ts = [], []
    for p in sorted(c.paths.scored_dir("train", B.TAG["B06"]).glob("*.parquet")):
        tb = pq.read_table(p, columns=["s1_row", "pool_row", "score"])
        cd = tb.column("s1_row").to_numpy().astype(np.int64) * SHIFT + tb.column("pool_row").to_numpy().astype(np.int64)
        y = B.is_true(cd, gts)
        tc.append(cd[y])
        ts.append(tb.column("score").to_numpy().astype(np.float32)[y])
    tc, ts = np.concatenate(tc), np.concatenate(ts)
    o = np.argsort(tc)
    tc, ts = tc[o], ts[o]
    card = cardinality_bin(c.n_true)
    rep = {"definition": "oracle F0.5 gain [UB] from fixing ONE bucket alone; population E half B (held out); macro per-S1 F0.5", "systems": {}}
    for sysname, acc in (("B06+D01(public-control stage 1)", d06["codes"]), ("S02(offline KEEP)", s02c)):
        ya = B.is_true(acc, gts)
        tp, k = np.bincount(acc[ya] // SHIFT, minlength=n), np.bincount(acc // SHIFT, minlength=n)
        fn = gts[~B.is_true(gts, acc)]
        pos = np.minimum(np.searchsorted(tc, fn), len(tc) - 1)
        inc = tc[pos] == fn
        fs = np.where(inc, ts[pos], np.nan)
        in_ctl = B.is_true(fn, d06["codes"])  # accepted by B-06 + D-01 (only possible for S-02: vetoed)
        buckets = {"A_never_retrieved(not in top-20 list)": ~inc, "B1_retrieved_score<0.02": inc & (fs < 0.02),
                   "B2_retrieved_score_0.02-0.5": inc & (fs >= 0.02) & (fs < 0.5), "B3_retrieved_score_0.5-0.67": inc & (fs >= 0.5) & (fs < 0.67),
                   "E_score>=0.67_lost_D01": inc & (fs >= 0.67) & ~in_ctl, "G_vetoed_by_S01(S-02 only)": in_ctl}
        fp = acc[~ya]
        single = c.n_true == 0
        fpb = {"C_false_merges(S1 with true matches)": ~single[fp // SHIFT], "D_singleton_false_positives": single[fp // SHIFT]}
        R = {}
        for pname, pm in (("E_half_B", c.P["E_half_B"]), ("E_half_B:india", c.P["E_half_B"] & (c.ctry == "india")), ("E_half_B:us", c.P["E_half_B"] & (c.ctry == "us"))):
            base = weighted_metrics(tp[pm], k[pm], c.n_true[pm])
            r = {"s1": int(pm.sum()), "true_pairs": int(c.n_true[pm].sum()), "f05": base["f05"], "precision": base["precision"], "recall": base["recall"], "buckets": {}}
            for bname, bm in buckets.items():
                a = np.bincount(fn[bm] // SHIFT, minlength=n)
                r["buckets"][bname] = {"true_pairs": int(a[pm].sum()), "share_of_true_pairs": float(a[pm].sum() / max(c.n_true[pm].sum(), 1)),
                                       "s1_affected": int((a[pm] > 0).sum()), "oracle_gain[UB]": oracle_gain(tp[pm], k[pm], c.n_true[pm], add_true=a[pm]),
                                       "by_cardinality": {str(b): int(a[pm & (card == b)].sum()) for b in range(1, 5)}}
            for bname, bm in fpb.items():
                rm = np.bincount(fp[bm] // SHIFT, minlength=n)
                r["buckets"][bname] = {"false_pairs": int(rm[pm].sum()), "s1_affected": int((rm[pm] > 0).sum()),
                                       "oracle_gain[UB]": oracle_gain(tp[pm], k[pm], c.n_true[pm], remove_false=rm[pm]),
                                       "by_cardinality": {str(b): int(rm[pm & (card == b)].sum()) for b in range(0, 5)}}
            eop = pm & (c.n_true > 0) & (k == 0)
            r["buckets"]["F_empty_on_positive(overlaps A/B/E)"] = {"s1": int(eop.sum()), "share_of_positive_s1": float(eop.sum() / max((pm & (c.n_true > 0)).sum(), 1)),
                                                                  "true_pairs": int(c.n_true[eop].sum())}
            r["cardinality_mix"] = {str(b): int((pm & (card == b)).sum()) for b in range(5)}
            r["f05_by_cardinality"] = {str(b): weighted_metrics(tp[pm & (card == b)], k[pm & (card == b)], c.n_true[pm & (card == b)])["f05"] for b in range(5)}
            R[pname] = r
        R["D01_effect(all train S1)"] = {"conflicted_records_before": d06["conflicts_before_d01"]["conflicted_records"],
                                         "pairs_removed_by_D01": int(d06["conflicts_before_d01"]["predicted_pairs"] - d06["conflicts_after_d01"]["predicted_pairs"])}
        rep["systems"][sysname] = R
        LOG.info("%s: %s", sysname, {k2: round(v.get("oracle_gain[UB]", 0), 4) for k2, v in R["E_half_B"]["buckets"].items()})
    write_json(OUT / "current_error_budget.json", rep)


# ----------------------------------------------------------------------------- no-key forensics
def profile_rows(pairs_s1, pairs_pool, s1v, pv, name_df, word_df) -> list[dict]:
    out = []
    for s, p in zip(pairs_s1.tolist(), pairs_pool.tolist()):
        a, b = s1v.get(s), pv.get(p)
        if a is None or b is None:
            continue
        ctry = a["country"]
        out.append(pair_profile(a, b, name_df.get(ctry, {}), word_df.get(ctry, {})) | {"country": ctry})
    return out


def summarise_profiles(rows: list[dict]) -> dict:
    if not rows:
        return {"pairs": 0}
    keys = [k for k in rows[0] if k not in ("country", "s1_script", "rec_script")]
    out = {"pairs": len(rows)}
    for k in keys:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        if set(np.unique(v[~np.isnan(v)])) <= {0.0, 1.0} and k not in ("xlit_shared", "name_core_shared", "name_norm_shared", "shared_words", "s1_words", "rec_words",
                                                                             "rare_xlit_shared", "shared_rare_words", "shared_common_words", "xlit_prefix_len"):
            out[k] = float(np.nanmean(v))
        else:
            out[k] = {"mean": float(np.nanmean(v)), "p25": float(np.nanquantile(v, 0.25)), "median": float(np.nanquantile(v, 0.5)), "p75": float(np.nanquantile(v, 0.75)),
                      "share_ge_1": float(np.nanmean(v >= 1)), "share_ge_2": float(np.nanmean(v >= 2))}
    sc = {}
    for r in rows:
        sc[f"{r['s1_script']}->{r['rec_script']}"] = sc.get(f"{r['s1_script']}->{r['rec_script']}", 0) + 1
    out["script_combinations"] = {k: v / len(rows) for k, v in sorted(sc.items(), key=lambda x: -x[1])}
    return out


def cmd_nokey(c):
    f = np.load(FATES)
    rank, gs, gp, gctry = f["g_rank"], f["gs"].astype(np.int64), f["gp"].astype(np.int64), f["g_country"]
    cls = fate_class(rank)
    xmap = xmap_b06(c)
    name_df, word_df = pool_df_fast(c, xmap)
    rng = np.random.default_rng(42)
    rep = {"population": "frozen validation fold true pairs (diagnostic only)", "true_pairs": int(len(rank)),
           "fate_counts": {k: int((cls == k).sum()) for k in ("top20", "rank21_30", "rank31_1000", "no_key")},
           "fate_by_country": {g: {k: int(((cls == k) & (gctry == g)).sum()) for k in ("top20", "rank21_30", "rank31_1000", "no_key")} for g in np.unique(gctry).tolist()}}
    sel = {}
    for k in ("no_key", "rank31_1000", "rank21_30"):
        sel[k] = np.flatnonzero(cls == k)
    top = np.flatnonzero(cls == "top20")
    sel["top20(sample 30k)"] = np.sort(rng.choice(top, min(30000, len(top)), replace=False))
    allidx = np.unique(np.concatenate(list(sel.values())))
    s1v, pv = load_views(c, "train", gs[allidx], gp[allidx], xmap)
    prof = {}
    for k, idx in sel.items():
        rows = profile_rows(gs[idx], gp[idx], s1v, pv, name_df, word_df)
        prof[k] = {"all": summarise_profiles(rows)}
        for g in ("india", "us"):
            prof[k][g] = summarise_profiles([r for r in rows if r["country"] == g])
    rep["profiles"] = prof
    write_json(OUT / "no_key_forensics.json", rep)
    LOG.info("nokey: %s", rep["fate_counts"])


def pool_df_fast(c, xmap) -> tuple[dict, dict]:
    from ber.blocking.blocker import encode_columns
    man = F.load_manifest(c.paths, "train")
    name_df, word_df = {}, {}
    for key in man["source1"]["countries"]:
        pool = country_pool(c, key, xmap)
        for col, dst in (("name_xlit", name_df), ("addr_words", word_df)):
            rec, tok, vocab = encode_columns([pool.column(col)], True)
            dst[key] = dict(zip(vocab.to_pylist(), np.bincount(tok, minlength=len(vocab)).tolist()))
        del pool
    return name_df, word_df


# ----------------------------------------------------------------------------- key hypotheses
VARIANTS = [  # (label, family, k_a, k_b, max_df, cap, status)
    ("existing: name_token.pair (2 rarest name tokens)", "tok_pair", 2, 0, 0, 50, "existing"),
    ("existing: addr_pair (3 rarest useful words)", "word_pair", 3, 0, 0, 50, "existing"),
    ("existing: name_addr (2 name tokens x 3 words)", "tok_x_word", 2, 3, 0, 50, "existing"),
    ("existing: addr_num_word (2 numbers x 4 words)", "num_x_word", 2, 4, 0, 50, "existing"),
    ("existing: addr_rare_word (word DF<=20)", "single_word", 0, 0, 20, 10**9, "existing"),
    ("widen: name_token.pair 3 rarest tokens", "tok_pair", 3, 0, 0, 50, "widen"),
    ("widen: name_token.pair cap 200", "tok_pair", 2, 0, 0, 200, "widen"),
    ("widen: addr_pair 5 rarest words", "word_pair", 5, 0, 0, 50, "widen"),
    ("widen: name_addr 3 tokens x 6 words", "tok_x_word", 3, 6, 0, 50, "widen"),
    ("widen: addr_num_word 3 numbers x 8 words", "num_x_word", 3, 8, 0, 50, "widen"),
    ("new: name token x house number (2 x 2)", "tok_x_num", 2, 2, 0, 50, "new"),
    ("new: name token x house number (3 x 3)", "tok_x_num", 3, 3, 0, 50, "new"),
    ("new: single rare name token (DF<=20)", "single_tok", 0, 0, 20, 10**9, "new"),
    ("new: single name token (DF<=200)", "single_tok", 0, 0, 200, 10**9, "new"),
]


def cmd_keys(c):
    """Memory-safe: gross pairs are computed arithmetically; recovery of missed true pairs uses share_key (no emission);
    pairs are materialised only in S1 chunks and only when the gross count is <= MAX_EMIT."""
    from ber.forensics.audit import gross_pairs, pool_key_index, share_key
    MAX_EMIT = 60_000_000
    f = np.load(FATES)
    rank, gs, gp = f["g_rank"], f["gs"].astype(np.int64), f["gp"].astype(np.int64)
    cls = fate_class(rank)
    xmap = xmap_b06(c)
    val = c.P["validation"]
    ex = []
    for q in sorted(c.paths.scored_dir("train", "scored_EXP-B-06").glob("*.parquet")):
        tb = pq.read_table(q, columns=["s1_row", "pool_row"])
        ex.append(tb.column("s1_row").to_numpy().astype(np.int64) * SHIFT + tb.column("pool_row").to_numpy().astype(np.int64))
    ex = np.unique(np.concatenate(ex))
    gts = np.sort(c.gtc)
    missed = rank >= 20
    nokey = cls == "no_key"
    rep = {"population": "validation S1 (diagnostic; [UB] upper bounds)", "no_key_pairs": int(nokey.sum()), "never_retrieved_top20_pairs": int(missed.sum()),
           "existing_top20_candidates": int(len(ex)), "validation_s1": int(val.sum()), "max_materialised_pairs": MAX_EMIT, "variants": {}}
    per = {}
    for key in F.load_manifest(c.paths, "train")["source1"]["countries"]:
        t0 = time.perf_counter()
        with PeakMemory() as pm:
            s1t = country_s1(c, key, xmap, val)
            pool = country_pool(c, key, xmap)
            enc = encode_views(s1t, pool)
            s1g, pg = s1t.column("grow").to_numpy(), pool.column("grow").to_numpy()
            s1_loc = {int(g): i for i, g in enumerate(s1g)}
            p_loc = {int(g): i for i, g in enumerate(pg)}
            sel = np.flatnonzero(missed & np.isin(gs, s1g))
            ls = np.array([s1_loc[int(x)] for x in gs[sel]], np.int64)
            lp = np.array([p_loc.get(int(x), -1) for x in gp[sel]], np.int64)
            okp = lp >= 0
            sel, ls, lp = sel[okp], ls[okp], lp[okp]
            is_nk = nokey[sel]
            for label, fam, ka, kb, mdf, cap, status in VARIANTS:
                r, k = family_keys(enc, fam, ka, kb, max_df=mdf)
                uk, cnt, sr, sk = pool_key_index(r, k, enc["n_s1"])
                g, over = gross_pairs(uk, cnt, sr, sk, cap)
                shc = share_key(r, k, enc["n_s1"], ls, lp, cap=cap)
                shu = share_key(r, k, enc["n_s1"], ls, lp, cap=None)
                d = per.setdefault(label, {"family": fam, "k": [ka, kb], "max_df": mdf, "cap": cap, "status": status, "by_country": {}})
                e = {"gross_pairs": int(g.sum()), "gross_per_s1_mean": float(g.sum() / max(len(s1g), 1)), "gross_per_s1_p99": float(np.quantile(np.pad(g, (0, max(0, len(s1g) - len(g)))), 0.99)),
                     "s1_keys_over_cap": over, "no_key_recovered(capped)": int(shc[is_nk].sum()), "no_key_recovered(uncapped)": int(shu[is_nk].sum()),
                     "never_retrieved_recovered(capped)": int(shc.sum()), "never_retrieved_recovered(uncapped)": int(shu.sum()),
                     "no_key_pairs_in_country": int(is_nk.sum()), "never_retrieved_in_country": int(len(sel))}
                if e["gross_pairs"] <= MAX_EMIT:
                    new = tru = 0
                    for lo in range(0, len(s1g), 20000):
                        m = np.zeros(len(s1g), bool)
                        m[lo:lo + 20000] = True
                        a_, b_, _ = emit_pairs(r, k, enc["n_s1"], cap, s1_mask=m)
                        code = np.unique(s1g[a_] * SHIFT + pg[b_])
                        nw = code[~B.is_true(code, ex)]
                        new += len(nw)
                        tru += int(B.is_true(nw, gts).sum())
                    e |= {"new_pairs_not_in_top20": new, "new_true": tru, "new_false": new - tru, "materialised": True}
                else:
                    e |= {"materialised": False, "note": "gross pairs > MAX_EMIT: candidate explosion, not materialised"}
                d["by_country"][key] = e
                del r, k, uk, cnt, sr, sk, g
            del enc, s1t, pool
        LOG.info("keys %s: %.0fs peak %.2f GB", key, time.perf_counter() - t0, pm.peak_gb)
    nval = int(val.sum())
    for label, d in per.items():
        bc = d["by_country"]
        tot = {m: sum(bc[g][m] for g in bc) for m in ("gross_pairs", "no_key_recovered(capped)", "no_key_recovered(uncapped)", "never_retrieved_recovered(capped)",
                                                     "never_retrieved_recovered(uncapped)", "no_key_pairs_in_country", "never_retrieved_in_country")}
        tot["no_key_recall(capped)[UB]"] = tot["no_key_recovered(capped)"] / max(tot["no_key_pairs_in_country"], 1)
        tot["no_key_recall(uncapped)[UB]"] = tot["no_key_recovered(uncapped)"] / max(tot["no_key_pairs_in_country"], 1)
        tot["never_retrieved_recall(capped)[UB]"] = tot["never_retrieved_recovered(capped)"] / max(tot["never_retrieved_in_country"], 1)
        tot["gross_pairs_per_validation_s1"] = tot["gross_pairs"] / nval
        if all(bc[g].get("materialised") for g in bc):
            for m in ("new_pairs_not_in_top20", "new_true", "new_false"):
                tot[m] = sum(bc[g][m] for g in bc)
            tot["new_pairs_per_validation_s1"] = tot["new_pairs_not_in_top20"] / nval
            tot["new_true_share"] = tot["new_true"] / max(tot["new_pairs_not_in_top20"], 1)
        d["total"] = tot
        rep["variants"][label] = d
    write_json(OUT / "key_hypothesis_upper_bounds.json", rep)


# ----------------------------------------------------------------------------- train vs test distributions
def record_stats(t: pa.Table) -> dict:
    core = pc.fill_null(t.column("name_core"), "")
    addr = pc.fill_null(t.column("addr_norm"), "")
    nums = pc.fill_null(t.column("addr_nums"), "")
    ntok = pc.list_value_length(pc.split_pattern(core, " ")).to_numpy(zero_copy_only=False).astype(np.float64)
    ntok[np.asarray(pc.equal(core, ""), bool)] = 0
    atok = pc.list_value_length(pc.split_pattern(addr, " ")).to_numpy(zero_copy_only=False).astype(np.float64)
    atok[np.asarray(pc.equal(addr, ""), bool)] = 0
    nonlatin = np.asarray(pc.match_substring_regex(core, r"[^\x00-\x{024F}]"), bool)
    latin = np.asarray(pc.match_substring_regex(core, r"[A-Za-z]"), bool)
    return {"records": t.num_rows, "name_tokens_mean": float(ntok.mean()), "addr_tokens_mean": float(atok.mean()),
            "name_empty": float((ntok == 0).mean()), "addr_empty": float((atok == 0).mean()), "has_number": float(np.asarray(pc.not_equal(nums, ""), bool).mean()),
            "script_nonlatin_only": float((nonlatin & ~latin).mean()), "script_mixed": float((nonlatin & latin).mean()), "script_latin": float((~nonlatin).mean())}


def cmd_testdist(c):
    rep = {"labels": "train = E population S1 / train pool [M]; test = all test S1 / test pool [D, unlabeled]"}
    for split in ("train", "test"):
        man = F.load_manifest(c.paths, split)
        n_s2 = int(man["source2"]["rows"])
        r = {}
        for key in man["source1"]["countries"]:
            t = F.read_country(c.paths, split, "source1", key, ["row", "name_core", "addr_norm", "addr_nums"])
            if split == "train":
                t = t.filter(pa.array(c.P["E"][t.column("row").to_numpy()]))
            r[f"S1:{key}"] = record_stats(t)
            pt = pa.concat_tables([F.read_country(c.paths, split, s, key, ["name_core", "addr_norm", "addr_nums"]) for s in ("source2", "source3")])
            r[f"pool:{key}"] = record_stats(pt)
            r[f"pool_records_per_S1:{key}"] = pt.num_rows / max(t.num_rows if split == "test" else int(((c.ctry == key)).sum()), 1)
        rep[split] = r
    # candidate / score density (B-06 lists), claimant density (S-01 feature parts)
    for split, tag, m in (("train", B.TAG["B06"], c.P["E"]), ("test", B.TEST_TAG["B06"], None)):
        s1s, scs = [], []
        for p in sorted(c.paths.scored_dir(split, tag).glob("*.parquet")):
            tb = pq.read_table(p, columns=["s1_row", "score"])
            s1s.append(tb.column("s1_row").to_numpy().astype(np.int64))
            scs.append(tb.column("score").to_numpy())
        s1, sc = np.concatenate(s1s), np.concatenate(scs)
        if m is not None:
            keep = m[s1]
            s1, sc = s1[keep], sc[keep]
        nS1 = int(m.sum()) if m is not None else len(c.ctry_te)
        ctry = (c.ctry if split == "train" else c.ctry_te)[s1]
        rep[split]["candidates"] = {g: {"cand_per_s1": float((ctry == g).sum() / max(((c.ctry if split == 'train' else c.ctry_te) == g)[m if m is not None else slice(None)].sum(), 1)),
                                        "score_ge_0.67_per_s1": float(((ctry == g) & (sc >= np.float32(0.67))).sum() / max(((c.ctry if split == 'train' else c.ctry_te) == g)[m if m is not None else slice(None)].sum(), 1)),
                                        "score_0.1_0.67_per_s1": float(((ctry == g) & (sc >= 0.1) & (sc < np.float32(0.67))).sum() / max(((c.ctry if split == 'train' else c.ctry_te) == g)[m if m is not None else slice(None)].sum(), 1)),
                                        "score_quantiles": summary(sc[ctry == g])} for g in np.unique(ctry).tolist()}
        rep[split]["candidates"]["all"] = {"cand_per_s1": float(len(s1) / nS1), "score_ge_0.67_per_s1": float((sc >= np.float32(0.67)).sum() / nS1)}
        x = F.read_cols(split, ["s1_row", "rec_n_claim", "score", "peer_max_sim"])
        xs = x["s1_row"].astype(np.int64)
        keep = m[xs] if m is not None else np.ones(len(xs), bool)
        rc = x["rec_n_claim"][keep]
        la = (x["score"][keep] < 0.5) & (rc <= 1) & (x["peer_max_sim"][keep] >= 0.8)
        rep[split]["claimants(eligible pairs, score>=0.02)"] = {"eligible_per_s1": float(keep.sum() / nS1), "mean_claimants": float(rc.mean()),
                                                                "contested_share": float((rc >= 2).mean()), "lookalike_per_s1": float(la.sum() / nS1)}
        del s1, sc, x
    write_json(OUT / "test_distribution.json", rep)
    LOG.info("testdist done")


# ----------------------------------------------------------------------------- cross-script
def cmd_crossscript(c):
    f = np.load(FATES)
    rank, gs, gp, gcode = f["g_rank"], f["gs"].astype(np.int64), f["gp"].astype(np.int64), f["gcode"].astype(np.int64)
    s1v, pv = load_views(c, "train", gs, gp, None)
    combo = np.array([f"{script_class(s1v.get(int(s), {}).get('name_core', ''))}->{script_class(pv.get(int(p), {}).get('name_core', ''))}" for s, p in zip(gs, gp)])
    ctry = np.array([s1v.get(int(s), {}).get("country", "?") for s in gs])
    d06 = B.decisions(c, "train", B.TAG["B06"], 0.67)
    acc06 = B.is_true(gcode, d06["codes"])
    s1, pool, prob, sc, meta = F.stage2_arrays(F.CFG_S01, "train")
    code = s1 * SHIFT + pool
    s01 = S.decide(s1, pool, prob, meta["threshold"], meta["variant"])
    s02c = np.sort(code[restrict_additions(np.isin(code, d06["codes"]), s01, sc, np.inf)])
    acc02 = B.is_true(gcode, s02c)
    del s1, pool, prob, sc, code, s01
    rep = {"population": "validation true pairs [M]; script class of name_core (latin / nonlatin / mixed / empty)", "true_pairs": int(len(gs)), "combos": {}}
    for k in sorted(set(combo.tolist()), key=lambda x: -(combo == x).sum()):
        m = combo == k
        rep["combos"][k] = {"pairs": int(m.sum()), "share": float(m.mean()), "india_share": float((ctry[m] == "india").mean()),
                            "recall_top20": float((rank[m] < 20).mean()), "recall_top30": float((rank[m] < 30).mean()), "recall_any_key": float((rank[m] < 1000).mean()),
                            "decision_recall_B06D01": float(acc06[m].mean()), "decision_recall_S02": float(acc02[m].mean())}
    for g in ("india", "us"):
        m = ctry == g
        cross = m & np.array(["nonlatin" in x or "mixed" in x for x in combo]) & np.array([x.split("->")[0] != x.split("->")[1] for x in combo])
        rep[f"{g}:cross_script_share"] = float(cross.sum() / max(m.sum(), 1))
        rep[f"{g}:cross_script_share_of_never_retrieved_top20"] = float((cross & (rank >= 20)).sum() / max((m & (rank >= 20)).sum(), 1))
        rep[f"{g}:cross_script_share_of_no_key"] = float((cross & (rank >= 1000)).sum() / max((m & (rank >= 1000)).sum(), 1))
    write_json(OUT / "cross_script_measured.json", rep)
    LOG.info("crossscript: %s", {k: (v["pairs"], round(v["recall_top20"], 3)) for k, v in rep["combos"].items()})


CMDS = {"budget": cmd_budget, "nokey": cmd_nokey, "keys": cmd_keys, "testdist": cmd_testdist, "crossscript": cmd_crossscript}


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
