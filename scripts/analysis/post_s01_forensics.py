"""POST-S-01 public-regression forensics (READ-ONLY w.r.t. production; deterministic).

    python scripts/analysis/post_s01_forensics.py verify | delta | shift | shadow-build | shadow-features | shadow-eval | leakage

Outputs: experiments/runs/POST-S01-FORENSIC/*.json and experiments/shadow_tests/ (manifest + id lists).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process

from ber.config import Paths, load_config
from ber.data.normalize_records import load_manifest, read_country
from ber.decision.competition import best_claimant_keep, claim_stats
from ber.decision.stage2 import feature_names
from ber.forensics.diagnostics import binned, cell_ids, expected_precision, flip_codes, flip_counts, floor_decision, psi, summary
from ber.forensics.shadow import (cand_bin, chain_orphan_mask, leakage_checks, match_weights, orphan_fraction, orphan_mask, shadow_split,
                                  weighted_metrics)
from ber.normalization.crossscript import XlitMap
from ber.pipeline import run_stage2 as S
from ber.pipeline.run_baseline import _ids, _read_parts
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, write_json

OUT = Path("experiments/runs/POST-S01-FORENSIC")
SHADOW_DIR = Path("experiments/shadow_tests")
CFG_S01 = "configs/exp_s01_stage2.yaml"
CFG_S01B = "configs/exp_s01b_stage2_robust.yaml"
SHIFT = S.SHIFT
# stage-1 systems as they were SUBMITTED (tag, threshold, D-01)
STAGE1 = {"B04": ("forensic_alltrain_B04", "scored_EXP-B-04", 0.67, False), "N02": ("forensic_alltrain_N02", "scored_EXP-N-02", 0.67, True),
          "B07_t067": ("forensic_alltrain_B07", "scored_EXP-B-07", 0.67, True), "B06": ("forensic_alltrain_B06", "scored_EXP-B-06", 0.67, True)}
PUBLIC = {"B04": 0.91, "N02": 0.928, "S01": 0.920}  # as reported by the team; 0.935 = best pre-S-01 file (B-07 or B-06, unattributed)
FEATS_SUMMARY = ["score", "s1_rank", "s1_n_cand", "s1_best", "s1_second", "s1_gap_best", "s1_n_ge_05", "rec_n_claim", "rec_n_claim_ge_05",
                 "rec_best", "rec_gap_best_other", "peer_n", "peer_max_sim", "peer_wmean_sim", "p1_score", "p1_num_jacc", "p1_addr_tokset"]


# ----------------------------------------------------------------------------- context
class Ctx:
    def __init__(self):
        self.cfg = load_config(CFG_S01)
        self.paths = Paths(self.cfg)
        self.P, self.n = S.populations(self.cfg, self.paths)
        self.gtc, self.n_true = S.gt_codes(self.paths)
        self.ids = _ids(self.paths, "train", "source1").to_pylist()
        self.ctry = self.country("train")
        self.ctry_te = self.country("test")

    def country(self, split):
        man = load_manifest(self.paths, split)
        c = np.empty(int(man["source1"]["rows"]), dtype=object)
        for k in man["source1"]["countries"]:
            c[read_country(self.paths, split, "source1", k, ["row"]).column("row").to_numpy()] = k
        return c


def stage1_codes(paths, split, tag, t, d01, drop=None):
    """Accepted codes of a stage-1 system (threshold t, optional D-01); ``drop``: S1 removed from the competition."""
    d = _read_parts(paths.scored_dir(split, tag), ["s1_row", "pool_row", "score"])
    s1, pool, sc = d["s1_row"].astype(np.int64), d["pool_row"].astype(np.int64), d["score"].astype(np.float32)
    del d
    if drop is not None:
        k = ~drop[s1]
        s1, pool, sc = s1[k], pool[k], sc[k]
    acc = sc >= np.float32(t)
    if d01:
        acc = best_claimant_keep(pool, s1, sc, acc)
    return np.sort(s1[acc] * SHIFT + pool[acc])


def stage2_arrays(cfg_path, split, fdir=None):
    cfg = load_config(cfg_path)
    paths = Paths(cfg)
    booster, meta = S.load_stage2(cfg, paths)
    s1, pool, prob, sc = S.predict_all(cfg, paths, split, booster, meta["features"], fdir)
    return s1, pool, prob, sc, meta


def stage2_codes(s1, pool, prob, sc, meta, variant="model"):
    t = np.float32(meta["threshold"])
    acc = prob >= t
    if variant == "model":
        acc = S.decide(s1, pool, prob, meta["threshold"], meta["variant"])
    elif variant == "d01_by_stage1_score":
        acc = best_claimant_keep(pool, s1, sc, acc)
    elif variant != "threshold_only":
        raise ValueError(variant)
    return np.sort(s1[acc] * SHIFT + pool[acc])


def counts(codes, gtc, n):
    s1 = codes // SHIFT
    return np.bincount(s1[np.isin(codes, gtc)], minlength=n), np.bincount(s1, minlength=n)


def metrics_for(codes, c: Ctx, pops: dict, weights: dict | None = None):
    tp, k = counts(codes, c.gtc, c.n)
    out = {}
    for name, m in pops.items():
        w = weights.get(name) if weights else None
        r = weighted_metrics(tp[m], k[m], c.n_true[m], None if w is None else w[m])
        by = {}
        for cc in ("india", "us"):
            mm = m & (c.ctry == cc)
            if mm.any():
                by[cc] = weighted_metrics(tp[mm], k[mm], c.n_true[mm], None if w is None else w[mm])["f05"]
        out[name] = r | {"by_country_f05": by}
    return out


# ----------------------------------------------------------------------------- views for pair similarity
def s1_views(paths, split, rows):
    want = pa.array(np.unique(rows).astype(np.int32))
    out = {}
    man = load_manifest(paths, split)
    for key in man["source1"]["countries"]:
        t = read_country(paths, split, "source1", key, ["row", "name_core", "addr_norm"])
        t = t.filter(pc.is_in(t.column("row"), value_set=want))
        out.update({r: (a or "", b or "") for r, a, b in zip(t.column("row").to_pylist(), t.column("name_core").to_pylist(), t.column("addr_norm").to_pylist())})
    return out


def pair_sims(paths, split, s1, pool, xmap):
    """(name token-set S1 name_core vs record name_xlit, address token-set) for given pairs."""
    sv = s1_views(paths, split, s1)
    pv = S.PoolViews(paths, split, xmap)
    pv.prepare(pool)
    P = pv(pool)
    a_n = [sv.get(int(r), ("", ""))[0] for r in s1]
    a_a = [sv.get(int(r), ("", ""))[1] for r in s1]
    nm = np.asarray(process.cpdist(a_n, P["name_xlit"], scorer=fuzz.token_set_ratio, workers=8), np.float32) / 100
    ad = np.asarray(process.cpdist(a_a, P["addr_norm"], scorer=fuzz.token_set_ratio, workers=8), np.float32) / 100
    ad[np.array([not x for x in P["addr_norm"]], bool)] = np.nan  # empty record address
    return nm, ad


def sim_category(nm, ad):
    cat = np.full(len(nm), "both_weak", dtype=object)
    cat[np.isnan(ad)] = "record_address_empty"
    ok = ~np.isnan(ad)
    cat[ok & (nm >= 0.8) & (ad >= 0.8)] = "both_strong"
    cat[ok & (nm >= 0.8) & (ad < 0.5)] = "name_heavy(strong name, weak address)"
    cat[ok & (nm < 0.5) & (ad >= 0.8)] = "address_heavy(weak name, strong address)"
    cat[ok & (cat == "both_weak") & ((nm >= 0.5) | (ad >= 0.5))] = "mixed_medium"
    return cat


def sample_idx(codes, k):
    """Deterministic sample of up to k indices (by a hash of the pair code)."""
    if len(codes) <= k:
        return np.arange(len(codes))
    h = (codes * np.int64(2654435761)) % np.int64(1000003)
    return np.sort(np.argsort(h, kind="stable")[:k])


# ----------------------------------------------------------------------------- verify
def cmd_verify(c: Ctx):
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        return h.hexdigest()

    def scan(path):
        rows, ids, empty, dup = 0, 0, 0, 0
        with open(path, "rb") as f:
            next(f)
            for line in f:
                a, _, b = line.rstrip(b"\n").partition(b"\t")
                L = [x for x in b.split(b",") if x]
                rows += 1
                ids += len(L)
                empty += not L
                dup += len(L) != len(set(L))
        return rows, ids, empty, dup

    files = {"output(submitted)": Path("output"), "EXP-S-01(validated)": Path("experiments/runs/EXP-S-01/output"),
             "EXP-B-06": Path("experiments/runs/EXP-B-06/output"), "EXP-B-07_t067": Path("experiments/runs/EXP-B-07/output_t067")}
    rep = {}
    for k, d in files.items():
        rep[k] = {}
        for f in ("candidate_pairs.tsv", "matching_results.tsv"):
            p = d / f
            if p.exists():
                r = scan(p)
                rep[k][f] = {"sha256": sha(p), "rows": r[0], "ids": r[1], "empty_rows": r[2], "rows_with_duplicate_ids": r[3]}
    rep["submitted_equals_validated_S01"] = all(rep["output(submitted)"][f]["sha256"] == rep["EXP-S-01(validated)"][f]["sha256"]
                                                 for f in ("candidate_pairs.tsv", "matching_results.tsv"))
    rep["S01_candidates_equal_B06"] = rep["EXP-S-01(validated)"]["candidate_pairs.tsv"]["sha256"] == rep["EXP-B-06"]["candidate_pairs.tsv"]["sha256"]
    write_json(OUT / "verify_submission.json", rep)
    LOG.info("verify: %s", {k: v for k, v in rep.items() if not isinstance(v, dict)})


# ----------------------------------------------------------------------------- delta (test, label-free) + train mirror (labels)
def decisions_on_eligible(split, c: Ctx, drop=None):
    """Eligible stage-2 pairs of a split with: stage-1 score, S-01 prob, and decisions of every system."""
    s1, pool, prob, sc, meta = stage2_arrays(CFG_S01, split)
    code = s1 * SHIFT + pool
    tag = STAGE1["B06"][0] if split == "train" else STAGE1["B06"][1]
    A = np.isin(code, stage1_codes(c.paths, split, tag, 0.67, True))
    dec = {"A_B06_D01": A, "C_S01_submitted": np.isin(code, stage2_codes(s1, pool, prob, sc, meta, "model")),
           "B_S01_no_D01": np.isin(code, stage2_codes(s1, pool, prob, sc, meta, "threshold_only")),
           "D_S01_D01_by_stage1_score": np.isin(code, stage2_codes(s1, pool, prob, sc, meta, "d01_by_stage1_score"))}
    s1b, poolb, probb, scb, metab = stage2_arrays(CFG_S01B, split)
    assert np.array_equal(s1b, s1) and np.array_equal(poolb, pool)  # same feature parts, same order
    dec["S01b_submitted_variant"] = np.isin(code, stage2_codes(s1b, poolb, probb, scb, metab, "model"))
    return s1, pool, prob, sc, code, dec, meta


def read_cols(split, cols, fdir=None):
    cfg = load_config(CFG_S01)
    paths = Paths(cfg)
    tabs = [pq.read_table(p, columns=cols) for p in sorted((fdir or S.feat_dir(cfg, paths, split)).glob("*.parquet"))]
    t = pa.concat_tables(tabs)
    return {k: t.column(k).to_numpy() for k in cols}


def cmd_delta(c: Ctx):
    rep = {"label": "test: UNLABELED diagnostics; train: E half B with labels (mirror)"}
    xmap = XlitMap.load(Path(c.cfg["stage2"]["xlit_map"]))
    for split in ("test", "train"):
        with PeakMemory() as pm:
            s1, pool, prob, sc, code, dec, meta = decisions_on_eligible(split, c)
        F = read_cols(split, FEATS_SUMMARY)
        ctry = c.ctry_te if split == "test" else c.ctry
        if split == "train":
            m = c.P["E_half_B"][s1]
            s1, pool, prob, sc, code = s1[m], pool[m], prob[m], sc[m], code[m]
            dec = {k: v[m] for k, v in dec.items()}
            F = {k: v[m] for k, v in F.items()}
            y = np.isin(code, c.gtc)
        pc_ = ctry[s1]
        R = {"eligible_pairs": int(len(s1)), "peak_gb": round(pm.peak_gb, 2)}
        groups = ["all"] + sorted(set(pc_.tolist()))
        # predictions per S1, acceptance, conflicts
        n_s1 = {g: int(((ctry == g) if g != "all" else np.ones(len(ctry), bool)).sum()) if split == "test" else
                int((c.P["E_half_B"] & ((c.ctry == g) if g != "all" else True)).sum()) for g in groups}
        R["predictions"] = {g: {k: {"pairs": int(v[(pc_ == g) if g != "all" else slice(None)].sum()),
                                    "per_s1": float(v[(pc_ == g) if g != "all" else slice(None)].sum() / max(n_s1[g], 1))}
                                for k, v in dec.items()} for g in groups}
        R["rise_C_vs_A"] = {g: R["predictions"][g]["C_S01_submitted"]["pairs"] / max(R["predictions"][g]["A_B06_D01"]["pairs"], 1) - 1 for g in groups}
        R["conflicts"] = {k: claim_stats(pool, s1, v)["conflicted_records"] for k, v in dec.items()}
        R["conflicts"]["S01_before_D01(p>=t)"] = claim_stats(pool, s1, prob >= np.float32(meta["threshold"]))["conflicted_records"]
        R["conflicts"]["B06_before_D01(score>=0.67)"] = claim_stats(pool, s1, sc >= np.float32(0.67))["conflicted_records"]
        fc = flip_codes(dec["A_B06_D01"], dec["C_S01_submitted"])
        R["flips"] = {g: flip_counts(fc[(pc_ == g) if g != "all" else slice(None)]) for g in groups}
        # acceptance rate by candidate density and claimant count
        nb = cand_bin(F["s1_n_cand"].astype(np.int64))
        cb = binned(F["rec_n_claim"], [2, 3, 5])
        R["acceptance_by_candidate_bin"] = {str(b): {k: float(v[nb == b].mean()) for k, v in (("A", dec["A_B06_D01"]), ("C", dec["C_S01_submitted"]))} | {"pairs": int((nb == b).sum())}
                                            for b in range(5) if (nb == b).any()}
        R["acceptance_by_claimants"] = {str(b): {k: float(v[cb == b].mean()) for k, v in (("A", dec["A_B06_D01"]), ("C", dec["C_S01_submitted"]))} | {"pairs": int((cb == b).sum())}
                                        for b in range(5) if (cb == b).any()}
        # profile of added / removed pairs
        prof = {}
        for fl, name in ((1, "added"), (2, "removed"), (3, "kept")):
            sel = fc == fl
            prof[name] = {"pairs": int(sel.sum()), "by_country": {g: int((sel & (pc_ == g)).sum()) for g in groups[1:]},
                          "prob": summary(prob[sel]), **{f: summary(F[f][sel]) for f in FEATS_SUMMARY}}
            if split == "train":
                prof[name]["true_share"] = float(y[sel].mean()) if sel.any() else None
                prof[name]["true_share_by_country"] = {g: float(y[sel & (pc_ == g)].mean()) for g in groups[1:] if (sel & (pc_ == g)).any()}
            idx = np.flatnonzero(sel)[sample_idx(code[sel], 200_000)]
            nm, ad = pair_sims(c.paths, split, s1[idx], pool[idx], xmap)
            cat = sim_category(nm, ad)
            u, cn = np.unique(cat, return_counts=True)
            prof[name]["pair_type_share(sample)"] = {k: float(v / len(cat)) for k, v in zip(u, cn)}
            if split == "train":
                prof[name]["true_share_by_pair_type(sample)"] = {k: float(y[idx][cat == k].mean()) for k in u}
        R["profiles"] = prof
        # cells for the expected-precision comparison
        cells = cell_ids(nb + 1, np.minimum(cb, 3), binned(sc, [0.2, 0.4, 0.67]), binned(F["peer_max_sim"], [0.5, 0.8, 0.95]),
                         binned(F["rec_gap_best_other"], [-0.2, 0.0, 0.2]))
        rep[split] = R
        if split == "test":
            test_add_cells, test_rem_cells = cells[fc == 1], cells[fc == 2]
        else:
            rep["expected_precision_of_test_additions_if_train_like"] = expected_precision(cells[fc == 1], y[fc == 1], test_add_cells)
            rep["expected_true_share_of_test_removals_if_train_like"] = expected_precision(cells[fc == 2], y[fc == 2], test_rem_cells)
        del s1, pool, prob, sc, code, dec, F
    write_json(OUT / "decision_flip_analysis.json", rep)
    LOG.info("rise C vs A: test %s | train E_B %s", {k: round(v, 4) for k, v in rep["test"]["rise_C_vs_A"].items()},
             {k: round(v, 4) for k, v in rep["train"]["rise_C_vs_A"].items()})
    LOG.info("expected precision of test additions (train-like): %s", rep["expected_precision_of_test_additions_if_train_like"])


# ----------------------------------------------------------------------------- shift
def cmd_shift(c: Ctx):
    names = feature_names(3)
    tr_s1 = read_cols("train", ["s1_row"])["s1_row"].astype(np.int64)
    te_s1 = read_cols("test", ["s1_row"])["s1_row"].astype(np.int64)
    mB, mV = c.P["E_half_B"][tr_s1], c.P["validation"][tr_s1]
    cte = c.ctry_te[te_s1]
    ctr = c.ctry[tr_s1]
    rep = {"method": "PSI of test vs E half B (quantile bins of E half B; NaN own bin). Noise reference: PSI(validation vs E half B)."}
    flagged = []
    for f in names:
        a = read_cols("train", [f])[f]
        b = read_cols("test", [f])[f]
        r = {"E_half_B": summary(a[mB]), "validation": summary(a[mV]), "test": summary(b)}
        for g in ("france", "india", "us"):
            r[f"test_{g}"] = summary(b[cte == g])
        r["psi_validation_vs_EB"] = psi(a[mB], a[mV])
        r["psi_test_vs_EB"] = psi(a[mB], b)
        r["psi_test_india_vs_EB_india"] = psi(a[mB & (ctr == "india")], b[cte == "india"])
        r["psi_test_us_vs_EB_us"] = psi(a[mB & (ctr == "us")], b[cte == "us"])
        r["psi_test_france_vs_EB"] = psi(a[mB], b[cte == "france"])
        r["flag"] = "major" if r["psi_test_vs_EB"] > 0.25 else "moderate" if r["psi_test_vs_EB"] > 0.1 else "stable"
        if r["flag"] != "stable":
            flagged.append((f, round(r["psi_test_vs_EB"], 3), r["flag"]))
        rep[f] = r
    rep["flagged"] = sorted(flagged, key=lambda x: -x[1])
    write_json(OUT / "test_vs_train_shift.json", rep)
    LOG.info("shift flagged: %s", rep["flagged"])


# ----------------------------------------------------------------------------- shadow benchmark
def orphan_frac(c: Ctx):
    mt, me = load_manifest(c.paths, "train"), load_manifest(c.paths, "test")
    tp = int(mt["source2"]["rows"]) + int(mt["source3"]["rows"])
    ep = int(me["source2"]["rows"]) + int(me["source3"]["rows"])
    return orphan_fraction(tp, int(mt["source1"]["rows"]), ep, int(me["source1"]["rows"])), tp, ep


def cmd_shadow_build(c: Ctx):
    sp = shadow_split(c.P["E_half_B"], c.ids, exclude=c.P["validation"])  # never the frozen validation fold
    shadow = sp["shadow_base"]
    frac, tp, ep = orphan_frac(c)
    orph = orphan_mask(c.ids, frac)
    # S1-level label-free properties from the stage-2 train features (base competition)
    F = read_cols("train", ["s1_row", "pool_row", "score", "s1_rank", "s1_n_cand", "s1_best", "s1_second", "rec_n_claim", "peer_max_sim"])
    s1 = F["s1_row"].astype(np.int64)
    ncand = np.bincount(s1, minlength=c.n)
    maxclaim = np.zeros(c.n)
    np.maximum.at(maxclaim, s1, F["rec_n_claim"])
    top = F["s1_rank"] == 0
    best_row = np.full(c.n, -1, np.int64)
    best_row[s1[top]] = np.flatnonzero(top)
    b = best_row[shadow]
    has = b >= 0
    xmap = XlitMap.load(Path(c.cfg["stage2"]["xlit_map"]))
    idx = b[has]
    nm, ad = pair_sims(c.paths, "train", s1[idx], F["pool_row"][idx].astype(np.int64), xmap)
    sh_rows = np.flatnonzero(shadow)
    flags = {k: np.zeros(c.n, bool) for k in ("dense_pool(>=8 eligible candidates)", "multiple_claimants(any candidate >=2)",
                                              "record_address_empty(best candidate)", "weak_name_strong_address(best)",
                                              "strong_name_weak_address(best)", "collective_ambiguity(best peer sim<0.5, >=3 candidates)",
                                              "low_margin(best-second<0.1, second>=0.5)", "generic_name(S1 name_core shared by >=20 train S1)")}
    flags["dense_pool(>=8 eligible candidates)"] = shadow & (ncand >= 8)
    flags["multiple_claimants(any candidate >=2)"] = shadow & (maxclaim >= 2)
    r = sh_rows[has]
    flags["record_address_empty(best candidate)"][r] = np.isnan(ad)
    flags["weak_name_strong_address(best)"][r] = (nm < 0.5) & (ad >= 0.8)
    flags["strong_name_weak_address(best)"][r] = (nm >= 0.8) & (ad < 0.5)
    bb = best_row[r]
    flags["collective_ambiguity(best peer sim<0.5, >=3 candidates)"][r] = (np.nan_to_num(F["peer_max_sim"][bb], nan=1.0) < 0.5) & (ncand[r] >= 3)
    flags["low_margin(best-second<0.1, second>=0.5)"][r] = (np.nan_to_num(F["s1_second"][bb], nan=0) >= 0.5) & ((F["s1_best"][bb] - np.nan_to_num(F["s1_second"][bb], nan=0)) < 0.1)
    # generic S1 names: name_core shared by >= 20 train S1
    names = np.empty(c.n, dtype=object)
    man = load_manifest(c.paths, "train")
    for k in man["source1"]["countries"]:
        t = read_country(c.paths, "train", "source1", k, ["row", "name_core"])
        names[t.column("row").to_numpy()] = t.column("name_core").to_pylist()
    vc = pc.value_counts(pa.array(names.tolist(), pa.string()))
    cnt = dict(zip(vc.field("values").to_pylist(), vc.field("counts").to_pylist()))
    flags["generic_name(S1 name_core shared by >=20 train S1)"] = shadow & np.array([cnt.get(x, 0) >= 20 for x in names], bool)
    chain_member = np.array([cnt.get(x, 0) >= 2 for x in names], bool)
    chain_orph = chain_orphan_mask(c.ids, names, protect=shadow, frac=0.5)
    hard = np.zeros(c.n, bool)
    for v in flags.values():
        hard |= v
    SHADOW_DIR.mkdir(parents=True, exist_ok=True)
    ids = np.asarray(c.ids, dtype=object)
    for nm_, m in (("shadow_base_s1_ids.txt", shadow), ("shadow_hard_s1_ids.txt", hard), ("orphaned_s1_ids.txt", orph)):
        (SHADOW_DIR / nm_).write_text("\n".join(sorted(ids[m].tolist())) + "\n", encoding="utf-8")
    np.savez_compressed(SHADOW_DIR / "masks.npz", shadow=shadow, gate_B=sp["gate_B"], orphan=orph, hard=hard, chain_orphan=chain_orph,
                        chain_member=chain_member,
                        **{f"flag_{i}": v for i, v in enumerate(flags.values())})

    def dig(m):
        return hashlib.sha256("\n".join(sorted(ids[m].tolist())).encode()).hexdigest()

    man = {"seed": 42, "definitions": sys.modules["ber.forensics.shadow"].__doc__,
           "SHADOW-BASE": {"s1": int(shadow.sum()), "sha256_ids": dig(shadow), "file": str(SHADOW_DIR / "shadow_base_s1_ids.txt"),
                           "by_country": {g: int((shadow & (c.ctry == g)).sum()) for g in ("india", "us")}},
           "GATE-B (E half B minus shadow; ordinary gate set from now on)": {"s1": int(sp["gate_B"].sum())},
           "ORPHANED competition": {"orphan_fraction": frac, "train_pool": tp, "test_pool": ep, "orphaned_s1": int(orph.sum()),
                                    "sha256_ids": dig(orph), "shadow_s1_not_orphaned": int((shadow & ~orph).sum())},
           "SHADOW-TEST-MATCHED": "shadow S1 not orphaned, orphaned competition, weights by (country, eligible-candidate bin) of test India/US",
           "CHAIN-ORPHANED competition": {"definition": "50% of NON-shadow chain S1 (name_core shared by >= 2 train S1) removed; their records stay as "
                                                        "uncontested look-alike records for kept siblings", "chain_member_s1": int(chain_member.sum()),
                                          "orphaned_s1": int(chain_orph.sum()), "shadow_chain_members": int((shadow & chain_member).sum()),
                                          "sha256_ids": dig(chain_orph)},
           "SHADOW-HARD": {"s1": int(hard.sum()), "sha256_ids": dig(hard), "categories": {k: int(v.sum()) for k, v in flags.items()}},
           "never_used_for": ["stage-1 training", "stage-2 training", "transliteration learning", "threshold selection", "early stopping",
                              "model selection (from 2026-09-27 on)"],
           "caveat": "E half B (which contains SHADOW-BASE) was used as an aggregate gate for EXP-B-07/B-06/S-01 decisions before this benchmark existed."}
    write_json(OUT / "shadow_benchmark_manifest.json", man)
    write_json(SHADOW_DIR / "manifest.json", man)
    LOG.info("shadow: base %d, hard %d, orphan frac %.4f (%d S1)", shadow.sum(), hard.sum(), frac, orph.sum())


def orphan_fdir(c: Ctx):
    return c.paths.processed_dir("train") / "stage2_EXP-S-01_orphan"


def chain_fdir(c: Ctx):
    return c.paths.processed_dir("train") / "stage2_EXP-S-01_chainorphan"


def cmd_shadow_features(c: Ctx):
    M = np.load(SHADOW_DIR / "masks.npz")
    for mask, fdir in ((M["orphan"], orphan_fdir(c)), (M["chain_orphan"], chain_fdir(c))):
        with PeakMemory() as pm:
            st = S.build_features(c.cfg, c.paths, "train", drop_s1=mask, out=fdir)
        LOG.info("%s: %d pairs, dropped %d S1, %.0fs, peak %.2f GB", fdir.name, st["eligible_pairs"], st["dropped_s1"], pm.seconds, pm.peak_gb)


def lookalike_density(fdir, split, m_s1, n_s1):
    """Label-free: weak (stage-1 < 0.5), uncontested, peer-similar (>= 0.8) candidates per S1 of the population."""
    d = read_cols(split, ["s1_row", "score", "rec_n_claim", "peer_max_sim"], fdir)
    s1 = d["s1_row"].astype(np.int64)
    g = (d["score"] < 0.5) & (d["rec_n_claim"] <= 1) & (d["peer_max_sim"] >= 0.8) & m_s1[s1]
    return float(g.sum() / max(n_s1, 1))


def test_cell_shares(c: Ctx):
    s1 = read_cols("test", ["s1_row"])["s1_row"].astype(np.int64)
    ncand = np.bincount(s1, minlength=len(c.ctry_te))
    cells = {}
    for g, gi in (("india", 1), ("us", 2)):
        m = c.ctry_te == g
        for b, cnt in zip(*np.unique(cand_bin(ncand[m]) + 1, return_counts=True)):
            cells[gi * 8 + int(b)] = cnt
    tot = sum(cells.values())
    return {k: v / tot for k, v in cells.items()}


def cmd_shadow_eval(c: Ctx):
    M = np.load(SHADOW_DIR / "masks.npz")
    shadow, orph, hard, gate = M["shadow"], M["orphan"], M["hard"], M["gate_B"]
    s1_tr = read_cols("train", ["s1_row"])["s1_row"].astype(np.int64)
    ncand = np.bincount(s1_tr, minlength=c.n)
    gi = np.where(c.ctry == "india", 1, np.where(c.ctry == "us", 2, 0))
    cells = gi * 8 + (cand_bin(ncand) + 1)
    base_pops = {"E": c.P["E"], "validation": c.P["validation"], "E_half_B": c.P["E_half_B"], "GATE-B": gate, "SHADOW-BASE": shadow,
                 "SHADOW-HARD": hard}
    tm = shadow & ~orph
    w_tm = np.zeros(c.n)
    w_tm[tm] = match_weights(cells[tm], test_cell_shares(c))
    orph_pops = {"SHADOW-TEST-MATCHED": tm, "SHADOW-ORPHANED-unweighted": tm}
    chain_pops = {"SHADOW-CHAIN-ORPHANED": shadow, "SHADOW-CHAIN-ORPHANED:chain_members": shadow & M["chain_member"]}
    weights = {"SHADOW-TEST-MATCHED": w_tm}
    hard_cats = {k: M[k] for k in M.files if k.startswith("flag_")}
    man = json.loads((OUT / "shadow_benchmark_manifest.json").read_text(encoding="utf-8"))
    cat_names = list(man["SHADOW-HARD"]["categories"])
    rep = {"populations": {k: int(v.sum()) for k, v in (base_pops | orph_pops).items()}, "systems": {}}
    for sysname, (tag, _, t, d01) in STAGE1.items():
        with PeakMemory() as pm:
            base = metrics_for(stage1_codes(c.paths, "train", tag, t, d01), c, base_pops | {f"HARD:{cat_names[i]}": hard_cats[f"flag_{i}"] for i in range(len(cat_names))})
            orp = metrics_for(stage1_codes(c.paths, "train", tag, t, d01, drop=orph), c, orph_pops, weights)
            chn = metrics_for(stage1_codes(c.paths, "train", tag, t, d01, drop=M["chain_orphan"]), c, chain_pops)
        rep["systems"][sysname] = base | orp | chn
        LOG.info("%-10s SHADOW-BASE %.4f | TEST-MATCHED %.4f | %.0fs", sysname, base["SHADOW-BASE"]["f05"], orp["SHADOW-TEST-MATCHED"]["f05"], pm.seconds)
    for cfgp, label in ((CFG_S01, "S01"), (CFG_S01B, "S01b")):
        for fdir, pops, wts in ((None, base_pops | {f"HARD:{cat_names[i]}": hard_cats[f"flag_{i}"] for i in range(len(cat_names))}, None),
                                (orphan_fdir(c), orph_pops, weights), (chain_fdir(c), chain_pops, None)):
            s1, pool, prob, sc, meta = stage2_arrays(cfgp, "train", fdir)
            variants = ("model", "threshold_only", "d01_by_stage1_score") if label == "S01" else ("model",)
            for v in variants:
                key = label if v == "model" else f"{label}[{v}]"
                rep["systems"].setdefault(key, {}).update(metrics_for(stage2_codes(s1, pool, prob, sc, meta, v), c, pops, wts))
            del s1, pool, prob, sc
        LOG.info("%-10s SHADOW-BASE %.4f | TEST-MATCHED %.4f | CHAIN-ORPHANED %.4f", label, rep["systems"][label]["SHADOW-BASE"]["f05"],
                 rep["systems"][label]["SHADOW-TEST-MATCHED"]["f05"], rep["systems"][label]["SHADOW-CHAIN-ORPHANED"]["f05"])
    te_s1 = read_cols("test", ["s1_row"])["s1_row"]
    rep["lookalike_density_per_s1(weak, uncontested, peer_sim>=0.8)"] = {
        "test": lookalike_density(None, "test", np.ones(len(c.ctry_te), bool), len(c.ctry_te)),
        "SHADOW-BASE (normal competition)": lookalike_density(None, "train", shadow, shadow.sum()),
        "random-orphaned competition (shadow not orphaned)": lookalike_density(orphan_fdir(c), "train", tm, tm.sum()),
        "chain-orphaned competition (shadow)": lookalike_density(chain_fdir(c), "train", shadow, shadow.sum()),
        "chain-orphaned competition (shadow chain members)": lookalike_density(chain_fdir(c), "train", shadow & M["chain_member"], (shadow & M["chain_member"]).sum())}
    LOG.info("look-alike density: %s", {k: round(v, 3) for k, v in rep["lookalike_density_per_s1(weak, uncontested, peer_sim>=0.8)"].items()})
    # calibration table
    cal = []
    for sysname, pub in (("B04", PUBLIC["B04"]), ("N02", PUBLIC["N02"]), ("B07_t067", None), ("B06", None), ("S01", PUBLIC["S01"]), ("S01b", None)):
        s = rep["systems"][sysname]
        cal.append({"system": sysname, "E": s["E"]["f05"], "validation": s["validation"]["f05"], "shadow_base": s["SHADOW-BASE"]["f05"],
                    "shadow_test_matched": s["SHADOW-TEST-MATCHED"]["f05"], "shadow_chain_orphaned": s["SHADOW-CHAIN-ORPHANED"]["f05"],
                    "shadow_chain_members": s["SHADOW-CHAIN-ORPHANED:chain_members"]["f05"], "shadow_hard": s["SHADOW-HARD"]["f05"], "public": pub,
                    "gap_public_minus_E": (pub - s["E"]["f05"]) if pub else None,
                    "gap_public_minus_test_matched": (pub - s["SHADOW-TEST-MATCHED"]["f05"]) if pub else None})
    rep["calibration"] = cal
    rep["note_public"] = "0.935 = best pre-S-01 public file (B-07 t0.67 or B-06, not attributed in any artifact)."
    write_json(OUT / "shadow_metrics.json", rep)
    for r in cal:
        LOG.info("CAL %-9s E %.4f val %.4f base %.4f test-matched %.4f chain %.4f chain-members %.4f public %s", r["system"], r["E"], r["validation"],
                 r["shadow_base"], r["shadow_test_matched"], r["shadow_chain_orphaned"], r["shadow_chain_members"], r["public"])


# ----------------------------------------------------------------------------- leakage
def cmd_leakage(c: Ctx):
    M = np.load(SHADOW_DIR / "masks.npz")
    shadow = M["shadow"]
    xm = XlitMap.load(Path(c.cfg["stage2"]["xlit_map"]))
    fit_ids = [i for i, f in zip(c.ids, c.P["fit"]) if f]
    s2m = json.loads((c.paths.run_dir("EXP-S-01") / "stage2_model.json").read_text(encoding="utf-8"))
    rep = leakage_checks(shadow, stage1_fit=c.P["fit"], stage2_train=c.P["stage2_train"], stage2_inner_threshold_and_early_stopping=c.P["stage2_inner"],
                         validation=c.P["validation"])
    rep["xlit_map_learned_from_fit_set_only"] = xm.meta.get("learned_from_digest") == XlitMap.ids_digest(fit_ids)
    rep["stage2_train_s1_count_matches_model_meta"] = int(c.P["stage2_train"].sum()) == s2m["train_s1"]
    rep["stage2_inner_s1_count_matches_model_meta"] = int(c.P["stage2_inner"].sum()) == s2m["inner_s1"]
    rep["shadow_subset_of_E_half_B"] = bool(not (shadow & ~c.P["E_half_B"]).any())
    rep["all_zero"] = all(v == 0 for k, v in rep.items() if k.startswith("shadow_in_"))
    write_json(OUT / "leakage_test_results.json", rep)
    LOG.info("leakage: %s", rep)


FLOORS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 9.0]  # 9.0 = stage 2 may only remove pairs


def cmd_floor(c: Ctx):
    """DIAGNOSTIC: S-01 with additions restricted to stage-1 score >= floor (no retraining, no production change)."""
    M = np.load(SHADOW_DIR / "masks.npz")
    shadow = M["shadow"]
    rep = {"floors": FLOORS, "note": "floor 0.0 = S-01 as submitted; 9.0 = removals only", "scenarios": {}}
    scen = {"base": (None, "train", None, {"SHADOW-BASE": shadow, "validation": c.P["validation"], "E_half_B": c.P["E_half_B"]}),
            "chain_orphaned": (chain_fdir(c), "train", M["chain_orphan"], {"SHADOW-CHAIN-ORPHANED": shadow,
                                                                          "chain_members": shadow & M["chain_member"]}),
            "test": (None, "test", None, None)}
    for name, (fdir, split, drop, pops) in scen.items():
        s1, pool, prob, sc, meta = stage2_arrays(CFG_S01, split, fdir)
        code = s1 * SHIFT + pool
        s2 = np.isin(code, stage2_codes(s1, pool, prob, sc, meta, "model"))
        tag = STAGE1["B06"][0] if split == "train" else STAGE1["B06"][1]
        ctl = np.isin(code, stage1_codes(c.paths, split, tag, 0.67, True, drop=drop))
        r = {}
        if pops:
            r["B06_D01"] = {k: v for k, v in metrics_for(code[ctl], c, pops).items()}
        n_s1 = len(c.ctry_te) if split == "test" else int(c.P["E_half_B"].sum())
        m_add = np.ones(len(s1), bool) if split == "test" else c.P["E_half_B"][s1]
        for f in FLOORS:
            d = floor_decision(ctl, s2, sc, f)
            row = {"additions_per_s1": float((d & ~ctl & m_add).sum() / n_s1), "removals_per_s1": float((~d & ctl & m_add).sum() / n_s1),
                   "prediction_rise_vs_B06": float((d & m_add).sum() / max((ctl & m_add).sum(), 1) - 1)}
            if pops:
                row |= metrics_for(code[d], c, pops)
            r[f"floor_{f}"] = row
        rep["scenarios"][name] = r
        first = next(iter(pops)) if pops else None
        LOG.info("%s: %s", name, {k: (round(v["additions_per_s1"], 3), round(v[first]["f05"], 4) if first else None)
                                  for k, v in r.items() if k.startswith("floor")})
        write_json(OUT / "floor_diagnostic.json", rep)  # saved after every scenario
        del s1, pool, prob, sc, code
    write_json(OUT / "floor_diagnostic.json", rep)


CMDS = {"verify": cmd_verify, "delta": cmd_delta, "shift": cmd_shift, "shadow-build": cmd_shadow_build,
        "shadow-features": cmd_shadow_features, "shadow-eval": cmd_shadow_eval, "leakage": cmd_leakage, "floor": cmd_floor}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=list(CMDS))
    a = ap.parse_args()
    setup_logging()
    configure_arrow()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    CMDS[a.cmd](Ctx())
    LOG.info("%s done in %.0fs", a.cmd, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
