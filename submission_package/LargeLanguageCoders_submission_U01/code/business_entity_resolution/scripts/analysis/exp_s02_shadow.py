"""EXP-S-02 (veto-only stage 2) + shadow benchmark v2 (density tiers, locked sets). Deterministic, seed 42.

    python scripts/analysis/exp_s02_shadow.py snapshot | smoke | tiers-build | tiers-features | eval | testdiag | budget | leakage | gates

snapshot        hashes of every artifact EXP-S-02 must not change (S-01 model / features, outputs) + resolved config copy
smoke           real-data invariants of the veto on ONE feature part per split (before any full run)
tiers-build     SHADOW-DEV / LOCKED-A / LOCKED-B + density tiers T1..T4 (label-free construction, frozen rules in notes.md)
tiers-features  stage-2 feature parts of the T2..T4 competitions (S-01 feature code, read-only model)
eval            B-06 (control) vs S-01 vs S-02 on every population and competition (labels of held-out train S1 only)
testdiag        label-free test behaviour: S-02 / S-01 output files vs the B-06 output file
budget          post-S-02 error budget (oracle F0.5 gain per error bucket) on E half B and SHADOW-DEV
leakage         partition overlaps + read-only proof (hashes vs snapshot)
gates           applies the pre-declared gates (notes.md) -> gates.json, metrics.json

Read-only w.r.t. output/, the S-01 model and feature parts, and all historical artifacts (v1 shadow files untouched).
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
import pyarrow.parquet as pq
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import post_s01_forensics as F  # noqa: E402  (shared, already validated forensic helpers)

from ber.config import Paths, load_config  # noqa: E402
from ber.decision.competition import claim_stats  # noqa: E402
from ber.decision.stage2 import restrict_additions  # noqa: E402
from ber.forensics.shadow import (cand_bin, leakage_checks, locked_split, lookalike_pairs, match_weights, tier_orphan_mask,  # noqa: E402
                                  u01, weighted_metrics)
from ber.pipeline import run_stage2 as S  # noqa: E402
from ber.pipeline.run_baseline import _ids, _read_parts  # noqa: E402
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, write_json  # noqa: E402

RUN = Path("experiments/runs/EXP-S-02")
OUT = RUN / "analysis"
V2 = Path("experiments/shadow_tests/v2")
V1 = Path("experiments/shadow_tests")
CFG_S02 = "configs/exp_s02_stage2_veto.yaml"
SHIFT = S.SHIFT
B06_TRAIN, B06_TEST, T_B06 = F.STAGE1["B06"][0], F.STAGE1["B06"][1], 0.67
Q_GRID = np.round(np.arange(0.0, 2.0 + 1e-9, 0.02), 2)
TIERS = ("T1", "T2", "T3", "T4")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_ids(ids: np.ndarray, m: np.ndarray) -> str:
    return hashlib.sha256("\n".join(sorted(ids[m].tolist())).encode()).hexdigest()


def tier_fdir(c, k: str) -> Path | None:
    return None if k == "T1" else c.paths.processed_dir("train") / f"stage2_EXP-S-01_v2tier{k[1]}"


def avg(x, w=None) -> float:
    x = np.asarray(x, np.float64)
    if len(x) == 0:
        return float("nan")
    return float(x.mean()) if w is None else float((w * x).sum() / w.sum())


# ----------------------------------------------------------------------------- snapshot
def readonly_state(c) -> dict:
    s01 = Path("experiments/runs/EXP-S-01")
    rep = {"s01_model": {f: sha_file(s01 / f) for f in ("stage2_model.lgb.txt", "stage2_model.json")}}
    for split in ("train", "test"):
        fd = S.feat_dir(c.cfg, c.paths, split)
        h = hashlib.sha256()
        for p in sorted(fd.glob("*.parquet")):
            h.update(p.name.encode())
            h.update(sha_file(p).encode())
        rep[f"s01_features_{split}"] = h.hexdigest()
    for name, d in (("root_output", Path("output")), ("s01_output", s01 / "output"), ("b06_output", Path("experiments/runs/EXP-B-06/output"))):
        rep[name] = {f: sha_file(d / f) for f in ("matching_results.tsv", "candidate_pairs.tsv")}
    if (V1 / "masks.npz").exists():
        rep["shadow_v1"] = {f: sha_file(V1 / f) for f in ("manifest.json", "masks.npz")}
    return rep


def cmd_snapshot(c):
    write_json(OUT / "readonly_snapshot_before.json", readonly_state(c))
    (RUN / "config.yaml").write_text("# resolved configs/exp_s02_stage2_veto.yaml (inherit chain expanded)\n"
                                     + yaml.safe_dump(load_config(CFG_S02), sort_keys=False), encoding="utf-8")
    LOG.info("snapshot written")


# ----------------------------------------------------------------------------- smoke
def cmd_smoke(c):
    cfg = load_config(CFG_S02)
    paths = Paths(cfg)
    booster, meta = S.load_stage2(cfg, paths)
    rep = {}
    for split in ("test", "train"):
        with PeakMemory() as pm:
            ctl = S.control_decisions(cfg, paths, split)[0]
            part = sorted(S.feat_dir(cfg, paths, split).glob("*.parquet"))[0]
            t = pq.read_table(part)
            X = np.column_stack([t.column(f).to_numpy() for f in meta["features"]]).astype(np.float32)
            prob = booster.predict(X, num_threads=8).astype(np.float32)
            s1, pool = t.column("s1_row").to_numpy().astype(np.int64), t.column("pool_row").to_numpy().astype(np.int64)
            sc = t.column("score").to_numpy().astype(np.float32)
            ctl_part = ctl[np.isin(ctl // SHIFT, np.unique(s1))]
            s01 = S.decide(s1, pool, prob, meta["threshold"], meta["variant"])  # D-01 inside this part only (smoke)
            s02 = S.final_decisions(cfg, s1, pool, prob, sc, meta, ctl_part)
            cm = np.isin(s1 * SHIFT + pool, ctl_part)
        r = {"part": part.name, "s1": int(len(np.unique(s1))), "pairs": int(len(s1)), "control_pairs": int(cm.sum()), "s01_pairs": int(s01.sum()),
             "s02_pairs": int(s02.sum()), "additions": int((s02 & ~cm).sum()), "removals": int((cm & ~s02).sum()),
             "veto_subset_of_control": bool(not (s02 & ~cm).any()), "veto_subset_of_s01": bool(not (s02 & ~s01).any()),
             "veto_equals_intersection": bool(np.array_equal(s02, cm & s01)), "conflicts_s02": claim_stats(pool, s1, s02)["conflicted_records"],
             "peak_gb": round(pm.peak_gb, 2), "seconds": round(pm.seconds, 1)}
        if split == "train":
            y = np.isin(s1 * SHIFT + pool, c.gtc)
            m = c.P["E_half_B"]
            for name, acc in (("control", cm), ("s01", s01), ("s02", s02)):
                tp, k = np.bincount(s1[acc & y], minlength=c.n), np.bincount(s1[acc], minlength=c.n)
                mm = m & (np.bincount(s1, minlength=c.n) > 0)
                r[f"f05_{name}_part_E_half_B"] = weighted_metrics(tp[mm], k[mm], c.n_true[mm])["f05"]
        ok = r["veto_subset_of_control"] and r["veto_subset_of_s01"] and r["veto_equals_intersection"] and r["additions"] == 0 and r["conflicts_s02"] == 0
        r["pass"] = bool(ok)
        rep[split] = r
        LOG.info("smoke %s: %s", split, r)
    rep["pass"] = all(v["pass"] for v in rep.values() if isinstance(v, dict))
    write_json(OUT / "smoke.json", rep)
    if not rep["pass"]:
        raise SystemExit("smoke FAILED")


# ----------------------------------------------------------------------------- tiers
def base_arrays():
    d = F.read_cols("train", ["s1_row", "pool_row", "score", "peer_max_sim"])
    return d["s1_row"].astype(np.int64), d["pool_row"].astype(np.int64), d["score"].astype(np.float32), d["peer_max_sim"].astype(np.float32)


def stage1_ncand(c) -> np.ndarray:
    s1 = _read_parts(c.paths.scored_dir("train", B06_TRAIN), ["s1_row"])["s1_row"].astype(np.int64)
    return np.bincount(s1, minlength=c.n)


def masks_v2() -> dict:
    return dict(np.load(V2 / "masks_v2.npz"))


def tier_drop(M, k: str) -> np.ndarray | None:
    return None if k == "T1" else M[f"orphan_{k}"]


def competition_stats(c, s1, pool, score, peer, drop, pops, ncand, chain_member, flags, ids) -> dict:
    keep = np.ones(len(s1), bool) if drop is None else ~drop[s1]
    claim = np.bincount(pool[keep], minlength=int(pool.max()) + 1)
    la = lookalike_pairs(s1, pool, score, peer, drop)
    la_s1 = np.bincount(s1[la], minlength=c.n)
    multi = np.zeros(c.n, bool)
    multi[s1[keep & (claim[pool] >= 2)]] = True
    dropped = np.zeros(c.n, bool) if drop is None else drop
    comp = {"orphaned_s1": int(dropped.sum()), "orphaned_chain_members": int((dropped & chain_member).sum()),
            "orphaned_by_country": {g: int((dropped & (c.ctry == g)).sum()) for g in ("india", "us")},
            "competing_s1": int(c.n - dropped.sum()), "pool_records_per_competing_s1": float(c.train_pool / max(c.n - dropped.sum(), 1)),
            "sha256_orphaned_ids": sha_ids(ids, dropped)}
    per = {}
    for name, m in pops.items():
        sel = m[s1] & keep
        n = int(m.sum())
        per[name] = {"s1": n, "lookalike_density": float(la_s1[m].sum() / max(n, 1)), "stage1_candidates_per_s1": avg(ncand[m]),
                     "eligible_per_s1": float(sel.sum() / max(n, 1)), "mean_claimants_per_eligible_pair": avg(claim[pool[sel]]),
                     "contested_pair_share": avg(claim[pool[sel]] >= 2), "s1_with_contested_candidate_share": avg(multi[m]),
                     "country_share": {g: avg(c.ctry[m] == g) for g in ("india", "us")},
                     "hard_case_share(v1 flags)": {k: avg(v[m]) for k, v in flags.items()}, "sha256_ids": sha_ids(ids, m)}
    return {"competition": comp, "populations": per}


def cmd_tiers_build(c):
    M1 = np.load(V1 / "masks.npz")
    shadow, chain_member = M1["shadow"], M1["chain_member"]
    man1 = json.loads((V1 / "manifest.json").read_text(encoding="utf-8"))
    flags = {name: M1[f"flag_{i}"] for i, name in enumerate(man1["SHADOW-HARD"]["categories"])}
    ids = np.asarray(c.ids, dtype=object)
    sp = locked_split(shadow, c.ids)
    dev, la_, lb_ = sp["dev"], sp["locked_A"], sp["locked_B"]
    for name, m in (("fit", c.P["fit"]), ("stage2_train", c.P["stage2_train"]), ("stage2_inner", c.P["stage2_inner"]), ("validation", c.P["validation"])):
        for pn, pm_ in (("dev", dev), ("locked_A", la_), ("locked_B", lb_)):
            if (pm_ & m).any():
                raise AssertionError(f"{pn} overlaps {name}")
    s1, pool, score, peer = base_arrays()
    # (1) the fast label-free density must reproduce the v1 feature-based numbers exactly
    v1m = json.loads(Path("experiments/runs/POST-S01-FORENSIC/shadow_metrics.json").read_text(encoding="utf-8"))
    v1d = v1m["lookalike_density_per_s1(weak, uncontested, peer_sim>=0.8)"]

    def dens(drop, m):
        la = lookalike_pairs(s1, pool, score, peer, drop)
        return float((la & m[s1]).sum() / m.sum())

    check = {"base_shadow": (dens(None, shadow), v1d["SHADOW-BASE (normal competition)"]),
             "chain_orphan_shadow": (dens(M1["chain_orphan"], shadow), v1d["chain-orphaned competition (shadow)"]),
             "chain_orphan_chain_members": (dens(M1["chain_orphan"], shadow & chain_member), v1d["chain-orphaned competition (shadow chain members)"])}
    for k, (a, b) in check.items():
        if abs(a - b) > 1e-12:
            raise AssertionError(f"fast look-alike density {k}: {a} != v1 feature-based {b}")
    test_density = float(v1d["test"])
    # (2) density curve on SHADOW-DEV over the nested family (label-free)
    u = u01(c.ids, "tier_orphan", 42)
    curve = []
    for q in Q_GRID:
        drop = (chain_member & ~shadow & (u < min(q, 1.0))) | (~chain_member & ~shadow & (u < max(q - 1.0, 0.0)))
        curve.append((float(q), dens(drop, dev), int(drop.sum())))
    base, top = curve[0][1], curve[-1][1]
    # amendment A1 (notes.md, 2026-09-27 04:40): orphaning cannot reach the test density (max = q 2) -> T2/T3 re-spaced
    # inside the reachable range; T4 = q 2 (maximum), flagged when it stays below the test density
    reach = min(test_density, top)
    targets = {"T1": None, "T2": base + (reach - base) / 3, "T3": base + 2 * (reach - base) / 3, "T4": test_density}
    qsel, reached = {"T1": 0.0}, {"T1": True}
    for k in ("T2", "T3", "T4"):
        hit = [q for q, d, _ in curve if d >= targets[k] - 1e-12]
        qsel[k], reached[k] = (hit[0], True) if hit else (2.0, False)
    masks = {"dev": dev, "locked_A": la_, "locked_B": lb_}
    for k in ("T2", "T3", "T4"):
        m = tier_orphan_mask(c.ids, chain_member, shadow, qsel[k])
        q = qsel[k]
        ref = (chain_member & ~shadow & (u < min(q, 1.0))) | (~chain_member & ~shadow & (u < max(q - 1.0, 0.0)))
        if not np.array_equal(m, ref) or (m & shadow).any():
            raise AssertionError(f"tier mask {k} inconsistent / touches shadow")
        masks[f"orphan_{k}"] = m
    ncand = stage1_ncand(c)
    pops = {"SHADOW-DEV": dev, "SHADOW-LOCKED-A": la_, "SHADOW-LOCKED-B": lb_}
    tiers = {}
    for k in TIERS:
        st = competition_stats(c, s1, pool, score, peer, masks.get(f"orphan_{k}"), pops, ncand, chain_member, flags, ids)
        st |= {"q": qsel[k], "target_density_dev": targets[k], "target_reached": reached[k]}
        tiers[k] = st
        LOG.info("%s q=%.2f target %s dev density %.4f (orphaned %d)", k, qsel[k], targets[k], st["populations"]["SHADOW-DEV"]["lookalike_density"],
                 st["competition"]["orphaned_s1"])
    # test reference (label-free), from the test stage-2 features
    te = F.read_cols("test", ["s1_row", "pool_row", "score", "rec_n_claim", "peer_max_sim"])
    n_te = len(c.ctry_te)
    te_ncand = np.bincount(_read_parts(c.paths.scored_dir("test", B06_TEST), ["s1_row"])["s1_row"].astype(np.int64), minlength=n_te)
    test_ref = {"s1": n_te, "lookalike_density": test_density, "stage1_candidates_per_s1": avg(te_ncand), "eligible_per_s1": len(te["s1_row"]) / n_te,
                "mean_claimants_per_eligible_pair": avg(te["rec_n_claim"]), "contested_pair_share": avg(te["rec_n_claim"] >= 2),
                "pool_records_per_s1": c.test_pool / n_te, "country_share": {g: avg(c.ctry_te == g) for g in ("france", "india", "us")}}
    V2.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(V2 / "masks_v2.npz", **masks)
    for name, m in (("shadow_dev_s1_ids.txt", dev), ("shadow_locked_A_s1_ids.txt", la_), ("shadow_locked_B_s1_ids.txt", lb_)):
        (V2 / name).write_text("\n".join(sorted(ids[m].tolist())) + "\n", encoding="utf-8")
    man = {"version": 2, "seed": 42, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "built_from": {"shadow_base_v1_sha256": man1["SHADOW-BASE"]["sha256_ids"], "v1_files_untouched": True},
           "definitions": {"SHADOW-DEV": "SHADOW-BASE & u01(id,'locked',42) < 0.5 (may be used while iterating)",
                           "SHADOW-LOCKED-A": "SHADOW-BASE & 0.5 <= u < 0.75 (evaluate only after an experiment is frozen)",
                           "SHADOW-LOCKED-B": "SHADOW-BASE & u >= 0.75 (evaluate only after an experiment is frozen)",
                           "tier competitions": "tier_orphan_mask(q): chain members (name_core shared by >= 2 train S1) with u01(id,'tier_orphan',42) < q "
                                                "removed first, other S1 with u < q-1 when q > 1; SHADOW-BASE S1 never removed; removed S1 keep their "
                                                "records in the pool, unclaimed. T1 = base competition.",
                           "look-alike density": "per evaluated S1: eligible candidates with stage-1 score < 0.5, <= 1 claimant among competing "
                                                 "S1 (eligible pairs), peer_max_sim >= 0.8 (label-free)",
                           "tier rule (notes.md 1.2 + amendment A1)": "T2/T3 target = base + 1/3, 2/3 of (min(test, max reachable) - base) on "
                                                                      "SHADOW-DEV, smallest q on a 0.02 grid reaching it; T4 = test density, else q = 2 (flagged)",
                           "weighted tiers (amendment A2)": "T1W / T4W: T1 / T4 competition, evaluated S1 re-weighted so their per-S1 look-alike count "
                                                            "distribution (bins 0..4, 5+) equals test's (label-free importance weights, mean 1)"},
           "sizes": {k: int(v.sum()) for k, v in pops.items()}, "sha256_ids": {k: sha_ids(ids, v) for k, v in pops.items()},
           "fast_density_matches_v1_features": {k: {"fast": a, "v1": b} for k, (a, b) in check.items()},
           "density_curve_dev(q, density, orphaned_s1)": curve, "tiers": tiers, "test_reference(label-free)": test_ref,
           "never_used_for": ["stage-1 training", "stage-2 training", "transliteration learning", "threshold selection", "early stopping",
                              "feature engineering", "tier construction (label-free only)"],
           "caveats": ["SHADOW-BASE (hence DEV and both LOCKED sets) lies inside E half B, used as an aggregate gate for B-07/B-06/S-01.",
                       "The EXP-S-02 veto was chosen (POST-S01-FORENSIC) from aggregate SHADOW-BASE metrics: the LOCKED sets are not fresh "
                       "for EXP-S-02; they are fresh for experiments frozen after 2026-09-27.",
                       "Orphaning can only make existing candidates uncontested; it cannot add candidates to an S1's list."]}
    write_json(V2 / "manifest_v2.json", man)
    write_json(OUT / "shadow_v2_manifest.json", man)
    LOG.info("v2: dev %d, locked A %d, B %d | q %s | reached %s", dev.sum(), la_.sum(), lb_.sum(), qsel, reached)


def cmd_tiers_features(c):
    M = masks_v2()
    rep = {}
    for k in ("T2", "T3", "T4"):
        fd = tier_fdir(c, k)
        with PeakMemory() as pm:
            st = S.build_features(c.cfg, c.paths, "train", drop_s1=M[f"orphan_{k}"], out=fd)
        dens = F.lookalike_density(fd, "train", M["dev"], M["dev"].sum())
        man = json.loads((V2 / "manifest_v2.json").read_text(encoding="utf-8"))
        fast = man["tiers"][k]["populations"]["SHADOW-DEV"]["lookalike_density"]
        if abs(dens - fast) > 1e-12:
            raise AssertionError(f"{k}: feature-based density {dens} != label-free construction {fast}")
        rep[k] = {"dir": fd.as_posix(), "eligible_pairs": st["eligible_pairs"], "dropped_s1": st["dropped_s1"], "seconds": round(pm.seconds, 1),
                  "peak_gb": round(pm.peak_gb, 2), "dev_density_from_features": dens}
        LOG.info("%s features: %s", k, rep[k])
        write_json(OUT / "tier_features.json", rep)


# ----------------------------------------------------------------------------- eval
def competitions(c):
    M1, M = np.load(V1 / "masks.npz"), masks_v2()
    shadow, chain_member = M1["shadow"], M1["chain_member"]
    tm = shadow & ~M1["orphan"]
    s1_tr = F.read_cols("train", ["s1_row"])["s1_row"].astype(np.int64)
    ncand_elig = np.bincount(s1_tr, minlength=c.n)
    gi = np.where(c.ctry == "india", 1, np.where(c.ctry == "us", 2, 0))
    cells = gi * 8 + (cand_bin(ncand_elig) + 1)
    w_tm = np.zeros(c.n)
    w_tm[tm] = match_weights(cells[tm], F.test_cell_shares(c))
    lk = {"SHADOW-DEV": M["dev"], "SHADOW-LOCKED-A": M["locked_A"], "SHADOW-LOCKED-B": M["locked_B"]}
    base = {"E": c.P["E"], "E_half_A(stage-2 training half: in-sample for S-01)": c.P["E_half_A"], "E_half_B": c.P["E_half_B"],
            "validation": c.P["validation"], "validation_half_A": c.P["validation_half_A"], "validation_half_B": c.P["validation_half_B"],
            "GATE-B": M1["gate_B"], "SHADOW-BASE": shadow} | lk
    return {"T1(base)": (None, None, base, {}),
            "test_matched(v1 random orphan)": (F.orphan_fdir(c), M1["orphan"], {"SHADOW-TEST-MATCHED": tm, "SHADOW-ORPHANED-unweighted": tm},
                                               {"SHADOW-TEST-MATCHED": w_tm}),
            "chain_orphaned(v1)": (F.chain_fdir(c), M1["chain_orphan"], {"SHADOW-CHAIN-ORPHANED": shadow, "SHADOW-CHAIN-MEMBERS": shadow & chain_member}, {}),
            **{f"{k}": (tier_fdir(c, k), M[f"orphan_{k}"], lk, {}) for k in ("T2", "T3", "T4")}}


def system_metrics(codes, ctl_codes, c, pops, weights, ncand, la_s1):
    tp, k = F.counts(codes, c.gtc, c.n)
    n_add = np.bincount(np.setdiff1d(codes, ctl_codes) // SHIFT, minlength=c.n)
    n_rem = np.bincount(np.setdiff1d(ctl_codes, codes) // SHIFT, minlength=c.n)
    out = {}
    for name, m in pops.items():
        w = weights.get(name)
        wm = None if w is None else w[m]
        r = weighted_metrics(tp[m], k[m], c.n_true[m], wm)
        r |= {"predictions_per_s1": avg(k[m], wm), "empty_rate": avg(k[m] == 0, wm), "additions_per_s1_vs_B06": avg(n_add[m], wm),
              "removals_per_s1_vs_B06": avg(n_rem[m], wm), "stage1_candidates_per_s1": avg(ncand[m], wm), "lookalike_density": avg(la_s1[m], wm)}
        r["by_country_f05"] = {g: weighted_metrics(tp[m & (c.ctry == g)], k[m & (c.ctry == g)], c.n_true[m & (c.ctry == g)],
                                                   None if w is None else w[m & (c.ctry == g)])["f05"] for g in ("india", "us") if (m & (c.ctry == g)).any()}
        r["by_country_removals_per_s1_vs_B06"] = {g: avg(n_rem[m & (c.ctry == g)]) for g in ("india", "us") if (m & (c.ctry == g)).any()}
        out[name] = r
    return out


def test_lookalike_count_share(c) -> dict[int, float]:
    """Label-free: share of test S1 by number of look-alike candidates (bins 0..4, 5+)."""
    te = F.read_cols("test", ["s1_row", "score", "rec_n_claim", "peer_max_sim"])
    la = (te["score"] < 0.5) & (te["rec_n_claim"] <= 1) & (te["peer_max_sim"] >= 0.8)
    b = np.minimum(np.bincount(te["s1_row"].astype(np.int64)[la], minlength=len(c.ctry_te)), 5)
    return {int(k): float(v / len(b)) for k, v in zip(*np.unique(b, return_counts=True))}


WEIGHTED = ("T1(base)", "T4")  # amendment A2: density-matched weighted tiers T1W / T4W


def cmd_eval(c):
    ncand = stage1_ncand(c)
    share = test_lookalike_count_share(c)
    rep = {"systems": ["B06 (control: B-06 + D-01 @ 0.67)", "S01 (as submitted)", "S02 (veto: B06 AND S01)"], "test_lookalike_count_share": share,
           "competitions": {}}
    for comp, (fdir, drop, pops, weights) in competitions(c).items():
        pops, weights = dict(pops), dict(weights)
        t0 = time.perf_counter()
        with PeakMemory() as pm:
            ctl = F.stage1_codes(c.paths, "train", B06_TRAIN, T_B06, True, drop=drop)
            s1, pool, prob, sc, meta = F.stage2_arrays(F.CFG_S01, "train", fdir)
            x = F.read_cols("train", ["s1_row", "rec_n_claim", "peer_max_sim"], fdir)
            if not np.array_equal(x["s1_row"].astype(np.int64), s1):
                raise AssertionError("feature columns not aligned")
            la = (sc < 0.5) & (x["rec_n_claim"] <= 1) & (x["peer_max_sim"] >= 0.8)
            la_s1 = np.bincount(s1[la], minlength=c.n)
            del x, la
            ess = {}
            if comp in WEIGHTED:
                for pn in ("SHADOW-DEV", "SHADOW-LOCKED-A", "SHADOW-LOCKED-B"):
                    m = pops[pn]
                    w = np.zeros(c.n)
                    w[m] = match_weights(np.minimum(la_s1[m], 5), share)
                    pops[f"{pn}[W]"], weights[f"{pn}[W]"] = m, w
                    ess[f"{pn}[W]"] = float(w[m].sum() ** 2 / (w[m] ** 2).sum())
            code = s1 * SHIFT + pool
            if not np.isin(ctl, code).all():
                raise AssertionError("control pair outside the eligible pairs")
            s01 = S.decide(s1, pool, prob, meta["threshold"], meta["variant"])
            cm = np.isin(code, ctl)
            s02 = restrict_additions(cm, s01, sc, np.inf)
            codes = {"B06": ctl, "S01": np.sort(code[s01]), "S02": np.sort(code[s02])}
            if len(np.setdiff1d(codes["S02"], ctl)) or not np.array_equal(codes["S02"], np.intersect1d(ctl, codes["S01"])):
                raise AssertionError("S02 is not the intersection of B06 and S01")
            del s1, pool, prob, sc, code, s01, cm, s02
            res = {name: system_metrics(cd, ctl, c, pops, weights, ncand, la_s1) for name, cd in codes.items()}
        rep["competitions"][comp] = {"populations": {k: int(v.sum()) for k, v in pops.items()}, "orphaned_s1": int(drop.sum()) if drop is not None else 0,
                                     "effective_sample_size(weighted)": ess, "metrics": res, "seconds": round(time.perf_counter() - t0, 1),
                                     "peak_gb": round(pm.peak_gb, 2)}
        first = next(iter(k for k in pops if k.startswith("SHADOW")))
        LOG.info("%-32s %s: B06 %.4f S01 %.4f S02 %.4f | density %.3f | %.0fs peak %.2f GB", comp, first, res["B06"][first]["f05"], res["S01"][first]["f05"],
                 res["S02"][first]["f05"], res["B06"][first]["lookalike_density"], time.perf_counter() - t0, pm.peak_gb)
        write_json(OUT / "shadow_eval.json", rep)
    rep["density_analysis"] = density_analysis(rep)
    write_json(OUT / "shadow_eval.json", rep)
    write_json(OUT / "density_analysis.json", rep["density_analysis"])


def density_analysis(rep) -> dict:
    C = rep["competitions"]
    tier_keys = {"T1": "T1(base)", "T2": "T2", "T3": "T3", "T4": "T4"}
    out = {"rule": "material decay (DEV, LOCKED-A, LOCKED-B each): adv(T1) - adv(X) > max(0.001, 0.25 * adv(T1)) for X in T4, T1W, T4W -> REJECT "
                   "(notes.md 1.3 + amendment A)"}
    for pop in ("SHADOW-DEV", "SHADOW-LOCKED-A", "SHADOW-LOCKED-B"):
        rows = []
        for t, ck, pk in [(t, ck, pop) for t, ck in tier_keys.items()] + [("T1W", "T1(base)", f"{pop}[W]"), ("T4W", "T4", f"{pop}[W]")]:
            m = C[ck]["metrics"]
            b, s1_, s2_ = m["B06"][pk], m["S01"][pk], m["S02"][pk]
            rows.append({"tier": t, "lookalike_density": b["lookalike_density"], "B06_f05": b["f05"], "S01_f05": s1_["f05"], "S02_f05": s2_["f05"],
                         "adv_S02": s2_["f05"] - b["f05"], "adv_S01": s1_["f05"] - b["f05"],
                         "S01_additions_per_s1": s1_["additions_per_s1_vs_B06"], "S01_removals_per_s1": s1_["removals_per_s1_vs_B06"],
                         "S02_removals_per_s1": s2_["removals_per_s1_vs_B06"], "S02_additions_per_s1": s2_["additions_per_s1_vs_B06"],
                         "B06_fm_per_1k": b["false_merges_per_1k_s1"], "S01_fm_per_1k": s1_["false_merges_per_1k_s1"], "S02_fm_per_1k": s2_["false_merges_per_1k_s1"],
                         "B06_singleton_fp": b["singleton_fp_rate"], "S01_singleton_fp": s1_["singleton_fp_rate"], "S02_singleton_fp": s2_["singleton_fp_rate"],
                         "B06_P": b["precision"], "S01_P": s1_["precision"], "S02_P": s2_["precision"]})
        for r in rows:
            r["effective_sample_size"] = C["T1(base)" if r["tier"] == "T1W" else "T4"]["effective_sample_size(weighted)"].get(f"{pop}[W]")                 if r["tier"] in ("T1W", "T4W") else None
        pure = rows[:4]
        d = np.array([r["lookalike_density"] for r in pure])
        slope = lambda key: float(np.polyfit(d, [r[key] for r in pure], 1)[0]) if np.ptp(d) > 0 else float("nan")  # noqa: E731
        a1 = rows[0]["adv_S02"]
        thr = max(0.001, 0.25 * a1)
        decays = {r["tier"]: a1 - r["adv_S02"] for r in rows if r["tier"] in ("T4", "T1W", "T4W")}
        out[pop] = {"tiers": rows, "slope_adv_S02_per_unit_density(T1..T4)": slope("adv_S02"), "slope_adv_S01_per_unit_density(T1..T4)": slope("adv_S01"),
                    "decay_S02_vs_T1": decays, "material_threshold": thr, "material_decay": {k: bool(v > thr) for k, v in decays.items()},
                    "any_material_decay": bool(any(v > thr for v in decays.values())),
                    "S01_advantage_falls_with_density(T1..T4)": bool(all(pure[i + 1]["adv_S01"] < pure[i]["adv_S01"] for i in range(3))),
                    "S01_additions_rise_with_density(T1..T4)": bool(all(pure[i + 1]["S01_additions_per_s1"] > pure[i]["S01_additions_per_s1"] for i in range(3))),
                    "S01_advantage_at_test_density": {r["tier"]: r["adv_S01"] for r in rows if r["tier"] in ("T1W", "T4W")}}
    return out


# ----------------------------------------------------------------------------- test-side diagnostics (label-free)
def row_counts(files: dict[str, Path]) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    """Streams the matching files in lockstep (same S1 order is asserted): per row, the number of predictions and
    the additions / removals relative to the B06 file. Memory: a few int arrays, never the id lists."""
    names = list(files)
    fh = [open(files[k], "rb") for k in names]
    try:
        heads = [next(f) for f in fh]
        if len(set(heads)) != 1:
            raise AssertionError("different headers")
        ids, cnt = [], {k: {"k": [], "add": [], "rem": []} for k in names}
        for lines in zip(*fh):
            parts = [ln.rstrip(b"\n").partition(b"\t") for ln in lines]
            if len({x[0] for x in parts}) != 1:
                raise AssertionError("row order differs between files")
            ids.append(parts[0][0].decode())
            sets = [{x for x in q[2].split(b",") if x} for q in parts]
            base = sets[names.index("B06")]
            for k, st in zip(names, sets):
                cnt[k]["k"].append(len(st))
                cnt[k]["add"].append(len(st - base))
                cnt[k]["rem"].append(len(base - st))
        for f in fh:
            if next(f, None) is not None:
                raise AssertionError("files have different row counts")
    finally:
        for f in fh:
            f.close()
    return ids, {k: {x: np.asarray(v, np.int64) for x, v in d.items()} for k, d in cnt.items()}


def cmd_testdiag(c):
    files = {"B06": Path("experiments/runs/EXP-B-06/output/matching_results.tsv"), "S01": Path("experiments/runs/EXP-S-01/output/matching_results.tsv"),
             "S02": RUN / "output" / "matching_results.tsv"}
    order, cnt = row_counts(files)
    if order != _ids(c.paths, "test", "source1").to_pylist():
        raise AssertionError("row order differs from the test S1 order")
    ctry = c.ctry_te
    kb = cnt["B06"]["k"]
    rep = {"label": "UNLABELED test diagnostics: output files vs the B-06 output file", "files": {k: v.as_posix() for k, v in files.items()}}
    for sysname in ("S02", "S01"):
        kx, add, rem = cnt[sysname]["k"], cnt[sysname]["add"], cnt[sysname]["rem"]
        r = {}
        for g in ("all", "france", "india", "us"):
            m = np.ones(len(kb), bool) if g == "all" else ctry == g
            r[g] = {"s1": int(m.sum()), "B06_predictions": int(kb[m].sum()), "predictions": int(kx[m].sum()),
                    "prediction_change": float(kx[m].sum() / kb[m].sum() - 1), "additions": int(add[m].sum()), "removals": int(rem[m].sum()),
                    "additions_per_s1": float(add[m].sum() / m.sum()), "removals_per_s1": float(rem[m].sum() / m.sum()),
                    "removal_rate(of B06 predictions)": float(rem[m].sum() / kb[m].sum()), "empty_rate": float((kx[m] == 0).mean()),
                    "B06_empty_rate": float((kb[m] == 0).mean())}
        rep[sysname] = r
    ev = json.loads((RUN / "stage2_eval.json").read_text(encoding="utf-8"))
    tr = ev["results"]["E_half_B"]["vs_control"]["removals_per_s1"]
    sh = json.loads((OUT / "shadow_eval.json").read_text(encoding="utf-8"))
    tr_c = sh["competitions"]["T1(base)"]["metrics"]["S02"]["E_half_B"]["by_country_removals_per_s1_vs_B06"]
    s2 = rep["S02"]
    rep["train_reference(E_half_B)"] = {"removals_per_s1": tr, "by_country": tr_c}
    rep["ratios"] = {"test_over_train_removals_per_s1": s2["all"]["removals_per_s1"] / tr,
                     "india_test_over_train": s2["india"]["removals_per_s1"] / tr_c["india"], "us_test_over_train": s2["us"]["removals_per_s1"] / tr_c["us"],
                     "france_removal_rate_over_overall": s2["france"]["removal_rate(of B06 predictions)"] / s2["all"]["removal_rate(of B06 predictions)"],
                     "S01_test_additions_per_s1_over_train": rep["S01"]["all"]["additions_per_s1"]
                     / sh["competitions"]["T1(base)"]["metrics"]["S01"]["E_half_B"]["additions_per_s1_vs_B06"]}
    write_json(OUT / "test_diagnostics.json", rep)
    LOG.info("testdiag S02: %s | ratios %s", {g: (round(v["prediction_change"], 4), v["additions"]) for g, v in s2.items()}, rep["ratios"])


# ----------------------------------------------------------------------------- error budget (post-S-02)
def cmd_budget(c):
    d = _read_parts(c.paths.scored_dir("train", B06_TRAIN), ["s1_row", "pool_row", "score"])
    cand = d["s1_row"].astype(np.int64) * SHIFT + d["pool_row"].astype(np.int64)
    csc = d["score"].astype(np.float32)
    del d
    o = np.argsort(cand, kind="stable")
    cand, csc = cand[o], csc[o]
    ctl = F.stage1_codes(c.paths, "train", B06_TRAIN, T_B06, True)
    s1, pool, prob, sc, meta = F.stage2_arrays(F.CFG_S01, "train")
    code = s1 * SHIFT + pool
    s01 = S.decide(s1, pool, prob, meta["threshold"], meta["variant"])
    s02c = np.sort(code[restrict_additions(np.isin(code, ctl), s01, sc, np.inf)])
    del s1, pool, prob, sc, s01
    gt = np.sort(c.gtc)
    pred_true = np.isin(s02c, gt)
    fn = gt[~np.isin(gt, s02c)]  # true pairs not predicted
    pos = np.searchsorted(cand, fn)
    pos_c = np.minimum(pos, len(cand) - 1)
    in_cand = cand[pos_c] == fn
    fscore = np.where(in_cand, csc[pos_c], np.nan)
    in_ctl = np.isin(fn, ctl)
    buckets_fn = {"never_retrieved (not in the top-20 candidate list)": ~in_cand,
                  "retrieved, stage-1 < 0.02 (not eligible for stage 2)": in_cand & (fscore < 0.02),
                  "eligible, stage-1 0.02-0.5 (control rejects)": in_cand & (fscore >= 0.02) & (fscore < 0.5),
                  "eligible, stage-1 0.5-0.67 (control rejects)": in_cand & (fscore >= 0.5) & (fscore < T_B06),
                  "stage-1 >= 0.67 but lost D-01 (control rejects)": in_cand & (fscore >= T_B06) & ~in_ctl,
                  "control accepts, vetoed by S-01": in_ctl}
    fp = s02c[~pred_true]
    single = c.n_true == 0
    buckets_fp = {"false merges on S1 with true matches": ~single[fp // SHIFT], "false merges on singleton S1": single[fp // SHIFT]}
    tp, k = np.bincount(s02c[pred_true] // SHIFT, minlength=c.n), np.bincount(s02c // SHIFT, minlength=c.n)
    M = masks_v2()
    rep = {"system": "S02", "definition": "oracle F0.5 gain from fixing ONE bucket alone (upper bound); macro per-S1 F0.5", "populations": {}}
    for pname, m in (("E_half_B", c.P["E_half_B"]), ("SHADOW-DEV", M["dev"])):
        base = weighted_metrics(tp[m], k[m], c.n_true[m])["f05"]
        r = {"f05": base, "s1": int(m.sum()), "true_pairs": int(c.n_true[m].sum()), "fn": {}, "fp": {}}
        for name, bm in buckets_fn.items():
            add = np.bincount(fn[bm] // SHIFT, minlength=c.n)
            r["fn"][name] = {"pairs": int(add[m].sum()), "share_of_true_pairs": float(add[m].sum() / max(c.n_true[m].sum(), 1)),
                             "s1_affected": int((add[m] > 0).sum()), "oracle_gain": weighted_metrics(tp[m] + add[m], k[m] + add[m], c.n_true[m])["f05"] - base}
        for name, bm in buckets_fp.items():
            rm = np.bincount(fp[bm] // SHIFT, minlength=c.n)
            r["fp"][name] = {"pairs": int(rm[m].sum()), "s1_affected": int((rm[m] > 0).sum()),
                             "oracle_gain": weighted_metrics(tp[m], k[m] - rm[m], c.n_true[m])["f05"] - base}
        rep["populations"][pname] = r
        LOG.info("budget %s F0.5 %.4f: %s", pname, base, {k2: round(v["oracle_gain"], 4) for k2, v in (r["fn"] | r["fp"]).items()})
    # profile of the largest FN bucket on E half B (never-retrieved), label-free pair views on a deterministic sample
    nr = fn[buckets_fn["never_retrieved (not in the top-20 candidate list)"]]
    nr = nr[c.P["E_half_B"][nr // SHIFT]]
    idx = F.sample_idx(nr, 100_000)
    xmap = F.XlitMap.load(Path(c.cfg["stage2"]["xlit_map"]))
    nm, ad = F.pair_sims(c.paths, "train", nr[idx] // SHIFT, nr[idx] % SHIFT, xmap)
    cat = F.sim_category(nm, ad)
    u, cn = np.unique(cat, return_counts=True)
    rep["never_retrieved_profile(E_half_B, sample)"] = {"pairs": int(len(nr)), "sample": int(len(idx)), "pair_type_share": {a: float(b / len(cat)) for a, b in zip(u, cn)},
                                                       "by_country": {g: float((c.ctry[nr // SHIFT] == g).mean()) for g in ("india", "us")},
                                                       "name_tokset_quantiles": {q: float(np.nanquantile(nm, q)) for q in (0.1, 0.25, 0.5, 0.75)}}
    write_json(OUT / "error_budget.json", rep)


# ----------------------------------------------------------------------------- leakage + read-only proof
def cmd_leakage(c):
    M1, M = np.load(V1 / "masks.npz"), masks_v2()
    rep = {}
    for pn in ("dev", "locked_A", "locked_B"):
        rep[pn] = leakage_checks(M[pn], stage1_fit=c.P["fit"], stage2_train=c.P["stage2_train"], stage2_inner_threshold_and_early_stopping=c.P["stage2_inner"],
                                 validation=c.P["validation"])
    rep["dev_locked_disjoint"] = {"dev&A": int((M["dev"] & M["locked_A"]).sum()), "dev&B": int((M["dev"] & M["locked_B"]).sum()),
                                  "A&B": int((M["locked_A"] & M["locked_B"]).sum())}
    rep["union_equals_shadow_base_v1"] = bool(((M["dev"] | M["locked_A"] | M["locked_B"]) == M1["shadow"]).all())
    rep["tier_masks_never_remove_an_evaluated_s1"] = {k: int((M[f"orphan_{k}"] & M1["shadow"]).sum()) for k in ("T2", "T3", "T4")}
    rep["tiers_nested"] = bool((M["orphan_T2"] <= M["orphan_T3"]).all() and (M["orphan_T3"] <= M["orphan_T4"]).all())
    xm = F.XlitMap.load(Path(c.cfg["stage2"]["xlit_map"]))
    rep["xlit_map_learned_from_fit_set_only"] = xm.meta.get("learned_from_digest") == F.XlitMap.ids_digest([i for i, f in zip(c.ids, c.P["fit"]) if f])
    before = json.loads((OUT / "readonly_snapshot_before.json").read_text(encoding="utf-8"))
    now = readonly_state(c)
    rep["readonly_unchanged"] = {k: now[k] == before[k] for k in before}
    rep["exp_s02_run_dir_has_no_model"] = not list(RUN.glob("stage2_model*"))
    rep["all_ok"] = (all(v == 0 for pn in ("dev", "locked_A", "locked_B") for v in rep[pn].values())
                     and all(v == 0 for v in rep["dev_locked_disjoint"].values()) and rep["union_equals_shadow_base_v1"]
                     and all(v == 0 for v in rep["tier_masks_never_remove_an_evaluated_s1"].values()) and rep["tiers_nested"]
                     and rep["xlit_map_learned_from_fit_set_only"] and all(rep["readonly_unchanged"].values()) and rep["exp_s02_run_dir_has_no_model"])
    write_json(OUT / "leakage.json", rep)
    LOG.info("leakage: %s", rep)


# ----------------------------------------------------------------------------- gates (pre-declared in notes.md 1.3)
def cmd_gates(c):
    ev = json.loads((RUN / "stage2_eval.json").read_text(encoding="utf-8"))
    sh = json.loads((OUT / "shadow_eval.json").read_text(encoding="utf-8"))
    td = json.loads((OUT / "test_diagnostics.json").read_text(encoding="utf-8"))
    lk = json.loads((OUT / "leakage.json").read_text(encoding="utf-8"))
    sub = json.loads((RUN / "submission_summary.json").read_text(encoding="utf-8"))
    base = sh["competitions"]["T1(base)"]["metrics"]
    g = {"local": {}, "shadow": {}, "density": {}, "test_behaviour": {}, "other": {}}
    for pop in ("validation", "GATE-B", "E_half_B"):
        b, s = base["B06"][pop], base["S02"][pop]
        g["local"][pop] = {"f05_ge_B06_plus_0.002": s["f05"] >= b["f05"] + 0.002, "precision_ge_B06": s["precision"] >= b["precision"],
                           "singleton_fp_le_B06": s["singleton_fp_rate"] <= b["singleton_fp_rate"],
                           "india_ge_B06": s["by_country_f05"]["india"] >= b["by_country_f05"]["india"],
                           "us_ge_B06": s["by_country_f05"]["us"] >= b["by_country_f05"]["us"], "delta_f05": s["f05"] - b["f05"]}
    for comp, pop in (("T1(base)", "SHADOW-BASE"), ("test_matched(v1 random orphan)", "SHADOW-TEST-MATCHED"),
                      ("chain_orphaned(v1)", "SHADOW-CHAIN-ORPHANED"), ("chain_orphaned(v1)", "SHADOW-CHAIN-MEMBERS")):
        m = sh["competitions"][comp]["metrics"]
        g["shadow"][pop] = {"delta_f05": m["S02"][pop]["f05"] - m["B06"][pop]["f05"], "pass": m["S02"][pop]["f05"] >= m["B06"][pop]["f05"] + 0.002}
    da = sh["density_analysis"]
    for pop in ("SHADOW-DEV", "SHADOW-LOCKED-A", "SHADOW-LOCKED-B"):
        g["density"][pop] = {"decay": da[pop]["decay_S02_vs_T1"], "threshold": da[pop]["material_threshold"], "pass": not da[pop]["any_material_decay"]}
    s2 = td["S02"]
    g["test_behaviour"] = {"prediction_change_in_[-2%,0]": -0.02 <= s2["all"]["prediction_change"] <= 0, "additions_zero": s2["all"]["additions"] == 0,
                           "removals_per_s1_within_25pct_of_train": abs(td["ratios"]["test_over_train_removals_per_s1"] - 1) <= 0.25,
                           "per_country_change_in_[-3%,0]": all(-0.03 <= s2[k]["prediction_change"] <= 0 for k in ("france", "india", "us")),
                           "france_removal_rate_le_1.5x_overall": td["ratios"]["france_removal_rate_over_overall"] <= 1.5}
    g["other"] = {"leakage_and_readonly_ok": lk["all_ok"], "strict_validator_pass": sub["strict_validation_errors"] == [],
                  "evaluate_peak_le_5GB": ev["eval_peak_rss_gb"] <= 5.0}
    flat = [v for d in g["local"].values() for k, v in d.items() if k != "delta_f05"] + [d["pass"] for d in g["shadow"].values()] \
        + [d["pass"] for d in g["density"].values()] + list(g["test_behaviour"].values()) + list(g["other"].values())
    g["all_pass"] = bool(all(flat))
    write_json(OUT / "gates.json", g)
    metrics = {"experiment": "EXP-S-02", "decision_rule": ev["decision"], "local": {p: {s: {k: base[s][p][k] for k in ("f05", "precision", "recall",
               "false_merges_per_1k_s1", "singleton_fp_rate", "predictions_per_s1", "empty_rate", "additions_per_s1_vs_B06", "removals_per_s1_vs_B06",
               "stage1_candidates_per_s1", "by_country_f05")} for s in ("B06", "S01", "S02")} for p in base["B06"]},
               "stage2_eval(run_stage2 evaluate)": {p: {"control_f05": r["control"]["f05"], "s02_f05": r["stage2"]["f05"], "s01_f05": r["stage2_unrestricted"]["f05"],
                                                        "vs_control": r["vs_control"]} for p, r in ev["results"].items()},
               "density_analysis": da, "test_diagnostics": td, "gates": g, "eval_peak_rss_gb": ev["eval_peak_rss_gb"], "eval_seconds": ev["eval_seconds"],
               "submission": {k: sub[k] for k in ("matching_file", "candidate_file", "competition_after", "vs_control_test", "strict_validation_errors")}}
    write_json(RUN / "metrics.json", metrics)
    LOG.info("gates: all_pass=%s | %s", g["all_pass"], json.dumps(g, default=str)[:1500])


# ----------------------------------------------------------------------------- main
class Ctx(F.Ctx):
    def __init__(self):
        super().__init__()
        mt, me = F.load_manifest(self.paths, "train"), F.load_manifest(self.paths, "test")
        self.train_pool = int(mt["source2"]["rows"]) + int(mt["source3"]["rows"])
        self.test_pool = int(me["source2"]["rows"]) + int(me["source3"]["rows"])


CMDS = {"snapshot": cmd_snapshot, "smoke": cmd_smoke, "tiers-build": cmd_tiers_build, "tiers-features": cmd_tiers_features, "eval": cmd_eval,
        "testdiag": cmd_testdiag, "budget": cmd_budget, "leakage": cmd_leakage, "gates": cmd_gates}


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
