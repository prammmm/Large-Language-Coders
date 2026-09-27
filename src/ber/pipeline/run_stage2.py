"""EXP-S-01 pipeline: collective second-stage decision model on top of a frozen stage-1 system.

    python -m ber.pipeline.run_stage2 <features|fit|evaluate|submit|all> --config configs/exp_s01_stage2.yaml [--split train|test]

features  eligible pairs (stage-1 score >= min_score) of ALL S1 of a split (full competition)
          -> context + collective features -> data/processed/<split>/stage2_<exp>/  (train: from the
          stage-1 all-train scored parts; test: from the stage-1 test scored parts)
fit       LightGBM on E half A minus frozen validation ONLY (E = train S1 not in the stage-1 fit set);
          early stopping + threshold + decision variant (threshold / threshold + D-01) on an inner
          hash split of that set. Guards raise if any validation / E half B / stage-1-fit S1 is used.
evaluate  held-out E half B and frozen validation (and halves) vs the stage-1 control (t + D-01)
submit    test: stage-2 decisions -> matching_results.tsv; candidate_pairs.tsv = the unchanged stage-1
          candidate set; strict validation. Writes to ``paths.output_dir`` (an experiment folder).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ber.config import Paths, load_config, parse_sets
from ber.data.normalize_records import load_manifest, read_country
from ber.data.split import load_val_ids, match_counts
from ber.decision.competition import best_claimant_keep, claim_stats
from ber.decision.stage2 import VIEW_COLUMNS, context_features, feature_names, peer_features, peer_index, restrict_additions
from ber.evaluation.metrics import evaluate_arrays
from ber.normalization.crossscript import XlitMap
from ber.pipeline.run_baseline import _clear_dir, _hash_frac, _ids, _read_parts, fit_s1_subset, pool_ids
from ber.submission.strict_validator import validate as strict_validate
from ber.submission.writer import CANDIDATE_HEADER, MATCHING_HEADER, write_id_lists
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, stage, write_json

SHIFT = np.int64(1) << np.int64(24)
KEYS = ("entities", "f05", "macro_precision", "macro_recall", "false_merges", "singleton_false_positive", "empty_on_positive",
        "predicted_pairs", "tp_pairs", "predicted_per_s1")


def s2cfg(cfg: dict) -> dict:
    return cfg["stage2"]


def feat_dir(cfg, paths, split) -> Path:
    src = s2cfg(cfg).get("features_from") or cfg["experiment"]["id"]  # variants may re-use another run's feature parts
    return paths.processed_dir(split) / f"stage2_{src}"


def model_dir(cfg, paths) -> Path:
    """Run folder holding the stage-2 model: ``stage2.model_from`` (read-only re-use) or this experiment."""
    return paths.run_dir(s2cfg(cfg).get("model_from") or cfg["experiment"]["id"])


def reuses(cfg, key: str) -> bool:
    v = s2cfg(cfg).get(key)
    return bool(v) and v != cfg["experiment"]["id"]


def add_floor(cfg) -> float | None:
    """``stage2.add_floor``: None (default) = stage-2 decisions as they are; a number = stage 2 may only ADD
    pairs whose stage-1 score >= it (``.inf`` = veto only: final = control AND stage 2, EXP-S-02)."""
    v = s2cfg(cfg).get("add_floor")
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not v >= 0:
        raise ValueError(f"stage2.add_floor must be null or a number >= 0 (.inf = veto only), got {v!r}")
    return float(v)


def decision_record(cfg, paths, meta) -> dict:
    f = add_floor(cfg)
    return {"variant": meta["variant"], "threshold": meta["threshold"], "add_floor": None if f is None else ("inf" if np.isinf(f) else f),
            "model_dir": model_dir(cfg, paths).as_posix(), "features_from": s2cfg(cfg).get("features_from") or cfg["experiment"]["id"],
            "control": f"stage-1 threshold {stage1_threshold(cfg)} + D-01 (full competition)"}


def stage1_parts(cfg, paths, split) -> Path:
    return paths.scored_dir(split, s2cfg(cfg)["train_tag"]) if split == "train" else paths.scored_dir(split, cfg["scoring"]["tag"])


def stage1_threshold(cfg) -> float:
    return float(json.loads(Path(cfg["matcher"]["model_path"]).read_text(encoding="utf-8"))["meta"]["threshold"])


# ----------------------------------------------------------------------------- views
class PoolViews:
    """Record views for global pool rows (S2 rows, then S3 rows + n_s2), loaded per chunk."""

    def __init__(self, paths: Paths, split: str, xmap: XlitMap | None):
        self.paths, self.split, self.xmap = paths, split, xmap
        self.man = load_manifest(paths, split)
        self.n_s2 = int(self.man["source2"]["rows"])
        self.cache: dict[int, tuple] = {}

    def prepare(self, rows: np.ndarray) -> None:
        self.cache = {}
        rows = np.unique(rows)
        for source, lo, hi in (("source2", 0, self.n_s2), ("source3", self.n_s2, 1 << 62)):
            sel = rows[(rows >= lo) & (rows < hi)] - lo
            if len(sel) == 0:
                continue
            want = pa.array(sel.astype(np.int32))
            for key in self.man[source]["countries"]:
                t = read_country(self.paths, self.split, source, key, ["row", "name_core", "addr_norm", "addr_nums", "addr_words"])
                t = t.filter(pc.is_in(t.column("row"), value_set=want))
                if t.num_rows == 0:
                    continue
                core = [x or "" for x in t.column("name_core").to_pylist()]
                xl = self.xmap.apply(core) if self.xmap is not None else core
                for r, a, x, b, c, d in zip(t.column("row").to_pylist(), core, xl, t.column("addr_norm").to_pylist(),
                                            t.column("addr_nums").to_pylist(), t.column("addr_words").to_pylist()):
                    self.cache[r + lo] = (a, x, b or "", c or "", d or "")

    def __call__(self, rows: np.ndarray) -> dict[str, list[str]]:
        empty = ("", "", "", "", "")
        vals = [self.cache.get(int(r), empty) for r in rows]
        return {c: [v[i] for v in vals] for i, c in enumerate(VIEW_COLUMNS)}


# ----------------------------------------------------------------------------- features
def build_features(cfg: dict, paths: Paths, split: str, drop_s1: np.ndarray | None = None, out: Path | None = None) -> dict:
    """``drop_s1`` (bool per S1 row, diagnostics only): those S1 are removed from the competition before any
    feature is computed (their records stay in the pool, unclaimed). ``out``: alternative output directory."""
    c = s2cfg(cfg)
    if out is None and reuses(cfg, "features_from"):
        raise RuntimeError(f"stage2.features_from={c['features_from']}: those feature parts are re-used read-only; not rebuilding them")
    k, min_score = int(c["peers"]), float(c["min_score"])
    d = _read_parts(stage1_parts(cfg, paths, split), ["s1_row", "pool_row", "score"])
    s1, pool, sc = d["s1_row"].astype(np.int64), d["pool_row"].astype(np.int64), d["score"].astype(np.float32)
    del d
    keep = sc >= np.float32(min_score)
    if drop_s1 is not None:
        keep &= ~drop_s1[s1]
    s1, pool, sc = s1[keep], pool[keep], sc[keep]
    o = np.lexsort((pool, s1))  # deterministic storage order
    s1, pool, sc = s1[o], pool[o], sc[o]
    ctx = context_features(s1, pool, sc)
    xmap = XlitMap.load(Path(c["xlit_map"])) if c.get("xlit_map") else None
    views = PoolViews(paths, split, xmap)
    out = out or feat_dir(cfg, paths, split)
    _clear_dir(out)
    bounds = np.flatnonzero(np.r_[True, s1[1:] != s1[:-1]])  # S1 group starts (chunks never split an S1)
    target = int(c.get("chunk_pairs", 1_500_000))
    part, lo = 0, 0
    names = feature_names(k)
    while lo < len(s1):
        hi_i = np.searchsorted(bounds, lo + target, side="left")
        hi = int(bounds[hi_i]) if hi_i < len(bounds) else len(s1)
        if hi <= lo:
            hi = len(s1)
        cs1, cpool, csc = s1[lo:hi], pool[lo:hi], sc[lo:hi]
        peers = peer_index(cs1, cpool, csc, k)
        need = np.concatenate([cpool] + [cpool[p[p >= 0]] for p in peers])
        views.prepare(need)
        pf = peer_features(cpool, csc, peers, views, int(c.get("workers", 8)))
        cols = {"s1_row": pa.array(cs1.astype(np.int32)), "pool_row": pa.array(cpool.astype(np.int32))}
        for f in names:
            cols[f] = pa.array((ctx[f][lo:hi] if f in ctx else pf[f]).astype(np.float32))
        pq.write_table(pa.table(cols), out / f"part-{part:05d}.parquet", compression="zstd")
        LOG.info("stage2 features %s: part %d, pairs %d..%d of %d", split, part, lo, hi, len(s1))
        part += 1
        lo = hi
    stats = {"split": split, "eligible_pairs": int(len(s1)), "min_score": min_score, "peers": k, "parts": part, "features": names,
             "dropped_s1": int(drop_s1.sum()) if drop_s1 is not None else 0}
    write_json(out.parent / f"{out.name}_stats.json", stats)
    return stats


def read_features(cfg, paths, split, row_mask: np.ndarray | None = None, columns=None):
    parts = sorted(feat_dir(cfg, paths, split).glob("*.parquet"))
    tabs = []
    for p in parts:
        t = pq.read_table(p, columns=columns)
        if row_mask is not None:
            t = t.filter(pa.array(row_mask[t.column("s1_row").to_numpy().astype(np.int64)]))
        tabs.append(t)
    return pa.concat_tables(tabs) if tabs else None


# ----------------------------------------------------------------------------- populations
def populations(cfg, paths):
    """Train-split S1 masks. E = not in the stage-1 fit set. Halves = sha1(42|id) % 2 (project rule)."""
    ids = _ids(paths, "train", "source1")
    n = len(ids)
    ix = lambda v: pc.index_in(pa.array(v, pa.string()), value_set=ids).to_numpy(zero_copy_only=False).astype(np.int64)  # noqa: E731
    fit = np.zeros(n, bool)
    fit[ix(fit_s1_subset(cfg, paths))] = True
    val = np.zeros(n, bool)
    val[ix(load_val_ids(paths))] = True
    idl = ids.to_pylist()
    half = np.array([int(hashlib.sha1(f"42|{v}".encode()).hexdigest(), 16) % 2 for v in idl], np.int8)
    tf = float(s2cfg(cfg).get("inner_fraction", 0.2))
    inner = np.array([_hash_frac(cfg["seed"] + 7, v) < tf for v in idl], bool)
    E = ~fit
    train_all = E & (half == 0) & ~val
    tc = s2cfg(cfg).get("train_countries")  # diagnostic (leave-one-country-out); default: every country
    if tc:
        s1c = pq.read_table(paths.interim_file("train", "source1"), columns=["country"]).column("country").to_pylist()
        from ber.data.normalize_records import country_key
        train_all &= np.array([country_key(x) in set(tc) for x in s1c], bool)
    P = {"stage2_train": train_all & ~inner, "stage2_inner": train_all & inner, "E": E, "E_half_A": E & (half == 0),
         "E_half_B": E & (half == 1), "validation": val, "validation_half_A": val & (half == 0), "validation_half_B": val & (half == 1),
         "fit": fit}
    for bad in ("validation", "E_half_B", "fit"):
        if (P[bad] & (P["stage2_train"] | P["stage2_inner"])).any():
            raise RuntimeError(f"leakage: {bad} S1 inside the stage-2 training partition")
    return P, n


def gt_codes(paths):
    s1_ids, p_ids = _ids(paths, "train", "source1"), pool_ids(paths, "train")
    gt = pq.read_table(paths.interim_file("train", "ground_truth"))
    L = pc.split_pattern(gt.column("matched_entity_ids"), ",")
    flat, par = pc.list_flatten(L), pc.list_parent_indices(L)
    keep = pc.not_equal(flat, "")
    s = pc.index_in(pc.take(gt.column("source1_entity_id"), pc.filter(par, keep)), value_set=s1_ids).to_numpy(zero_copy_only=False).astype(np.int64)
    p = pc.index_in(pc.filter(flat, keep), value_set=p_ids).to_numpy(zero_copy_only=False).astype(np.int64)
    n_true = np.bincount(s, minlength=len(s1_ids))
    return np.sort(s * SHIFT + p), n_true


def decide(s1, pool, prob, t, variant):
    acc = prob >= np.float32(t)
    if variant == "threshold+D01":
        acc = best_claimant_keep(pool, s1, prob, acc)
    return acc


def control_decisions(cfg, paths, split):
    """The control = the frozen stage-1 system: its threshold + D-01 over ALL stage-1 pairs of the split (full
    competition). Returns the sorted accepted pair codes, their stage-1 scores, the conflict stats after D-01
    and the number of stage-1 candidates per S1. Called before stage 2 is loaded (memory).
    Streamed per part: D-01 only ever compares pairs above the threshold, so only those are kept (identical result,
    ~6x less memory than materialising every candidate pair; keeps the submit peak <= 5 GB)."""
    t1 = np.float32(stage1_threshold(cfg))
    s1s, pools, scs, n_cand = [], [], [], np.zeros(0, np.int64)
    for p in sorted(stage1_parts(cfg, paths, split).glob("*.parquet")):
        t = pq.read_table(p, columns=["s1_row", "pool_row", "score"])
        s1p, sp = t.column("s1_row").to_numpy().astype(np.int64), t.column("score").to_numpy().astype(np.float32)
        b = np.bincount(s1p)
        n_cand = np.pad(n_cand, (0, max(0, len(b) - len(n_cand))))
        n_cand[: len(b)] += b
        m = sp >= t1
        s1s.append(s1p[m])
        pools.append(t.column("pool_row").to_numpy()[m].astype(np.int64))
        scs.append(sp[m])
        del t, s1p, sp, m
    cs1, cpool, csc = (np.concatenate(x) if x else np.zeros(0, dt) for x, dt in ((s1s, np.int64), (pools, np.int64), (scs, np.float32)))
    del s1s, pools, scs
    acc1 = best_claimant_keep(cpool, cs1, csc, np.ones(len(cs1), bool))
    conflicts = claim_stats(cpool, cs1, acc1)
    code, score = cs1[acc1] * SHIFT + cpool[acc1], csc[acc1]
    del cs1, cpool, csc, acc1
    gc.collect()
    o = np.argsort(code, kind="stable")
    return code[o], score[o], conflicts, n_cand


def final_decisions(cfg, s1, pool, prob, sc, meta, control_codes=None):
    """Stage-2 decisions (threshold / threshold + D-01 of the model meta), then -- when ``stage2.add_floor`` is
    set -- additions over the control restricted to stage-1 score >= add_floor (inf: veto only)."""
    acc = decide(s1, pool, prob, meta["threshold"], meta["variant"])
    f = add_floor(cfg)
    if f is None:
        return acc
    if control_codes is None:
        raise ValueError("stage2.add_floor needs the control decisions")
    code = s1 * SHIFT + pool
    if not np.isin(control_codes, code).all():  # the control only accepts pairs with score >= t1 >= min_score
        raise RuntimeError("a control-accepted pair is missing from the stage-2 eligible pairs")
    return restrict_additions(np.isin(code, control_codes), acc, sc, f)


# ----------------------------------------------------------------------------- fit / predict
def lgb_params(c):
    p = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.9,
         "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "seed": 42, "deterministic": True, "force_row_wise": True,
         "verbosity": -1, "num_threads": 8}
    p.update((c.get("lightgbm", {}) or {}).get("params", {}) or {})
    return p


def predict_all(cfg, paths, split, booster, names, fdir: Path | None = None):
    """Stage-2 probability for every eligible pair of a split (streamed per part)."""
    s1s, pools, probs, scores = [], [], [], []
    for p in sorted((fdir or feat_dir(cfg, paths, split)).glob("*.parquet")):
        t = pq.read_table(p)
        X = np.column_stack([t.column(f).to_numpy() for f in names]).astype(np.float32)
        probs.append(booster.predict(X, num_threads=8).astype(np.float32))
        s1s.append(t.column("s1_row").to_numpy().astype(np.int64))
        pools.append(t.column("pool_row").to_numpy().astype(np.int64))
        scores.append(t.column("score").to_numpy().astype(np.float32))
    return np.concatenate(s1s), np.concatenate(pools), np.concatenate(probs), np.concatenate(scores)


def model_features(c: dict) -> list[str]:
    """Stage-2 model inputs: all built features minus ``stage2.exclude_features`` (e.g. count-dependent ones)."""
    excl = set(c.get("exclude_features", []) or [])
    unknown = excl - set(feature_names(int(c["peers"])))
    if unknown:
        raise ValueError(f"stage2.exclude_features: unknown features {sorted(unknown)}")
    return [f for f in feature_names(int(c["peers"])) if f not in excl]


def run_fit(cfg, paths) -> dict:
    c = s2cfg(cfg)
    if reuses(cfg, "model_from"):
        raise RuntimeError(f"stage2.model_from={c['model_from']}: that model is re-used read-only; fit is disabled for this experiment")
    names = model_features(c)
    P, n = populations(cfg, paths)
    gtc, n_true = gt_codes(paths)
    with PeakMemory() as pm:
        tab = read_features(cfg, paths, "train", P["stage2_train"] | P["stage2_inner"])
        s1 = tab.column("s1_row").to_numpy().astype(np.int64)
        y = np.isin(s1 * SHIFT + tab.column("pool_row").to_numpy().astype(np.int64), gtc)
        X = np.column_stack([tab.column(f).to_numpy() for f in names]).astype(np.float32)
        del tab
        tr, va = P["stage2_train"][s1], P["stage2_inner"][s1]
        lc = c.get("lightgbm", {}) or {}
        dtr = lgb.Dataset(X[tr], y[tr].astype(np.float32), feature_name=names, free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va].astype(np.float32), reference=dtr, free_raw_data=True)
        t0 = time.perf_counter()
        booster = lgb.train(lgb_params(c), dtr, num_boost_round=int(lc.get("num_boost_round", 2000)), valid_sets=[dva],
                            callbacks=[lgb.early_stopping(int(lc.get("early_stopping_rounds", 50)), verbose=False)])
        fit_s = time.perf_counter() - t0
        del X, dtr, dva
        # decision rule chosen on the inner split only, WITH all train S1 competing
        s1a, poola, proba, _ = predict_all(cfg, paths, "train", booster, names)
        ya = np.isin(s1a * SHIFT + poola, gtc)
        m = P["stage2_inner"]
        tc = c.get("thresholds", {"start": 0.30, "stop": 0.90, "step": 0.01})
        grid = np.round(np.arange(tc["start"], tc["stop"] + 1e-9, tc["step"]), 4)
        table, best = {}, None
        for variant in ("threshold", "threshold+D01"):
            for t in grid:
                acc = decide(s1a, poola, proba, t, variant)
                k = np.bincount(s1a[acc], minlength=n)
                tp = np.bincount(s1a[acc & ya], minlength=n)
                f = evaluate_arrays(tp[m], k[m], n_true[m])
                table[f"{variant}@{t:.2f}"] = {"f05": f["f05"], "P": f["macro_precision"], "R": f["macro_recall"]}
                key = (round(f["f05"], 6), variant == "threshold+D01", -abs(t - 0.5))
                if best is None or key > best[0]:
                    best = (key, variant, float(t))
    run_dir = paths.run_dir(cfg["experiment"]["id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(run_dir / "stage2_model.lgb.txt"))
    gain = booster.feature_importance("gain")
    meta = {"features": names, "variant": best[1], "threshold": best[2], "inner_f05": best[0][0], "best_iteration": booster.best_iteration,
            "params": lgb_params(c), "fit_seconds": round(fit_s, 1), "train_pairs": int(tr.sum()), "inner_pairs": int(va.sum()),
            "train_s1": int(P["stage2_train"].sum()), "inner_s1": int(P["stage2_inner"].sum()), "peak_rss_gb": round(pm.peak_gb, 2),
            "stage1_run": c["stage1_run"], "stage1_threshold": stage1_threshold(cfg), "min_score": c["min_score"], "peers": c["peers"],
            "training_partition": "E half A minus frozen validation (E = train S1 not in the stage-1 fit set)",
            "feature_importance_gain": dict(zip(names, (gain / max(gain.sum(), 1e-12)).round(5).tolist())), "inner_table": table}
    write_json(run_dir / "stage2_model.json", meta)
    LOG.info("STAGE2 FIT: %d pairs, best_iter %d, decision %s @ %.2f, inner F0.5 %.4f, %.0fs, peak %.2f GB", meta["train_pairs"],
             booster.best_iteration, best[1], best[2], best[0][0], fit_s, pm.peak_gb)
    return meta


def load_stage2(cfg, paths):
    run_dir = model_dir(cfg, paths)
    meta = json.loads((run_dir / "stage2_model.json").read_text(encoding="utf-8"))
    return lgb.Booster(model_file=str(run_dir / "stage2_model.lgb.txt")), meta


def report(tp, k, n_true, countries, m):
    r = evaluate_arrays(tp[m], k[m], n_true[m], countries=countries[m])
    out = {x: r[x] for x in KEYS}
    out["empty_rate"] = float((k[m] == 0).mean()) if m.any() else None
    out["by_country"] = {c: {x: v[x] for x in KEYS} for c, v in r["by_country"].items()}
    out["by_match_count"] = {b: {x: v[x] for x in ("entities", "f05", "macro_precision", "macro_recall")} for b, v in r["by_match_count"].items()}
    return out


def run_evaluate(cfg, paths) -> dict:
    booster, meta = load_stage2(cfg, paths)
    P, n = populations(cfg, paths)
    gtc, n_true = gt_codes(paths)
    countries = np.empty(n, dtype=object)
    for key in load_manifest(paths, "train")["source1"]["countries"]:
        countries[read_country(paths, "train", "source1", key, ["row"]).column("row").to_numpy()] = key
    floor = add_floor(cfg)
    t0 = time.perf_counter()
    with PeakMemory() as pm:
        # control first (stage-1 threshold + D-01 over ALL stage-1 pairs; pairs < min_score are never accepted
        # at t >= min_score); only the accepted pairs are kept, the 40M+ pair arrays are released (peak <= 5 GB)
        code1, score1, conflicts_control, nc = control_decisions(cfg, paths, "train")
        n_cand = np.zeros(n, np.int64)
        n_cand[: len(nc)] = nc
        cy = np.isin(code1, gtc)
        k1, tp1 = np.bincount(code1 // SHIFT, minlength=n), np.bincount((code1 // SHIFT)[cy], minlength=n)
        s1, pool, prob, sc = predict_all(cfg, paths, "train", booster, meta["features"])
        y = np.isin(s1 * SHIFT + pool, gtc)
        acc2 = final_decisions(cfg, s1, pool, prob, sc, meta, code1)
        k2, tp2 = np.bincount(s1[acc2], minlength=n), np.bincount(s1[acc2 & y], minlength=n)
        if floor is not None:  # reference: the unrestricted stage-2 decisions of the re-used model
            acc0 = decide(s1, pool, prob, meta["threshold"], meta["variant"])
            k0, tp0 = np.bincount(s1[acc0], minlength=n), np.bincount(s1[acc0 & y], minlength=n)
            del acc0
        ctl = np.isin(s1 * SHIFT + pool, code1)
        add, rem = acc2 & ~ctl, ctl & ~acc2
        n_add, n_add_t = np.bincount(s1[add], minlength=n), np.bincount(s1[add & y], minlength=n)
        n_rem, n_rem_t = np.bincount(s1[rem], minlength=n), np.bincount(s1[rem & y], minlength=n)
        n_elig = np.bincount(s1, minlength=n)
        del ctl, add, rem
    res = {}
    for pname in ("E", "E_half_A", "E_half_B", "validation", "validation_half_A", "validation_half_B", "stage2_train", "stage2_inner"):
        m = P[pname]
        a, b = report(tp1, k1, n_true, countries, m), report(tp2, k2, n_true, countries, m)
        ns = max(int(m.sum()), 1)
        res[pname] = {"control": a, "stage2": b, "delta_f05": b["f05"] - a["f05"], "delta_precision": b["macro_precision"] - a["macro_precision"],
                      "vs_control": {"additions_per_s1": float(n_add[m].sum() / ns), "added_true_share": float(n_add_t[m].sum() / max(n_add[m].sum(), 1)),
                                     "removals_per_s1": float(n_rem[m].sum() / ns), "removed_true_share": float(n_rem_t[m].sum() / max(n_rem[m].sum(), 1)),
                                     "candidates_per_s1": float(n_cand[m].sum() / ns), "eligible_per_s1": float(n_elig[m].sum() / ns)}}
        if floor is not None:
            res[pname]["stage2_unrestricted"] = report(tp0, k0, n_true, countries, m)
    # flips on held-out E half B: what stage 2 changed relative to the control
    code2 = s1[acc2] * SHIFT + pool[acc2]
    hb = P["E_half_B"]
    only2 = np.setdiff1d(code2[hb[s1[acc2]]], code1[hb[code1 // SHIFT]])
    only1 = np.setdiff1d(code1[hb[code1 // SHIFT]], code2[hb[s1[acc2]]])
    tset = gtc
    flips = {"added_pairs": int(len(only2)), "added_true": int(np.isin(only2, tset).sum()),
             "removed_pairs": int(len(only1)), "removed_true": int(np.isin(only1, tset).sum())}
    conflicts = {"control": conflicts_control, "stage2": claim_stats(pool, s1, acc2)}
    q = lambda x: {p: float(np.quantile(x, p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)} if len(x) else None  # noqa: E731
    dist = {"stage1_score_of_added_pairs": q(sc[np.isin(s1 * SHIFT + pool, only2)]),
            "stage1_score_of_removed_pairs": q(score1[np.isin(code1, only1)])}
    out = {"stage2_model": meta | {"inner_table": None}, "decision": decision_record(cfg, paths, meta), "results": res, "flips_E_half_B": flips,
           "conflicts_all_train": conflicts, "score_distribution": dist, "eval_peak_rss_gb": round(pm.peak_gb, 2),
           "eval_seconds": round(time.perf_counter() - t0, 1)}
    write_json(paths.run_dir(cfg["experiment"]["id"]) / "stage2_eval.json", out)
    for pname in ("E_half_B", "validation", "validation_half_A", "validation_half_B", "E_half_A", "E"):
        r = res[pname]
        LOG.info("%-18s control %.4f (P %.4f R %.4f) -> stage2 %.4f (P %.4f R %.4f) | delta %+.4f | FM %d -> %d | sFP %d -> %d", pname,
                 r["control"]["f05"], r["control"]["macro_precision"], r["control"]["macro_recall"], r["stage2"]["f05"],
                 r["stage2"]["macro_precision"], r["stage2"]["macro_recall"], r["delta_f05"], r["control"]["false_merges"],
                 r["stage2"]["false_merges"], r["control"]["singleton_false_positive"], r["stage2"]["singleton_false_positive"])
    LOG.info("flips E_B %s", flips)
    LOG.info("decision %s | vs control (E_half_B) %s | %.0fs, peak %.2f GB", out["decision"], res["E_half_B"]["vs_control"], out["eval_seconds"], pm.peak_gb)
    return out


def run_submit(cfg, paths) -> dict:
    floor = add_floor(cfg)
    ctl_codes = control_decisions(cfg, paths, "test")[0] if floor is not None else None  # before stage 2 is loaded (memory)
    gc.collect()
    booster, meta = load_stage2(cfg, paths)
    s1, pool, prob, sc = predict_all(cfg, paths, "test", booster, meta["features"])
    acc = final_decisions(cfg, s1, pool, prob, sc, meta, ctl_codes)
    acc_codes = np.sort(s1[acc] * SHIFT + pool[acc])
    vs_control = None
    if ctl_codes is not None:
        vs_control = {"control_pairs": int(len(ctl_codes)), "final_pairs": int(len(acc_codes)),
                      "additions": int(len(np.setdiff1d(acc_codes, ctl_codes))), "removals": int(len(np.setdiff1d(ctl_codes, acc_codes)))}
        if np.isinf(floor) and vs_control["additions"]:
            raise RuntimeError("veto-only stage 2 produced additions over the control")
    del s1, pool, prob, sc, acc, booster, ctl_codes  # keep the submit peak low (<= 5 GB)
    gc.collect()
    s1_ids = _ids(paths, "test", "source1").to_pylist()
    ids = pool_ids(paths, "test")
    sdir = stage1_parts(cfg, paths, "test")
    sc = _read_parts(sdir, ["s1_row", "pool_row", "cheap_rank"])
    key = sc["s1_row"].astype(np.int64) * 65536 + sc["cheap_rank"].astype(np.int64)
    order = np.argsort(key, kind="stable")
    del key
    cs1 = sc["s1_row"][order].astype(np.int64)
    cpool = sc["pool_row"][order].astype(np.int64)
    del sc, order
    # membership in the sorted, unique accepted codes via binary search (np.isin would sort ~40M keys: submit peak > 5 GB)
    code = cs1 * SHIFT + cpool
    pos = np.minimum(np.searchsorted(acc_codes, code), max(len(acc_codes) - 1, 0))
    match = (acc_codes[pos] == code) if len(acc_codes) else np.zeros(len(code), bool)
    del code, pos
    if int(match.sum()) != len(acc_codes):
        raise RuntimeError("stage-2 accepted a pair outside the stage-1 candidate set")
    out = paths.output
    with stage("write candidate_pairs.tsv"):
        c_stats = write_id_lists(out / "candidate_pairs.tsv", CANDIDATE_HEADER, s1_ids, cs1, (cpool, ids))
    with stage("write matching_results.tsv"):
        m_stats = write_id_lists(out / "matching_results.tsv", MATCHING_HEADER, s1_ids, cs1[match], (cpool[match], ids))
    summary = {"stage2_decision": {"variant": meta["variant"], "threshold": meta["threshold"]}, "stage1_parts": str(sdir),
               "candidate_file": c_stats, "matching_file": m_stats, "competition_after": claim_stats(cpool, cs1, match)}
    if floor is not None:
        summary["decision"] = decision_record(cfg, paths, meta)
        summary["vs_control_test"] = vs_control
    del cs1, cpool, match, acc_codes, ids
    gc.collect()
    with stage("strict validation"):
        errors, warnings, _ = strict_validate(out / "matching_results.tsv", out / "candidate_pairs.tsv", paths.raw / "test", None, sdir)
    for w in warnings:
        LOG.warning(w)
    summary["strict_validation_errors"] = errors
    write_json(paths.run_dir(cfg["experiment"]["id"]) / "submission_summary.json", summary)
    if errors:
        for e in errors:
            LOG.error(e)
        raise SystemExit("strict validation FAILED")
    LOG.info("strict validation PASS: %s", summary)
    return summary


STAGES = ["features", "fit", "evaluate", "submit", "all"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--set", dest="sets", action="append", default=[])
    a = ap.parse_args(argv)
    setup_logging()
    configure_arrow()
    cfg = load_config(a.config, parse_sets(a.sets))
    paths = Paths(cfg)
    if a.stage in ("features", "all"):
        for split in (["train", "test"] if a.split == "both" else [a.split]):
            with stage(f"stage2 features {split}"), PeakMemory() as pm:
                st = build_features(cfg, paths, split)
            LOG.info("features %s: %d pairs, %.0fs, peak %.2f GB", split, st["eligible_pairs"], pm.seconds, pm.peak_gb)
    if a.stage in ("fit", "all"):
        run_fit(cfg, paths)
    if a.stage in ("evaluate", "all"):
        run_evaluate(cfg, paths)
    if a.stage in ("submit", "all"):
        run_submit(cfg, paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
