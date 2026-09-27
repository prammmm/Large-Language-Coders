"""Entity-resolution pipeline: blocking -> pair features -> matcher -> global threshold.

Stages (run from the repo root, inside the ber_hackathon env):
    preprocess  raw TSV -> data/interim (Parquet) -> data/processed (normalized views, per country)
    split       frozen validation split -> data/splits/val_s1_ids.txt (never regenerated silently)
    block-eval  blocking ONLY for the validation S1s: recall / candidate cost / per-pass value
    fit         (matcher.type=logistic) fit the learned matcher on TRAIN-fold S1s only; the
                threshold is tuned on an inner split of those train-fold S1s
    train       block + score the validation S1s against the FULL train pool, then evaluate
    evaluate    re-evaluate the stored validation features with the configured matcher
                (no re-blocking: rule vs model on identical candidates)
    infer       block + score every test S1
    submit      threshold -> output/candidate_pairs.tsv + output/matching_results.tsv -> strict validation
    all         preprocess, split, (fit), train, infer, submit

    python -m ber.pipeline.run_baseline <stage> --config configs/<file>.yaml [--set key=value ...]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ber.blocking.blocker import PASS_ORDER, CountryBlocker
from ber.config import Paths, load_config, parse_sets
from ber.data.ingest import ingest_split
from ber.data.normalize_records import country_key, load_manifest, normalize_split, read_country
from ber.data.split import load_val_ids, make_split, match_counts
from ber.decision.competition import best_claimant_keep, claim_stats
from ber.normalization.crossscript import XlitMap
from ber.pipeline.xlit import crossfit_plan, ensure_final_map, final_map_path, learn_map_for, xlit_enabled
from ber.decision.threshold import apply_global_threshold, resolve_threshold, threshold_source
from ber.evaluation.metrics import evaluate_arrays, threshold_sweep
from ber.features.pair_features import FEATURE_NAMES, pair_features
from ber.models import make_scorer, model_path, scorer_features
from ber.models.logistic import LogisticModel, feature_matrix
from ber.submission.strict_validator import validate as strict_validate
from ber.submission.writer import CANDIDATE_HEADER, MATCHING_HEADER, write_id_lists
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, stage, write_json

VIEW_COLS = ["row", "entity_id", "name_norm", "name_core", "name_concat", "addr_norm", "addr_nums", "addr_words"]
FEATURE_COLS = ["name_norm", "name_core", "name_concat", "addr_norm", "addr_nums", "addr_words"]
PAIR_META = ["cheap_score", "cheap_rank"]


class LazyPool:
    """Country pool (source2 rows, then source3 rows) whose columns are read from Parquet
    only when the blocker asks for them, so at most one string column is resident."""

    def __init__(self, paths: Paths, split: str, key: str, xmap=None):
        self.paths, self.split, self.key = paths, split, key
        self.xmap = xmap  # EXP-N-02: learned cross-script map -> derived "name_xlit" column
        self._xlit = None
        m = load_manifest(paths, split)
        self.num_rows = sum(m[s]["countries"].get(key, {}).get("rows", 0) for s in ("source2", "source3"))

    def column(self, col: str) -> pa.ChunkedArray:
        if col == "name_xlit":
            if self.xmap is None:
                raise KeyError("name_xlit requested but no cross-script map was given")
            if self._xlit is None:  # computed once per country, from name_core
                self._xlit = pa.chunked_array([self.xmap.apply(self.column("name_core"))])
            return self._xlit
        chunks = []
        for source in ("source2", "source3"):
            chunks += read_country(self.paths, self.split, source, self.key, [col]).column(col).chunks
        return pa.chunked_array(chunks, pa.string()) if chunks else pa.chunked_array([], pa.string())


def _read_pool(paths: Paths, split: str, key: str, columns: list[str]) -> pa.Table:
    """Country pool = source2 rows then source3 rows, with a ``_src`` marker column."""
    parts = []
    for src, source in ((2, "source2"), (3, "source3")):
        t = read_country(paths, split, source, key, columns)
        parts.append(t.append_column("_src", pa.array(np.full(t.num_rows, src, np.int8))))
    return pa.concat_tables(parts)


# ----------------------------------------------------------------------------- helpers
def _ids(paths: Paths, split: str, source: str) -> pa.Array:
    return pq.read_table(paths.interim_file(split, source), columns=["entity_id"]).column("entity_id").combine_chunks()


def pool_ids(paths: Paths, split: str) -> pa.Array:
    """Global pool index space: all source2 rows, then all source3 rows."""
    return pa.concat_arrays([_ids(paths, split, "source2"), _ids(paths, split, "source3")])


def eval_s1_subset(cfg: dict, paths: Paths) -> list[str]:
    """Validation S1 ids used by `train`/`block-eval` (optionally a deterministic sample)."""
    ids = load_val_ids(paths)
    cap = cfg["train"].get("max_val_s1")
    if cap and cap < len(ids):
        rng = np.random.default_rng(cfg["seed"])
        ids = sorted(np.asarray(ids, dtype=object)[rng.choice(len(ids), int(cap), replace=False)].tolist())
    return ids


def fit_s1_subset(cfg: dict, paths: Paths) -> list[str]:
    """TRAIN-fold S1 ids for `fit`: every train S1 NOT in the frozen validation split,
    optionally a seeded sample. Validation S1s can never enter this list."""
    val = set(load_val_ids(paths))
    ids = [i for i in _ids(paths, "train", "source1").to_pylist() if i not in val]
    only = cfg["fit"].get("only_countries")  # diagnostic (leave-one-country-out); default None = all countries
    if only:
        s1c = pq.read_table(paths.interim_file("train", "source1"), columns=["entity_id", "country"])
        keep = {e for e, c in zip(s1c.column("entity_id").to_pylist(), s1c.column("country").to_pylist()) if country_key(c) in set(only)}
        ids = [i for i in ids if i in keep]
    cap = cfg["fit"].get("max_train_s1")
    if cap and cap < len(ids):
        rng = np.random.default_rng(cfg["seed"] + 1)
        ids = np.asarray(ids, dtype=object)[rng.choice(len(ids), int(cap), replace=False)].tolist()
    ids = sorted(ids)
    overlap = val.intersection(ids)
    if overlap:  # defensive: must be impossible
        raise RuntimeError(f"leakage: {len(overlap)} validation S1 in the fit set")
    return ids


def scored_dir(cfg: dict, paths: Paths, split: str, purpose: str = "eval") -> Path:
    """purpose: "eval" (this experiment's parts), "fit" (train-fold fit parts),
    "eval_read" (validation parts the `evaluate` stage reads; ``scoring.eval_source_tag``
    lets an experiment re-score another experiment's stored candidates read-only)."""
    s = cfg.get("scoring", {})
    tag = s.get("tag", "scored")
    if purpose == "fit":
        return paths.scored_dir(split, f"{tag}_fit")
    if purpose == "eval_read":
        return paths.scored_dir(split, s.get("eval_source_tag") or tag)
    return paths.scored_dir(split, tag)


def feature_options(cfg: dict) -> dict:
    f = cfg.get("features", {}) or {}
    return {"strip_leading_zeros": bool(f.get("strip_leading_zeros", False)), "extra": list(f.get("extra", []) or [])}


def feature_columns(cfg: dict) -> list[str]:
    return list(FEATURE_NAMES) + feature_options(cfg)["extra"]


def _clear_dir(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def _hash_frac(seed: int, s: str) -> float:
    return int(hashlib.sha1(f"{seed}|{s}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def label_pairs(paths: Paths, s1_ids: list[str], s1_row: np.ndarray, pool_row: np.ndarray):
    """Labels for TRAIN candidate pairs of the given S1 set, from the ground truth only.

    Returns (is_true per pair, S1 position per pair, n_true per S1 incl. blocking misses,
    country per S1)."""
    arr = pa.array(s1_ids, pa.string())
    m = len(s1_ids)
    gt = pq.read_table(paths.interim_file("train", "ground_truth"))
    gt = gt.filter(pc.is_in(gt.column("source1_entity_id"), value_set=arr))
    n_true = np.zeros(m, np.int64)
    n_true[pc.index_in(gt.column("source1_entity_id"), value_set=arr).to_numpy(zero_copy_only=False)] = match_counts(
        gt.column("matched_entity_ids")
    )
    lists = pc.split_pattern(gt.column("matched_entity_ids"), ",")
    parent, flat = pc.list_parent_indices(lists), pc.list_flatten(lists)
    keep = pc.not_equal(flat, "")
    gt_pairs = pc.binary_join_element_wise(
        pc.take(gt.column("source1_entity_id"), pc.filter(parent, keep)), pc.filter(flat, keep), "|"
    ).combine_chunks()
    s1_str = _ids(paths, "train", "source1").take(pa.array(np.asarray(s1_row, np.int64)))
    cand_str = pool_ids(paths, "train").take(pa.array(np.asarray(pool_row, np.int64)))
    is_true = pc.is_in(pc.binary_join_element_wise(s1_str, cand_str, "|"), value_set=gt_pairs).to_numpy(zero_copy_only=False)
    pos = pc.index_in(s1_str, value_set=arr).to_numpy(zero_copy_only=False).astype(np.int64)
    s1_tab = pq.read_table(paths.interim_file("train", "source1"), columns=["entity_id", "country"])
    s1_tab = s1_tab.filter(pc.is_in(s1_tab.column("entity_id"), value_set=arr))
    cpos = pc.index_in(s1_tab.column("entity_id"), value_set=arr).to_numpy(zero_copy_only=False)
    countries = np.empty(m, dtype=object)
    countries[cpos] = [country_key(c) for c in s1_tab.column("country").to_pylist()]
    return is_true, pos, n_true, countries


def _grid(cfg: dict) -> np.ndarray:
    tc = cfg["tuning"]["thresholds"]
    return np.round(np.arange(tc["start"], tc["stop"] + 1e-9, tc["step"]), 6)


# ----------------------------------------------------------------------------- stages
def run_preprocess(cfg: dict, paths: Paths, force: bool) -> None:
    pcfg = cfg["preprocess"]
    for split in pcfg["splits"]:
        with stage(f"ingest {split}"):
            ingest_split(paths, split, int(pcfg["read_block_bytes"]), force)
        with stage(f"normalize {split}"):
            normalize_split(paths, split, int(pcfg["batch_rows"]), int(pcfg["n_workers"]), force)


def run_split(cfg: dict, paths: Paths, force: bool) -> None:
    with stage("split"):
        make_split(paths, float(cfg["split"]["val_fraction"]), int(cfg["seed"]), force)


def with_xlit(s1: pa.Table, xmap) -> pa.Table:
    """Add the learned cross-script view "name_xlit" (EXP-N-02) to an S1 table."""
    return s1 if xmap is None else s1.append_column("name_xlit", xmap.apply(s1.column("name_core")))


def resolve_xmap(cfg: dict, paths: Paths, xmap="auto"):
    """``"auto"``: the final (fit-set-learned) map when xlit is enabled, else None."""
    if isinstance(xmap, str) and xmap == "auto":
        return ensure_final_map(cfg, paths) if xlit_enabled(cfg) else None
    return xmap


def _country_setup(cfg, paths, split, key, subset_arr, n_s2, xmap=None):
    s1 = read_country(paths, split, "source1", key, VIEW_COLS)
    if subset_arr is not None:
        s1 = s1.filter(pc.is_in(s1.column("entity_id"), value_set=subset_arr))
    if s1.num_rows == 0:
        return None
    s1 = with_xlit(s1, xmap)
    ids = _read_pool(paths, split, key, ["row"])
    pool_row = ids.column("row").to_numpy().astype(np.int64)
    pool_row[ids.column("_src").to_numpy() == 3] += n_s2
    return s1, s1.column("row").to_numpy().astype(np.int64), pool_row


def score_split(cfg: dict, paths: Paths, split: str, s1_subset, out_dir: Path, scorer, store_features: bool,
                xmap="auto", clear: bool = True, part_prefix: str = "part") -> dict:
    """Blocking + features + scores for every S1 (or the subset), country by country.

    ``xmap``: EXP-N-02 cross-script map ("auto" = the final fit-set map when enabled; `fit` passes
    out-of-fold maps). ``clear``/``part_prefix`` let `fit` write several cross-fit folds into one dir."""
    xmap = resolve_xmap(cfg, paths, xmap)
    if clear:
        _clear_dir(out_dir)  # never mix parts from different runs
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(paths, split)
    n_s2 = int(manifest["source2"]["rows"])
    s1_countries = sorted(manifest["source1"]["countries"])
    orphan = sorted((set(manifest["source2"]["countries"]) | set(manifest["source3"]["countries"])) - set(s1_countries))
    if orphan:
        LOG.info("pool-only countries (no S1, ignored): %s", orphan)
    scfg = cfg["scoring"]
    fopts = feature_options(cfg)
    fcols = feature_columns(cfg)
    subset_arr = pa.array(s1_subset, pa.string()) if s1_subset is not None else None
    stats: dict = {"split": split, "countries": {}, "top_k": cfg["blocking"]["top_k"], "out_dir": str(out_dir)}
    part_no = 0

    for key in s1_countries:
        with stage(f"score {split}/{key!r}"):
            setup = _country_setup(cfg, paths, split, key, subset_arr, n_s2, xmap)
            if setup is None:
                continue
            s1, s1_row, pool_row = setup
            cstats = {"n_s1": s1.num_rows, "n_pool": len(pool_row), "candidate_pairs": 0, "pairs_before_cap": 0, "chunks": 0}
            if len(pool_row) == 0:
                LOG.warning("country %r has no S2/S3 records: all its S1 get empty lists", key)
                stats["countries"][key] = cstats
                continue
            lazy_pool = LazyPool(paths, split, key, xmap)
            blocker = CountryBlocker(s1, lazy_pool, cfg["blocking"])
            gather_cols = FEATURE_COLS + (["name_xlit"] if xmap is not None else [])
            for lo, hi in blocker.chunk_bounds():
                cand = blocker.candidates(lo, hi)
                n = len(cand["s1_local"])
                cstats["chunks"] += 1
                cstats["pairs_before_cap"] += int(cand["pairs_before_cap"])
                cstats["candidate_pairs"] += n
                if n == 0:
                    continue
                # gather only this chunk's unique candidate rows, one pool column at a time
                uniq_rows, cand_pos = np.unique(cand["pool_local"], return_inverse=True)
                take_idx = pa.array(uniq_rows)
                gathered = pa.table({c: lazy_pool.column(c).take(take_idx) for c in gather_cols})
                del take_idx, uniq_rows
                score = np.empty(n, np.float32)
                feats_all = {f: np.empty(n, np.float32) for f in fcols} if store_features else None
                step = int(scfg["feature_batch_pairs"])
                for a in range(0, n, step):
                    b = min(a + step, n)
                    feats = pair_features(s1, gathered, cand["s1_local"][a:b], cand_pos[a:b], int(scfg["rapidfuzz_workers"]), fopts)
                    feats["cheap_score"] = cand["cheap_score"][a:b].astype(np.float32)
                    feats["cheap_rank"] = cand["cheap_rank"][a:b].astype(np.float32)
                    score[a:b] = scorer.score(feats)
                    if feats_all is not None:
                        for f in fcols:
                            feats_all[f][a:b] = feats[f]
                cols = {
                    "s1_row": pa.array(s1_row[cand["s1_local"]].astype(np.int32)),
                    "pool_row": pa.array(pool_row[cand["pool_local"]].astype(np.int32)),
                    "cheap_score": pa.array(cand["cheap_score"]),
                    "cheap_rank": pa.array(cand["cheap_rank"]),
                    "pass_flags": pa.array(cand["pass_flags"]),
                    "score": pa.array(score),
                }
                if feats_all is not None:
                    cols.update({f: pa.array(v) for f, v in feats_all.items()})
                pq.write_table(pa.table(cols), out_dir / f"{part_prefix}-{part_no:05d}.parquet", compression="zstd")
                part_no += 1
                del gathered, cand_pos, cols, cand, feats_all, score
            cstats["blocking"] = blocker.stats
            cstats["candidates_per_s1_mean"] = cstats["candidate_pairs"] / max(cstats["n_s1"], 1)
            stats["countries"][key] = cstats
            del blocker, lazy_pool, s1
    stats["candidate_pairs"] = sum(c["candidate_pairs"] for c in stats["countries"].values())
    stats["evaluated_s1"] = sum(c["n_s1"] for c in stats["countries"].values())
    if xmap is not None:
        stats["xlit_map"] = {k: xmap.meta.get(k) for k in ("label", "learned_from_s1", "learned_from_digest", "mappings")}
    if clear:
        write_json(out_dir.parent / f"{out_dir.name}_scoring_stats.json", stats)
    return stats


def _read_parts(d: Path, columns: list[str]) -> dict[str, np.ndarray]:
    parts = sorted(Path(d).glob("*.parquet"))
    if not parts:
        return {c: np.zeros(0, np.float32) for c in columns}
    t = pa.concat_tables([pq.read_table(p, columns=columns) for p in parts])
    return {c: t.column(c).to_numpy() for c in columns}


def _available_columns(d: Path) -> list[str]:
    parts = sorted(Path(d).glob("*.parquet"))
    return pq.ParquetFile(parts[0]).schema_arrow.names if parts else []


def _dist(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0}
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "max": int(x.max())}


# ----------------------------------------------------------------------------- block-eval
def run_block_eval(cfg: dict, paths: Paths) -> dict:
    """Blocking only (no features, no matcher) for the validation S1s: recall vs cost."""
    exp = cfg["experiment"]["id"]
    val_ids = eval_s1_subset(cfg, paths)
    subset_arr = pa.array(val_ids, pa.string())
    xmap = resolve_xmap(cfg, paths)
    manifest = load_manifest(paths, "train")
    n_s2 = int(manifest["source2"]["rows"])
    rows, pools, ranks, flags, pass_stats = [], [], [], [], {}
    t0 = time.perf_counter()
    with PeakMemory() as pm:
        for key in sorted(manifest["source1"]["countries"]):
            with stage(f"block-eval {key!r}"):
                setup = _country_setup(cfg, paths, "train", key, subset_arr, n_s2, xmap)
                if setup is None:
                    continue
                s1, s1_row, pool_row = setup
                b = CountryBlocker(s1, LazyPool(paths, "train", key, xmap), cfg["blocking"])
                pass_stats[key] = b.stats["passes"]
                for lo, hi in b.chunk_bounds():
                    c = b.candidates(lo, hi)
                    rows.append(s1_row[c["s1_local"]])
                    pools.append(pool_row[c["pool_local"]])
                    ranks.append(c["cheap_rank"].astype(np.int32))
                    flags.append(c["pass_flags"])
                del b, s1
    seconds = time.perf_counter() - t0
    s1_row = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    pool_row = np.concatenate(pools) if pools else np.zeros(0, np.int64)
    rank = np.concatenate(ranks) if ranks else np.zeros(0, np.int32)
    flag = np.concatenate(flags) if flags else np.zeros(0, np.uint8)
    is_true, pos, n_true, countries = label_pairs(paths, val_ids, s1_row, pool_row)
    m = len(val_ids)
    total_true = int(n_true.sum())
    ks = sorted({k for k in cfg["tuning"].get("block_eval_k", [10, 20, 30, 50]) if k <= cfg["blocking"]["top_k"]} | {cfg["blocking"]["top_k"]})
    at_k = {}
    for k in ks:
        sel = rank < k
        c_n = np.bincount(pos[sel], minlength=m)
        c_tp = np.bincount(pos[sel & is_true], minlength=m)
        by_c = {}
        for c in sorted(set(countries.tolist())):
            cs = countries == c
            by_c[c] = {"blocking_recall": float(c_tp[cs].sum() / max(n_true[cs].sum(), 1)),
                       "candidates_per_s1": _dist(c_n[cs])}
        at_k[str(k)] = {
            "blocking_recall": float(c_tp.sum() / max(total_true, 1)),
            "true_pairs_found": int(c_tp.sum()),
            "candidate_pairs": int(sel.sum()),
            "candidate_precision": float(c_tp.sum() / max(sel.sum(), 1)),  # share of candidates that are true
            "candidates_per_s1": _dist(c_n),
            "s1_with_all_true_found": float(((c_tp == n_true) & (n_true > 0)).sum() / max((n_true > 0).sum(), 1)),
            "by_country": by_c,
        }
    k = cfg["blocking"]["top_k"]
    sel = (rank < k) & is_true
    contrib = {}
    for i, name in enumerate(PASS_ORDER):
        bit = np.uint8(1 << i)
        has = (flag & bit) > 0
        if not has.any():
            continue
        contrib[name] = {
            "true_pairs_with_this_key": int((sel & has).sum()),
            "true_pairs_only_this_key": int((sel & (flag == bit)).sum()),
            "candidates_with_this_key": int(((rank < k) & has).sum()),
        }
    report = {
        "experiment": exp,
        "evaluated_s1": m,
        "true_pairs": total_true,
        "top_k": k,
        "blocking_config": cfg["blocking"],
        "at_k": at_k,
        "pass_contribution_at_top_k": contrib,
        "pass_stats_by_country": pass_stats,
        "runtime_seconds": round(seconds, 1),
        "peak_rss_gb": round(pm.peak_gb, 2),
    }
    write_json(paths.run_dir(exp) / "blocking_eval.json", report)
    a = at_k[str(k)]
    LOG.info("BLOCK-EVAL %s (%d S1): recall@%d %.4f | cand/S1 mean %.2f p99 %.0f | %.0fs peak %.2f GB",
             exp, m, k, a["blocking_recall"], a["candidates_per_s1"]["mean"], a["candidates_per_s1"]["p99"], seconds, pm.peak_gb)
    return report


# ----------------------------------------------------------------------------- fit (learned matcher)
def run_fit(cfg: dict, paths: Paths) -> dict:
    """Fit the learned matcher (``matcher.type``: logistic | lightgbm) on TRAIN-fold S1s only.

    * candidates/features come from `score_split` on the fit S1s, or are re-used read-only
      from another experiment's fit parts (``fit.reuse_fit_parts``) -- then the S1 set is
      checked to be exactly inside the configured fit set;
    * the inner split (seeded hash of the S1 id) is used for the threshold (both models) and
      for LightGBM early stopping. Validation S1s are never touched (checked, raises)."""
    exp = cfg["experiment"]["id"]
    fcfg = cfg["fit"]
    kind = cfg["matcher"].get("type", "logistic")
    names = list(cfg["matcher"]["features"])
    fit_ids = fit_s1_subset(cfg, paths)
    reuse = fcfg.get("reuse_fit_parts")
    if reuse:
        out_dir = paths.scored_dir("train", reuse)
        if not any(out_dir.glob("*.parquet")):
            raise FileNotFoundError(f"fit.reuse_fit_parts: no parts in {out_dir}")
        sstats = {"reused_parts": str(out_dir)}
        LOG.info("fit: re-using train-fold candidates/features from %s (read-only)", out_dir)
    else:
        out_dir = scored_dir(cfg, paths, "train", "fit")
        rule = make_scorer(cfg, paths, kind="weighted_rule")  # placeholder score; features are what matter
        if xlit_enabled(cfg):
            # EXP-N-02 cross-fitting: each fold is scored with a map that never saw its own pairs,
            # and the inner-tune fold (0) never contributes to any map (see ber.pipeline.xlit)
            _clear_dir(out_dir)
            sstats = {"crossfit_folds": []}
            for f, ids_f, learn_f in crossfit_plan(cfg, fit_ids):
                if set(ids_f) & set(learn_f):
                    raise RuntimeError(f"xlit cross-fit: fold {f} would learn from its own S1")
                xm = learn_map_for(cfg, paths, learn_f, f"crossfit fold {f}")
                with stage(f"fit: block + features on train-fold S1, fold {f}"):
                    st = score_split(cfg, paths, "train", ids_f, out_dir, rule, True, xmap=xm, clear=False, part_prefix=f"fold{f}")
                sstats["crossfit_folds"].append({"fold": f, "scored_s1": len(ids_f), "map_learned_from_s1": len(learn_f),
                                                 "map_learned_from_digest": xm.meta["learned_from_digest"],
                                                 "mappings": xm.meta["mappings"], "candidate_pairs": st["candidate_pairs"]})
            ensure_final_map(cfg, paths)  # the map for validation / E / test scoring (fit set)
        else:
            with stage("fit: block + features on train-fold S1"):
                sstats = score_split(cfg, paths, "train", fit_ids, out_dir, rule, store_features=True)
    need = sorted(set(sum([n.split("*") for n in names], [])))
    d = _read_parts(out_dir, ["s1_row", "pool_row"] + need)

    # ---- leakage / consistency guards on the actual training rows
    row_ids = _ids(paths, "train", "source1").take(pa.array(d["s1_row"].astype(np.int64)))
    if pc.any(pc.is_in(row_ids, value_set=pa.array(load_val_ids(paths), pa.string()))).as_py():
        raise RuntimeError("leakage: validation S1 present in the fit candidate parts")
    in_fit = pc.is_in(row_ids, value_set=pa.array(fit_ids, pa.string())).to_numpy(zero_copy_only=False)
    if not in_fit.all():
        raise RuntimeError(f"fit parts contain {int((~in_fit).sum())} pairs of S1 outside the configured fit set")
    dev_n = fcfg.get("dev_subsample_s1")
    if dev_n and int(dev_n) < len(fit_ids):  # optional quick run: seeded subset of the fit S1s
        rng = np.random.default_rng(cfg["seed"] + 2)
        fit_ids = sorted(np.asarray(fit_ids, dtype=object)[rng.choice(len(fit_ids), int(dev_n), replace=False)].tolist())
        keep = pc.is_in(row_ids, value_set=pa.array(fit_ids, pa.string())).to_numpy(zero_copy_only=False)
        d = {k: v[keep] for k, v in d.items()}
        LOG.info("fit: dev subsample of %d train-fold S1 (%d pairs)", len(fit_ids), int(keep.sum()))
    LOG.info("fit %s on %d train-fold S1 (validation excluded)", kind, len(fit_ids))

    is_true, pos, n_true, _ = label_pairs(paths, fit_ids, d["s1_row"], d["pool_row"])
    tune_frac = float(fcfg.get("inner_tune_fraction", 0.2))
    ent_tune = np.array([_hash_frac(cfg["seed"], s) < tune_frac for s in fit_ids])
    pair_tune = ent_tune[pos]
    X = feature_matrix(d, names)
    with stage(f"fit: {kind}"):
        if kind == "lightgbm":
            from ber.models.gbdt import LightGBMModel

            lcfg = cfg.get("lightgbm", {}) or {}
            model = LightGBMModel(names, lcfg.get("params")).fit(
                X[~pair_tune], is_true[~pair_tune], X[pair_tune], is_true[pair_tune],
                num_boost_round=int(lcfg.get("num_boost_round", 1000)),
                early_stopping_rounds=int(lcfg.get("early_stopping_rounds", 50)))
            solver = "LightGBM gbdt, binary logloss; early stopping + threshold on the inner train-fold split"
        else:
            model = LogisticModel(names).fit(X[~pair_tune], is_true[~pair_tune], l2=float(fcfg.get("l2", 1e-4)),
                                             max_iter=int(fcfg.get("max_iter", 50)))
            solver = "Newton-IRLS (numpy), standardised features, L2 on weights, no class weights"
    # threshold on the inner train-fold tune split (entity-level F0.5)
    tpos = np.flatnonzero(ent_tune)
    remap = -np.ones(len(fit_ids), np.int64)
    remap[tpos] = np.arange(len(tpos))
    p = model.predict_proba_matrix(X[pair_tune])
    sweep = threshold_sweep(remap[pos[pair_tune]], is_true[pair_tune], p, n_true[ent_tune], _grid(cfg))
    best = max(sweep, key=lambda r: (round(r["f05"], 6), r["threshold"]))
    model.meta.update({
        "threshold": best["threshold"],
        "inner_tune": {k2: best[k2] for k2 in ("f05", "macro_precision", "macro_recall", "singleton_false_positive", "predicted_per_s1")},
        "fit_s1": len(fit_ids), "inner_tune_s1": int(ent_tune.sum()), "seed": cfg["seed"],
        "config": cfg.get("_config_path"), "blocking": cfg["blocking"], "features_options": feature_options(cfg),
        "solver": solver, "training_parts": str(out_dir),
    })
    if xlit_enabled(cfg):
        fm = ensure_final_map(cfg, paths)  # the fit-set map, or the pinned `xlit.map_path` (diagnostics)
        model.meta["xlit"] = {"params": fm.meta.get("params"), "final_map": str(cfg["xlit"].get("map_path") or final_map_path(cfg, paths)),
                              "final_map_learned_from_digest": fm.meta["learned_from_digest"], "final_map_mappings": fm.meta["mappings"],
                              "training_features": "cross-fitted out-of-fold maps (fold 0 = inner tune, excluded from all maps)"}
    model.save(model_path(cfg, paths))
    LOG.info("FIT %s [%s]: %d pairs (%d positive) | inner-tune threshold %.2f F0.5 %.4f P %.4f R %.4f | %.1fs",
             exp, kind, model.meta["n_train_pairs"], model.meta["positives"], best["threshold"], best["f05"],
             best["macro_precision"], best["macro_recall"], model.meta["fit_seconds"])
    saved = __import__("json").loads(model_path(cfg, paths).read_text(encoding="utf-8"))
    write_json(paths.run_dir(exp) / "fit_report.json", {"model": saved, "scoring_stats": sstats, "inner_tune_sweep": sweep})
    return model.meta


# ----------------------------------------------------------------------------- evaluation
def threshold_table(sweep: list[dict], chosen: list[float]) -> list[dict]:
    keep = {round(t, 2) for t in chosen}
    rows = []
    for r in sweep:
        t = round(r["threshold"], 2)
        if abs((t * 100) % 5) < 1e-6 or t in keep:
            rows.append({"threshold": t, "f05": r["f05"], "precision": r["macro_precision"], "recall": r["macro_recall"],
                         "singleton_fp": r["singleton_false_positive"], "predicted_per_s1": r["predicted_per_s1"],
                         "false_merges": r["false_merges"]})
    return rows


def evaluate_val(cfg: dict, paths: Paths, val_ids: list[str], scorer, parts_dir: Path) -> dict:
    """Precision-first evaluation of the validation candidates in ``parts_dir``."""
    exp = cfg["experiment"]["id"]
    run_dir = paths.run_dir(exp)
    m = len(val_ids)
    avail = _available_columns(parts_dir)
    need = scorer_features(scorer)
    rescore = all(c in avail for c in need)
    cols = ["s1_row", "pool_row", "cheap_rank", "score"] + ([c for c in need if c not in ("cheap_rank",)] if rescore else [])
    d = _read_parts(parts_dir, list(dict.fromkeys(cols)))
    score = scorer.score(d).astype(np.float32) if rescore else d["score"].astype(np.float32)
    is_true, pos, n_true, countries = label_pairs(paths, val_ids, d["s1_row"], d["pool_row"])
    rank = d["cheap_rank"].astype(np.int64)
    c_n = np.bincount(pos, minlength=m)
    c_tp = np.bincount(pos[is_true], minlength=m)
    sweep = threshold_sweep(pos, is_true, score, n_true, _grid(cfg))
    best = max(sweep, key=lambda r: (round(r["f05"], 6), r["threshold"]))
    t_used, t_src = resolve_threshold(cfg["decision"], run_dir / "tuned_threshold.json", model_path(cfg, paths))
    if threshold_source(cfg["decision"]) == "val_sweep":
        t_used, t_src = best["threshold"], "val_sweep (tuned on this validation set: optimistic)"

    def at(t):
        sel = apply_global_threshold(score, t)
        return evaluate_arrays(np.bincount(pos[sel & is_true], minlength=m), np.bincount(pos[sel], minlength=m),
                               n_true, c_tp, c_n, countries)

    empty = evaluate_arrays(np.zeros(m, np.int64), np.zeros(m, np.int64), n_true)
    report = {
        "experiment": exp,
        "matcher": type(scorer).__name__,
        "rescored_from_stored_features": rescore,
        "evaluated_s1": m,
        "all_empty_f05": empty["f05"],
        "blocking_recall_at_rank": {str(k): float(((rank < k) & is_true).sum() / max(n_true.sum(), 1))
                                     for k in cfg["tuning"]["recall_at_k"] if k <= cfg["blocking"]["top_k"]},
        "threshold_used": t_used,
        "threshold_source": t_src,
        "at_threshold_used": at(t_used),
        "val_optimal_threshold": best["threshold"],
        "at_val_optimal_threshold": at(best["threshold"]),
        "threshold_table": threshold_table(sweep, [t_used, best["threshold"]]),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "threshold_sweep.csv", "w", encoding="utf-8", newline="") as f:
        cols = ["threshold", "f05", "macro_precision", "macro_recall", "predicted_pairs", "tp_pairs", "false_merges",
                "singleton_false_positive", "empty_on_positive", "predicted_per_s1"]
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in sweep:
            w.writerow([r[c] for c in cols])
    write_json(run_dir / "tuned_threshold.json", {"threshold": best["threshold"], "f05": best["f05"], "evaluated_s1": m})
    u = report["at_threshold_used"]
    LOG.info("VAL %s [%s] (%d S1) t=%.2f (%s): F0.5 %.4f | P %.4f | R %.4f | false merges %d | singleton FP %d | "
             "pred/S1 %.2f | blocking recall %.4f | oracle %.4f",
             exp, report["matcher"], m, t_used, t_src, u["f05"], u["macro_precision"], u["macro_recall"], u["false_merges"],
             u["singleton_false_positive"], u["predicted_per_s1"], u["blocking_recall"], u["oracle_f05"])
    for c, v in u.get("by_country", {}).items():
        LOG.info("   %-8s F0.5 %.4f | P %.4f | R %.4f | singleton FP %d", c, v["f05"], v["macro_precision"], v["macro_recall"],
                 v["singleton_false_positive"])
    return report


def _snapshot(cfg: dict, paths: Paths) -> None:
    import yaml

    run_dir = paths.run_dir(cfg["experiment"]["id"])
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")}, f, sort_keys=False)


def run_train(cfg: dict, paths: Paths) -> dict:
    """Score validation S1s with the configured matcher, then evaluate (precision-first)."""
    val_ids = eval_s1_subset(cfg, paths)
    LOG.info("train/eval on %d validation S1 entities", len(val_ids))
    scorer = make_scorer(cfg, paths)
    out_dir = scored_dir(cfg, paths, "train")
    with stage("score train-val"), PeakMemory() as pm:
        sstats = score_split(cfg, paths, "train", val_ids, out_dir, scorer,
                             bool(cfg["scoring"].get("store_features", {}).get("train", True)))
    sstats.update({"runtime_seconds": round(pm.seconds, 1), "peak_rss_gb": round(pm.peak_gb, 2)})
    with stage("evaluate"):
        report = evaluate_val(cfg, paths, val_ids, scorer, out_dir)
    report["scoring_stats"] = sstats
    write_json(paths.run_dir(cfg["experiment"]["id"]) / "metrics.json", report)
    _snapshot(cfg, paths)
    return report


def run_evaluate(cfg: dict, paths: Paths) -> dict:
    """Evaluate stored validation parts (no re-blocking) with the configured matcher."""
    val_ids = eval_s1_subset(cfg, paths)
    out_dir = scored_dir(cfg, paths, "train", "eval_read")
    report = evaluate_val(cfg, paths, val_ids, make_scorer(cfg, paths), out_dir)
    report["parts_dir"] = str(out_dir)
    write_json(paths.run_dir(cfg["experiment"]["id"]) / "metrics.json", report)
    _snapshot(cfg, paths)
    return report


def run_infer(cfg: dict, paths: Paths) -> dict:
    with stage("score test"):
        return score_split(cfg, paths, "test", None, scored_dir(cfg, paths, "test"), make_scorer(cfg, paths),
                           bool(cfg["scoring"].get("store_features", {}).get("test", False)))


def run_submit(cfg: dict, paths: Paths) -> dict:
    exp = cfg["experiment"]["id"]
    threshold, source = resolve_threshold(cfg["decision"], paths.run_dir(exp) / "tuned_threshold.json", model_path(cfg, paths))
    s1_ids = _ids(paths, "test", "source1").to_pylist()
    ids = pool_ids(paths, "test")
    sdir = scored_dir(cfg, paths, "test")
    sc = _read_parts(sdir, ["s1_row", "pool_row", "cheap_rank", "score"])
    key = sc["s1_row"].astype(np.int64) * 65536 + sc["cheap_rank"].astype(np.int64)
    order = np.argsort(key, kind="stable")
    del key
    s1_row = sc["s1_row"][order].astype(np.int64)
    pool_row = sc["pool_row"][order]
    score = sc["score"][order]
    match = apply_global_threshold(score, threshold)
    del sc, order
    comp = cfg["decision"].get("competition") or {}
    comp_stats = None
    if comp.get("enabled", False):  # EXP-D-01: best claimant per pool record, over ALL test S1
        if comp.get("rule", "best_claimant") != "best_claimant":
            raise ValueError(f"decision.competition.rule: unknown rule {comp.get('rule')!r}")
        with stage("competition: best claimant per pool record"):
            before = claim_stats(pool_row, s1_row, match)
            match = best_claimant_keep(pool_row, s1_row, score, match, margin=float(comp.get("margin", 0.0)))
            comp_stats = {"rule": "best_claimant", "margin": float(comp.get("margin", 0.0)), "before": before,
                          "after": claim_stats(pool_row, s1_row, match)}
        LOG.info("competition: removed %d predicted pairs (%d -> %d); conflicted records %d -> %d",
                 before["predicted_pairs"] - comp_stats["after"]["predicted_pairs"], before["predicted_pairs"],
                 comp_stats["after"]["predicted_pairs"], before["conflicted_records"], comp_stats["after"]["conflicted_records"])
    del score
    out = paths.output
    with stage("write candidate_pairs.tsv"):
        c_stats = write_id_lists(out / "candidate_pairs.tsv", CANDIDATE_HEADER, s1_ids, s1_row, (pool_row, ids))
    with stage("write matching_results.tsv"):
        m_stats = write_id_lists(out / "matching_results.tsv", MATCHING_HEADER, s1_ids, s1_row[match], (pool_row[match], ids))
    summary = {"threshold": threshold, "threshold_source": source, "scored_dir": str(sdir), "candidate_file": c_stats,
               "matching_file": m_stats, "competition": comp_stats}
    LOG.info("wrote outputs: %s", summary)
    if cfg["submit"].get("strict_validate", True):
        with stage("strict validation"):
            errors, warnings, _ = strict_validate(out / "matching_results.tsv", out / "candidate_pairs.tsv", paths.raw / "test", None, sdir)
        for w in warnings:
            LOG.warning(w)
        summary["strict_validation_errors"] = errors
        if errors:
            for e in errors:
                LOG.error(e)
            write_json(paths.run_dir(exp) / "submission_summary.json", summary)
            raise SystemExit("strict validation FAILED")
        LOG.info("strict validation PASS")
    write_json(paths.run_dir(exp) / "submission_summary.json", summary)
    return summary


# ----------------------------------------------------------------------------- CLI
STAGES = ["preprocess", "split", "block-eval", "fit", "train", "evaluate", "infer", "submit", "all"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--config", default="configs/baseline_v0.yaml")
    ap.add_argument("--force", action="store_true", help="rebuild preprocess outputs / regenerate the split")
    ap.add_argument("--max-val-s1", type=int, default=None, help="evaluate on a deterministic sample of validation S1s")
    ap.add_argument("--set", dest="sets", action="append", default=[], help="config override, e.g. --set blocking.top_k=30")
    a = ap.parse_args(argv)
    setup_logging()
    configure_arrow()
    overrides = parse_sets(a.sets)
    if a.max_val_s1:
        overrides.setdefault("train", {})["max_val_s1"] = a.max_val_s1
    cfg = load_config(a.config, overrides)
    paths = Paths(cfg)
    allowed = cfg["experiment"].get("allowed_stages")  # decision-only experiments re-use another run's artefacts
    if allowed and a.stage not in allowed:
        raise SystemExit(f"stage {a.stage!r} is not allowed for {cfg['experiment']['id']} (allowed: {allowed})")
    if a.stage in ("preprocess", "all"):
        run_preprocess(cfg, paths, a.force)
    if a.stage == "split" or (a.stage == "all" and not paths.val_ids.exists()):
        run_split(cfg, paths, a.force)
    if a.stage == "block-eval":
        run_block_eval(cfg, paths)
    if a.stage == "fit" or (a.stage == "all" and cfg["matcher"].get("type") in ("logistic", "lightgbm")):
        run_fit(cfg, paths)
    if a.stage in ("train", "all"):
        run_train(cfg, paths)
    if a.stage == "evaluate":
        run_evaluate(cfg, paths)
    if a.stage in ("infer", "all"):
        run_infer(cfg, paths)
    if a.stage in ("submit", "all"):
        run_submit(cfg, paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
