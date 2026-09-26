"""EXP-N-02 plumbing: learn cross-script token maps from TRAIN true pairs with strict S1 separation.

* ``learn_map_for(ids)`` learns only from the ground-truth pairs of the given train S1 ids and
  raises ``LeakageError`` if any frozen-validation S1 is among them.
* Final map (validation / E / test scoring) = learned from the fit S1 set (``fit_s1_subset``).
* Cross-fitting for `fit`: the fit S1s are split into ``folds`` hash folds, fold 0 being EXACTLY
  the inner-tune split used for early stopping + threshold (same hash as ``_hash_frac``):
    fold 0 (inner tune)  -> map learned from folds 1..F-1          (inner-tune S1s contribute nothing)
    fold k >= 1          -> map learned from folds {1..F-1} \\ {k}  (out-of-fold, never fold 0)
  so no S1 ever has its own pairs in the map used for its features, and the inner-tune S1s never
  contribute to any map used in training or tuning.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ber.config import Paths
from ber.data.normalize_records import load_manifest
from ber.data.split import load_val_ids
from ber.normalization.crossscript import DEFAULT_PARAMS, DROP, LeakageError, XlitMap, learn_legal_tokens, learn_token_map
from ber.normalization.views import LEGAL_FORMS
from ber.utils.runtime import LOG


def xlit_cfg(cfg: dict) -> dict:
    return cfg.get("xlit", {}) or {}


def xlit_enabled(cfg: dict) -> bool:
    return bool(xlit_cfg(cfg).get("enabled", False))


LEGAL_DEFAULTS = {"enabled": False, "min_support": 20, "min_confidence": 0.8, "max_content_confidence": 0.5}


def _params(cfg: dict) -> dict:
    x = xlit_cfg(cfg)
    out = {k: x.get(k, v) for k, v in DEFAULT_PARAMS.items()}
    lg = x.get("legal_tokens") or {}
    if lg.get("enabled", False):  # EXP-N-03; absent -> identical params (and maps) to EXP-N-02
        out["legal_tokens"] = {k: lg.get(k, v) for k, v in LEGAL_DEFAULTS.items()}
    return out


def _read_names(paths: Paths, split: str, source: str, rows: np.ndarray, nonlatin_only: bool, column: str = "name_core") -> dict[int, str]:
    want = pa.array(np.unique(rows).astype(np.int32))
    out: dict[int, str] = {}
    man = load_manifest(paths, split)
    for key, meta in man[source]["countries"].items():
        t = pq.read_table(paths.processed_dir(split) / source / meta["file"], columns=list(dict.fromkeys(["row", "name_core", "name_script", column])))
        m = pc.is_in(t.column("row"), value_set=want)
        if nonlatin_only:
            m = pc.and_(m, pc.is_in(t.column("name_script"), value_set=pa.array([1, 2], t.column("name_script").type)))
        t = t.filter(m)
        out.update(zip(t.column("row").to_pylist(), t.column(column).to_pylist()))
    return out


def train_pairs(paths: Paths, s1_ids: list[str], with_norm: bool = False):
    """(s1_id, S1 name_core, pool name_core) for TRUE train pairs of ``s1_ids`` whose pool name is
    non-Latin or mixed (the only pairs that can teach a cross-script correspondence).
    ``with_norm``: yield (s1_id, S1 name_core, S1 name_norm, pool name_core) instead (EXP-N-03)."""
    from ber.pipeline.run_baseline import _ids, pool_ids

    s1_all, p_all = _ids(paths, "train", "source1"), pool_ids(paths, "train")
    arr = pa.array(s1_ids, pa.string())
    gt = pq.read_table(paths.interim_file("train", "ground_truth"))
    gt = gt.filter(pc.is_in(gt.column("source1_entity_id"), value_set=arr))
    L = pc.split_pattern(gt.column("matched_entity_ids"), ",")
    flat, par = pc.list_flatten(L), pc.list_parent_indices(L)
    keep = pc.not_equal(flat, "")
    s = pc.index_in(pc.take(gt.column("source1_entity_id"), pc.filter(par, keep)), value_set=s1_all).to_numpy(zero_copy_only=False).astype(np.int64)
    p = pc.index_in(pc.filter(flat, keep), value_set=p_all).to_numpy(zero_copy_only=False).astype(np.int64)
    n_s2 = int(load_manifest(paths, "train")["source2"]["rows"])
    pool_names = _read_names(paths, "train", "source2", p[p < n_s2], True)
    pool3 = _read_names(paths, "train", "source3", p[p >= n_s2] - n_s2, True)
    sel = np.array([(pp < n_s2 and int(pp) in pool_names) or (pp >= n_s2 and int(pp - n_s2) in pool3) for pp in p], bool)
    s, p = s[sel], p[sel]
    s1_names = _read_names(paths, "train", "source1", s, False)
    s1_norm = _read_names(paths, "train", "source1", s, False, "name_norm") if with_norm else None
    ids = s1_all.take(pa.array(s)).to_pylist()
    for sid, a, b in zip(ids, s.tolist(), p.tolist()):
        pool_name = pool_names.get(b, "") if b < n_s2 else pool3.get(b - n_s2, "")
        if with_norm:
            yield sid, s1_names.get(a, ""), s1_norm.get(a, ""), pool_name
        else:
            yield sid, s1_names.get(a, ""), pool_name


def learn_map_for(cfg: dict, paths: Paths, s1_ids: list[str], label: str) -> XlitMap:
    """Learn a map from the train pairs of exactly ``s1_ids``. Validation S1s are forbidden."""
    bad = set(load_val_ids(paths)).intersection(s1_ids)
    if bad:
        raise LeakageError(f"xlit map '{label}': {len(bad)} frozen-validation S1 in the learning set")
    prm = _params(cfg)
    lg = prm.get("legal_tokens")
    rows = list(train_pairs(paths, s1_ids, with_norm=bool(lg)))
    mapping, stats = learn_token_map(((r[0], r[1], r[-1]) for r in rows), prm["min_support"], prm["min_confidence"])
    if lg:  # EXP-N-03 stage 2: Indic legal-form words -> dropped from the view
        legal, lstats = learn_legal_tokens(((r[0], r[2], r[3]) for r in rows), set(mapping), LEGAL_FORMS,
                                           lg["min_support"], lg["min_confidence"], lg["max_content_confidence"])
        mapping.update(dict.fromkeys(sorted(legal), DROP))
        stats["legal_stage"] = lstats
    stats["params"] = prm
    meta = {"label": label, "learned_from_s1": len(s1_ids), "learned_from_digest": XlitMap.ids_digest(s1_ids),
            "source": "provided train ground-truth pairs only (no external resource)", **stats}
    LOG.info("xlit map %s: %d mappings from %d S1 (%d usable pairs)", label, len(mapping), len(s1_ids), stats["pairs_usable_latin_x_nonlatin"])
    return XlitMap(mapping, meta)


def final_map_path(cfg: dict, paths: Paths) -> Path:
    return paths.run_dir(cfg["experiment"]["id"]) / "xlit_map.json"


def ensure_final_map(cfg: dict, paths: Paths) -> XlitMap:
    """The map used for validation / E / test scoring: learned from the fit S1 set only."""
    from ber.pipeline.run_baseline import fit_s1_subset

    fit_ids = fit_s1_subset(cfg, paths)
    path = final_map_path(cfg, paths)
    if path.exists():
        m = XlitMap.load(path)
        if m.meta.get("learned_from_digest") == XlitMap.ids_digest(fit_ids) and m.meta.get("params") == _params(cfg):
            return m
        LOG.warning("xlit map at %s was learned from a different S1 set / params: re-learning", path)
    m = learn_map_for(cfg, paths, fit_ids, "final(fit set)")
    m.save(path)
    return m


def fold_of(cfg: dict, ids: list[str]) -> np.ndarray:
    """Fold per id: 0 == the inner-tune split of `fit` (same hash and fraction), 1..F-1 otherwise."""
    from ber.pipeline.run_baseline import _hash_frac

    F = int(xlit_cfg(cfg).get("folds", 5))
    tf = float(cfg["fit"].get("inner_tune_fraction", 0.2))
    h = np.array([_hash_frac(cfg["seed"], s) for s in ids])
    k = 1 + np.minimum(((h - tf) / (1 - tf) * (F - 1)).astype(np.int64), F - 2)
    return np.where(h < tf, 0, k)


def crossfit_plan(cfg: dict, fit_ids: list[str]) -> list[tuple[int, list[str], list[str]]]:
    """[(fold, S1 ids scored with this fold's map, S1 ids the map is learned from)]."""
    fold = fold_of(cfg, fit_ids)
    ids = np.asarray(fit_ids, dtype=object)
    F = int(xlit_cfg(cfg).get("folds", 5))
    plan = []
    for f in range(F):
        learn = (fold != 0) & (fold != f)
        plan.append((f, sorted(ids[fold == f].tolist()), sorted(ids[learn].tolist())))
    return plan
