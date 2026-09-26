"""Blocking forensics for a blocker config (READ-ONLY diagnostic).

1. Uncapped re-blocking of the frozen validation S1s (top_k = --raw-topk): recall@K and
   candidates/S1 for K in 10..raw_topk, overall and per country; fate of every true pair
   (in top-K / retrieved but cut by top-K / never retrieved).
2. For never-retrieved true pairs: which evidence they share, and for the two combined key
   families of Blocking V2 whether the key was
     generated on both sides (so the block-size cap must have removed it),
     shared but not among either record's rarest-k selection, or
     not shared at all.
   (Rarest-k is re-computed from pool DF exactly as in blocker.py; ties are broken by token
    string here instead of dictionary id, so this attribution is approximate.)
3. Coverage of candidate NEW key families among never-retrieved pairs (upper bounds):
   name token pair, name token + number, name_concat equality, address-number + name token.

    python scripts/analysis/blocking_forensics.py --config configs/exp_b04_blocking_v2.yaml
"""

from __future__ import annotations

import argparse
import copy
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_exp_b01 import load_views  # noqa: E402

from ber.blocking.blocker import CountryBlocker  # noqa: E402
from ber.config import Paths, load_config  # noqa: E402
from ber.data.normalize_records import load_manifest, read_country  # noqa: E402
from ber.data.split import load_val_ids  # noqa: E402
from ber.features.pair_features import strip_zeros_list  # noqa: E402
from ber.normalization.views import ADDRESS_STOPWORDS  # noqa: E402
from ber.pipeline.run_baseline import VIEW_COLS, LazyPool, _ids, pool_ids, resolve_xmap, with_xlit  # noqa: E402
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, stage, write_json  # noqa: E402

SHIFT = np.int64(1) << np.int64(24)


def useful_words(s):
    return [w for w in s.split() if w not in ADDRESS_STOPWORDS and len(w) >= 2]


def rarest(tokens, df, k):
    toks = sorted({t for t in tokens if df.get(t, 0) >= 1}, key=lambda t: (df[t], t))
    return toks[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_b04_blocking_v2.yaml")
    ap.add_argument("--raw-topk", type=int, default=200)
    ap.add_argument("--out", default="experiments/runs/POST-0.91-FORENSIC/blocking_forensics.json")
    ap.add_argument("--dump", default=None, help="optional .npz with the rank of every validation true pair (for config comparisons)")
    a = ap.parse_args()
    setup_logging()
    configure_arrow()
    cfg = load_config(a.config)
    paths = Paths(cfg)
    val = load_val_ids(paths)
    val_arr = pa.array(val, pa.string())
    man = load_manifest(paths, "train")
    n_s2 = int(man["source2"]["rows"])
    s1_ids, p_ids = _ids(paths, "train", "source1"), pool_ids(paths, "train")

    # ---- ground truth for validation S1 as codes
    gt = pq.read_table(paths.interim_file("train", "ground_truth"))
    gt = gt.filter(pc.is_in(gt.column("source1_entity_id"), value_set=val_arr))
    L = pc.split_pattern(gt.column("matched_entity_ids"), ",")
    flat, par = pc.list_flatten(L), pc.list_parent_indices(L)
    keep = pc.not_equal(flat, "")
    gs = pc.index_in(pc.take(gt.column("source1_entity_id"), pc.filter(par, keep)), value_set=s1_ids).to_numpy(zero_copy_only=False).astype(np.int64)
    gp = pc.index_in(pc.filter(flat, keep), value_set=p_ids).to_numpy(zero_copy_only=False).astype(np.int64)
    gcode = gs * SHIFT + gp
    country_of = {}
    for k in man["source1"]["countries"]:
        for r in read_country(paths, "train", "source1", k, ["row"]).column("row").to_numpy():
            country_of[int(r)] = k
    g_country = np.array([country_of[int(s)] for s in gs], dtype=object)

    # ---- 1. uncapped re-blocking
    bcfg = copy.deepcopy(cfg["blocking"])
    bcfg["top_k"] = a.raw_topk
    xmap = resolve_xmap(cfg, paths)  # EXP-N-02: final (fit-set) cross-script map when enabled, else None
    codes, ranks, s1rows = [], [], []
    with stage("uncapped re-blocking"), PeakMemory() as pm:
        for key in sorted(man["source1"]["countries"]):
            s1 = read_country(paths, "train", "source1", key, VIEW_COLS)
            s1 = s1.filter(pc.is_in(s1.column("entity_id"), value_set=val_arr))
            prow = []
            for src, source in ((2, "source2"), (3, "source3")):
                prow.append(read_country(paths, "train", source, key, ["row"]).column("row").to_numpy().astype(np.int64) + (n_s2 if src == 3 else 0))
            prow = np.concatenate(prow)
            srow = s1.column("row").to_numpy().astype(np.int64)
            b = CountryBlocker(with_xlit(s1, xmap), LazyPool(paths, "train", key, xmap), bcfg)
            for lo, hi in b.chunk_bounds():
                c = b.candidates(lo, hi)
                codes.append(srow[c["s1_local"]] * SHIFT + prow[c["pool_local"]])
                ranks.append(c["cheap_rank"].astype(np.int32))
                s1rows.append(srow[c["s1_local"]])
            del b, s1
    code = np.concatenate(codes)
    rank = np.concatenate(ranks)
    srow_all = np.concatenate(s1rows)
    o = np.argsort(code)
    code_s, rank_s = code[o], rank[o]
    idx = np.minimum(np.searchsorted(code_s, gcode), len(code_s) - 1)
    found = code_s[idx] == gcode
    g_rank = np.where(found, rank_s[idx], 10**6)
    rep = {"config": a.config, "raw_topk": a.raw_topk, "true_pairs": int(len(gcode)),
           "reblock_runtime_s": round(pm.seconds, 1), "reblock_peak_gb": round(pm.peak_gb, 2), "recall_at_k": {}}
    s1_country = np.array([country_of[int(r)] for r in srow_all], dtype=object)
    val_rows = pc.index_in(val_arr, value_set=s1_ids).to_numpy(zero_copy_only=False).astype(np.int64)
    if a.dump:
        np.savez_compressed(a.dump, gcode=gcode, g_rank=g_rank, gs=gs, gp=gp, g_country=g_country.astype(str))
    rep["candidate_distribution"] = {}
    for K in (20, a.raw_topk):
        per = np.bincount(srow_all[rank < K], minlength=int(val_rows.max()) + 1)[val_rows]
        rep["candidate_distribution"][str(K)] = {
            "total_candidates": int(per.sum()), "mean": float(per.mean()), "median": float(np.median(per)),
            "p95": float(np.percentile(per, 95)), "p99": float(np.percentile(per, 99)), "max": int(per.max()),
            "s1_with_zero_candidates": int((per == 0).sum())}
    for K in (10, 20, 30, 50, 100, a.raw_topk):
        within = g_rank < K
        cand_k = rank < K
        row = {"recall": float(within.mean()), "candidates_per_s1": float(cand_k.sum() / len(val))}
        for c in sorted(set(g_country.tolist())):
            sel = g_country == c
            row[f"recall_{c}"] = float(within[sel].mean())
            row[f"candidates_per_s1_{c}"] = float((cand_k & (s1_country == c)).sum() / max(len(np.unique(srow_all[s1_country == c])), 1))
        rep["recall_at_k"][str(K)] = row
    top_k = int(cfg["blocking"]["top_k"])
    fate = np.where(g_rank < top_k, "in_topk", np.where(found, "cut_by_topk", "never_retrieved"))
    rep["fate"] = {c: {f: float((fate[g_country == c] == f).mean()) for f in ("in_topk", "cut_by_topk", "never_retrieved")}
                   for c in sorted(set(g_country.tolist()))} | {"all": {f: float((fate == f).mean()) for f in ("in_topk", "cut_by_topk", "never_retrieved")}}
    rep["fate_counts"] = {f: int((fate == f).sum()) for f in ("in_topk", "cut_by_topk", "never_retrieved")}

    # ---- 2+3. never-retrieved pairs: evidence and key attribution
    nr = np.flatnonzero(fate == "never_retrieved")
    with stage("views + DF"):
        v1 = load_views(paths, "train", "source1", np.unique(gs[nr]))
        v2 = load_views(paths, "train", "source2", np.unique(gp[nr][gp[nr] < n_s2]))
        v3 = load_views(paths, "train", "source3", np.unique(gp[nr][gp[nr] >= n_s2] - n_s2))
        name_df, word_df = {}, {}
        for k in man["source1"]["countries"]:
            nd, wd = Counter(), Counter()
            for src in ("source2", "source3"):
                t = read_country(paths, "train", src, k, ["name_core", "addr_words"])
                for n_, w_ in zip(t.column("name_core").to_pylist(), t.column("addr_words").to_pylist()):
                    nd.update(set(n_.split()))
                    wd.update(set(useful_words(w_)))
            name_df[k], word_df[k] = nd, wd
    pa_cfg, pp_cfg = cfg["blocking"]["passes"].get("name_addr", {}), cfg["blocking"]["passes"].get("addr_pair", {})
    kn, kw_na, kw_ap = pa_cfg.get("max_name_tokens", 2), pa_cfg.get("max_words", 3), pp_cfg.get("max_words", 3)
    cats: dict[str, Counter] = {}
    n_by = Counter()
    for i in nr:
        s, p = int(gs[i]), int(gp[i])
        x = v1.get(s)
        y = v2.get(p) if p < n_s2 else v3.get(p - n_s2)
        c = g_country[i]
        if not x or not y:
            continue
        n_by[c] += 1
        nd, wd = name_df[c], word_df[c]
        xn, yn = x[1].split(), y[1].split()
        xw, yw = useful_words(x[5]), useful_words(y[5])
        sn, sw = set(xn) & set(yn), set(xw) & set(yw)
        nums = set(strip_zeros_list([x[4]])[0].split()) & set(strip_zeros_list([y[4]])[0].split())
        # Blocking V2 key attribution
        xa, ya = rarest(xn, nd, kn), rarest(yn, nd, kn)
        xb, yb = rarest(xw, wd, kw_na), rarest(yw, wd, kw_na)
        na_keys = {(n, w) for n in xa for w in xb} & {(n, w) for n in ya for w in yb}
        xp, yp = rarest(xw, wd, kw_ap), rarest(yw, wd, kw_ap)
        ap_keys = {tuple(sorted((u, v))) for u in xp for v in xp if u < v} & {tuple(sorted((u, v))) for u in yp for v in yp if u < v}
        flags = {
            "name_addr: key on both sides -> removed by block cap": bool(na_keys),
            "name_addr: shared name+addr evidence but outside rarest-k": (not na_keys) and bool(sn) and bool(sw),
            "addr_pair: key on both sides -> removed by block cap": bool(ap_keys),
            "addr_pair: >=2 shared words but outside rarest-k": (not ap_keys) and len(sw) >= 2,
            "shares name token(s) only (no useful address word)": bool(sn) and not sw,
            "shares address word(s) only (no name token)": bool(sw) and not sn,
            "shares a house number": bool(nums),
            "address empty (either side)": x[3] == "" or y[3] == "",
            "script mismatch (Latin S1 vs non-Latin record)": x[2] == 0 and y[2] in (1, 2),
            "no shared name token, address word or number": not (sn or sw or nums),
            # candidate NEW key families (coverage upper bounds)
            "NEW name_token_pair (>=2 shared name tokens)": len(sn) >= 2,
            "NEW name_token + house number": bool(sn) and bool(nums),
            "NEW name_core concat equal": x[1].replace(" ", "") == y[1].replace(" ", "") and x[1] != "",
        }
        for k2, v in flags.items():
            if v:
                cats.setdefault(k2, Counter())[c] += 1
    n = sum(n_by.values())
    rep["never_retrieved"] = {"pairs": int(n), "by_country_pairs": dict(n_by),
                              "shares": {k2: {"all": sum(v.values()) / max(n, 1), **{c: v[c] / max(n_by[c], 1) for c in n_by}} for k2, v in cats.items()}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(a.out), rep)
    LOG.info("recall@K %s", {k: round(v["recall"], 4) for k, v in rep["recall_at_k"].items()})
    LOG.info("fate %s", rep["fate"]["all"])
    for k2, v in sorted(rep["never_retrieved"]["shares"].items(), key=lambda x: -x[1]["all"]):
        LOG.info("%.3f  %s  %s", v["all"], k2, {c: round(x, 3) for c, x in v.items() if c != "all"})


if __name__ == "__main__":
    main()
