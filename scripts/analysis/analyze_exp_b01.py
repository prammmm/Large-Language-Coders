"""EXP-B-01 error analysis (READ-ONLY diagnostics; nothing here changes the pipeline).

Inputs  : data/processed/train/scored (validation candidates + features + scores),
          data/interim + data/processed (ids, views), data/splits/val_s1_ids.txt,
          experiments/runs/<exp>/tuned_threshold.json, data/processed/test/scored.
Outputs : experiments/runs/<exp>/analysis/{analysis.json, examples.md}

What it measures
  1. Loss decomposition of validation F0.5 (singleton FPs, empty-on-positive, partial).
  2. Fate of every true pair: in top-K candidates / cut by the top-K cap / never blocked
     (re-runs blocking for the validation S1s WITHOUT the top-K cap).
  3. Descriptors of missed / low-scored / false-positive pairs (empty address, script
     mismatch, lexical overlap, ...).
  4. Diagnostic what-if simulations on the EXISTING scores (not adopted, only to rank
     next experiments): top-1 fallback, exclusivity among validation S1s, recall at larger K.
  5. Test-set behaviour per country (incl. France, which has no labels).

    python scripts/analysis/analyze_exp_b01.py --config configs/baseline_v0.yaml
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ber.blocking.blocker import CountryBlocker
from ber.config import Paths, load_config
from ber.data.normalize_records import load_manifest, read_country
from ber.data.split import load_val_ids, match_counts
from ber.evaluation.metrics import f05_from_counts
from ber.pipeline.run_baseline import VIEW_COLS, LazyPool, _ids, pool_ids
from ber.utils.runtime import LOG, configure_arrow, read_json, setup_logging, stage, write_json

SHIFT = np.int64(1) << np.int64(24)  # pool rows < 16.7M -> code = s1_row * 2^24 + pool_row
VIEWS = ["row", "name_norm", "name_core", "name_script", "addr_norm", "addr_nums", "addr_words"]
STOP_MIN_LEN = 2


def code(s1_row, pool_row) -> np.ndarray:
    return np.asarray(s1_row, np.int64) * SHIFT + np.asarray(pool_row, np.int64)


def frac(x) -> float:
    x = np.asarray(x)
    return float(x.mean()) if len(x) else float("nan")


# ----------------------------------------------------------------------------- loading
def load_views(paths: Paths, split: str, source: str, rows: np.ndarray) -> dict[int, tuple]:
    """Views for the given original-file rows of one source (streamed, low memory)."""
    want = pa.array(np.unique(rows).astype(np.int32))
    out: dict[int, tuple] = {}
    for key in load_manifest(paths, split)[source]["countries"]:
        f = paths.processed_dir(split) / source / load_manifest(paths, split)[source]["countries"][key]["file"]
        for b in pq.ParquetFile(f).iter_batches(columns=VIEWS, batch_size=500_000):
            t = pa.Table.from_batches([b])
            t = t.filter(pc.is_in(t.column("row"), value_set=want))
            if t.num_rows:
                cols = [t.column(c).to_pylist() for c in VIEWS]
                for vals in zip(*cols):
                    out[vals[0]] = vals[1:]
    return out


def load_raw(paths: Paths, split: str, source: str, rows: set[int]) -> dict[int, tuple]:
    """Original (un-normalized) name/address/country for a few rows, for examples."""
    out, start = {}, 0
    for b in pq.ParquetFile(paths.interim_file(split, source)).iter_batches(batch_size=500_000):
        n = b.num_rows
        hit = [r for r in rows if start <= r < start + n]
        if hit:
            d = b.to_pydict()
            for r in hit:
                i = r - start
                out[r] = (d["entity_id"][i], d["business_name"][i], d["business_address"][i], d["country"][i])
        start += n
    return out


def pair_descriptor(v1: tuple, v2: tuple | None) -> dict:
    """v = (name_norm, name_core, name_script, addr_norm, addr_nums, addr_words)."""
    from ber.normalization.views import ADDRESS_STOPWORDS

    if v2 is None:
        return {}
    n1, c1, sc1, a1, nu1, w1 = v1
    n2, c2, sc2, a2, nu2, w2 = v2
    t1, t2 = set(c1.split()), set(c2.split())
    uw1 = {w for w in w1.split() if w not in ADDRESS_STOPWORDS and len(w) >= STOP_MIN_LEN}
    uw2 = {w for w in w2.split() if w not in ADDRESS_STOPWORDS and len(w) >= STOP_MIN_LEN}
    return {
        "addr_empty": (a1 == "") or (a2 == ""),
        "script_mismatch": sc2 in (1, 2) and sc1 == 0,
        "name_norm_eq": n1 == n2 and n1 != "",
        "share_name_token": bool(t1 & t2),
        "share_useful_addr_word": bool(uw1 & uw2),
        "share_addr_number": bool(set(nu1.split()) & set(nu2.split())),
    }


def primary_miss_cause(d: dict) -> str:
    if d["addr_empty"]:
        return "address empty (either side)"
    if d["script_mismatch"] and not (d["share_useful_addr_word"] and d["share_addr_number"]):
        return "non-Latin pool name + weak address overlap"
    if not d["share_useful_addr_word"] and not d["share_addr_number"]:
        return "no shared address word or number"
    if d["share_addr_number"] and not d["share_useful_addr_word"]:
        return "shares only a number (no useful word)"
    if d["share_useful_addr_word"] and not d["share_addr_number"]:
        return "shares only address words (no number)"
    return "shares number+word but key dropped (DF/size cap or rarest-k)"


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_v0.yaml")
    ap.add_argument("--raw-topk", type=int, default=200, help="K for the uncapped re-blocking of validation S1s")
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--set", dest="sets", action="append", default=[], help="config override, e.g. experiment.id=X")
    a = ap.parse_args()
    setup_logging()
    configure_arrow()
    from ber.config import parse_sets

    cfg = load_config(a.config, parse_sets(a.sets))
    paths = Paths(cfg)
    exp = cfg["experiment"]["id"]
    out_dir = paths.run_dir(exp) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    mfile = paths.run_dir(exp) / "metrics.json"
    if mfile.exists() and "threshold_used" in read_json(mfile):
        t_best = float(read_json(mfile)["threshold_used"])  # the threshold the experiment actually uses
    else:
        t_best = float(read_json(paths.run_dir(exp) / "tuned_threshold.json")["threshold"])
    tag = cfg.get("scoring", {}).get("eval_source_tag") or cfg.get("scoring", {}).get("tag", "scored")
    rng = random.Random(cfg["seed"])
    R: dict = {"experiment": exp, "threshold": t_best}

    # ---------------- ids, validation truth
    with stage("load ids + truth"):
        s1_ids = _ids(paths, "train", "source1")
        p_ids = pool_ids(paths, "train")
        n_s2 = int(load_manifest(paths, "train")["source2"]["rows"])
        val = load_val_ids(paths)
        val_arr = pa.array(val, pa.string())
        m = len(val)
        val_rows = pc.index_in(val_arr, value_set=s1_ids).to_numpy(zero_copy_only=False).astype(np.int64)
        row_to_pos = {int(r): i for i, r in enumerate(val_rows)}
        gt = pq.read_table(paths.interim_file("train", "ground_truth"))
        all_lists = pc.split_pattern(gt.column("matched_entity_ids"), ",")
        all_flat = pc.list_flatten(all_lists)
        all_parent = pc.list_parent_indices(all_lists)
        keep = pc.not_equal(all_flat, "")
        all_flat, all_parent = pc.filter(all_flat, keep), pc.filter(all_parent, keep)
        all_gt_pool_row = pc.index_in(all_flat, value_set=p_ids).to_numpy(zero_copy_only=False).astype(np.int64)
        all_gt_s1_row = pc.index_in(pc.take(gt.column("source1_entity_id"), all_parent), value_set=s1_ids).to_numpy(zero_copy_only=False).astype(np.int64)
        owner_of_pool = np.full(len(p_ids), -1, np.int64)  # which S1 row owns each matched pool record
        owner_of_pool[all_gt_pool_row] = all_gt_s1_row
        is_val_s1 = np.zeros(len(s1_ids), bool)
        is_val_s1[val_rows] = True
        vsel = is_val_s1[all_gt_s1_row]
        gt_s1_row, gt_pool_row = all_gt_s1_row[vsel], all_gt_pool_row[vsel]
        gt_code = code(gt_s1_row, gt_pool_row)
        n_true = np.bincount([row_to_pos[int(r)] for r in gt_s1_row], minlength=m)
        del gt, all_lists, all_flat, all_parent
        s1_country = {}
        for key in load_manifest(paths, "train")["source1"]["countries"]:
            rows = read_country(paths, "train", "source1", key, ["row"]).column("row").to_numpy()
            for r in rows[is_val_s1[rows]]:
                s1_country[int(r)] = key
        country_pos = np.array([s1_country[int(r)] for r in val_rows], dtype=object)

    # ---------------- scored validation candidates
    with stage("load scored candidates"):
        parts = sorted(paths.scored_dir("train", tag).glob("*.parquet"))
        sc = pa.concat_tables([pq.read_table(p) for p in parts])
        s1r = sc.column("s1_row").to_numpy().astype(np.int64)
        plr = sc.column("pool_row").to_numpy().astype(np.int64)
        score = sc.column("score").to_numpy()
        rank = sc.column("cheap_rank").to_numpy().astype(np.int64)
        feats = {c: sc.column(c).to_numpy() for c in sc.column_names if c not in ("s1_row", "pool_row", "score", "cheap_rank", "cheap_score", "pass_flags")}
        flags = sc.column("pass_flags").to_numpy()
        del sc
        c_code = code(s1r, plr)
        is_true = np.isin(c_code, gt_code)
        pos = np.array([row_to_pos[int(r)] for r in s1r], np.int64)
        pred = score >= t_best

    # ---------------- 1. entity-level loss decomposition
    k = np.bincount(pos[pred], minlength=m)
    tp = np.bincount(pos[pred & is_true], minlength=m)
    c_tp = np.bincount(pos[is_true], minlength=m)
    f = f05_from_counts(tp, k, n_true)
    loss = 1.0 - f
    cat = np.full(m, "perfect (F=1)", dtype=object)
    single = n_true == 0
    cat[single & (k > 0)] = "singleton predicted non-empty (F=0)"
    cat[(~single) & (k == 0) & (c_tp == 0)] = "empty prediction, no true pair in candidates (F=0)"
    cat[(~single) & (k == 0) & (c_tp > 0)] = "empty prediction, true pair in candidates but below threshold (F=0)"
    part = (~single) & (k > 0) & (f < 1)
    cat[part & (tp < k)] = "partial: has false merge(s)"
    cat[part & (tp == k)] = "partial: precise but incomplete (missed matches)"
    decomposition = {}
    for c in sorted(set(cat.tolist())):
        sel = cat == c
        decomposition[c] = {"entities": int(sel.sum()), "share_entities": frac(sel), "f05_loss_points": float(loss[sel].sum() / m)}
    R["val_f05"] = float(f.mean())
    R["loss_decomposition"] = decomposition
    # split partial-incomplete loss into blocking vs matcher part
    inc = cat == "partial: precise but incomplete (missed matches)"
    R["incomplete_entities_missing_due_to_blocking_share"] = float(((n_true - c_tp)[inc]).sum() / max(((n_true - tp)[inc]).sum(), 1))
    LOG.info("val F0.5 recomputed %.4f", R["val_f05"])

    # ---------------- 2. fate of true pairs (needs uncapped re-blocking)
    with stage("uncapped re-blocking of validation S1"):
        bcfg = copy.deepcopy(cfg["blocking"])
        bcfg["top_k"] = a.raw_topk
        raw_codes, raw_ranks = [], []
        for key in sorted(load_manifest(paths, "train")["source1"]["countries"]):
            s1 = read_country(paths, "train", "source1", key, VIEW_COLS)
            s1 = s1.filter(pc.is_in(s1.column("entity_id"), value_set=val_arr))
            if s1.num_rows == 0:
                continue
            lp = LazyPool(paths, "train", key)
            prow = []
            for src, source in ((2, "source2"), (3, "source3")):
                r = read_country(paths, "train", source, key, ["row"]).column("row").to_numpy().astype(np.int64)
                prow.append(r + (n_s2 if src == 3 else 0))
            prow = np.concatenate(prow)
            srow = s1.column("row").to_numpy().astype(np.int64)
            b = CountryBlocker(s1, lp, bcfg)
            for lo, hi in b.chunk_bounds():
                cnd = b.candidates(lo, hi)
                raw_codes.append(code(srow[cnd["s1_local"]], prow[cnd["pool_local"]]))
                raw_ranks.append(cnd["cheap_rank"].astype(np.int64))
            del b, s1
        raw_code = np.concatenate(raw_codes)
        raw_rank = np.concatenate(raw_ranks)
    in_top = np.isin(gt_code, c_code)
    o = np.argsort(raw_code)
    raw_code_s, raw_rank_s = raw_code[o], raw_rank[o]
    idx = np.searchsorted(raw_code_s, gt_code)
    idx_c = np.minimum(idx, len(raw_code_s) - 1)
    in_raw = raw_code_s[idx_c] == gt_code
    gt_raw_rank = np.where(in_raw, raw_rank_s[idx_c], -1)
    total = len(gt_code)
    R["true_pairs"] = int(total)
    R["true_pair_fate"] = {
        "in_top20_candidates": float(in_top.mean()),
        "blocked_but_cut_by_top20": float((in_raw & ~in_top).mean()),
        f"never_blocked_within_top{a.raw_topk}": float((~in_raw).mean()),
    }
    raw_s1 = raw_code // SHIFT
    raw_pos = np.array([row_to_pos[int(r)] for r in raw_s1], np.int64)
    R["recall_if_top_k"] = {}
    for kk in (10, 20, 30, 50, 100, a.raw_topk):
        sel = raw_rank < kk
        R["recall_if_top_k"][str(kk)] = {
            "blocking_recall": float((in_raw & (gt_raw_rank < kk)).mean()),
            "candidates_per_s1_mean": float(np.bincount(raw_pos[sel], minlength=m).mean()),
        }
    gt_pos = np.array([row_to_pos[int(r)] for r in gt_s1_row], np.int64)
    R["true_pair_fate_by_country"] = {}
    for c in sorted(set(country_pos.tolist())):
        sel = country_pos[gt_pos] == c
        R["true_pair_fate_by_country"][c] = {
            "in_top20": frac(in_top[sel]),
            "cut_by_top20": frac((in_raw & ~in_top)[sel]),
            "never_blocked": frac((~in_raw)[sel]),
        }

    # ---------------- 3. descriptors (views for S1 + all needed pool rows)
    fp = pred & ~is_true
    low_true = is_true & ~pred
    with stage("load views"):
        need_pool = np.unique(np.concatenate([gt_pool_row, plr[fp]]))
        v1 = load_views(paths, "train", "source1", val_rows)
        v2 = load_views(paths, "train", "source2", need_pool[need_pool < n_s2])
        v3 = load_views(paths, "train", "source3", need_pool[need_pool >= n_s2] - n_s2)

    def pview(pr: int):
        return v2.get(pr) if pr < n_s2 else v3.get(pr - n_s2)

    def descr(sr, pr):
        return pair_descriptor(v1[int(sr)], pview(int(pr)))

    with stage("describe pairs"):
        # a) never-blocked true pairs
        nb_idx = np.flatnonzero(~in_raw)
        causes, nb_desc = {}, []
        for i in nb_idx:
            d = descr(gt_s1_row[i], gt_pool_row[i])
            nb_desc.append(d)
            cz = primary_miss_cause(d)
            causes[cz] = causes.get(cz, 0) + 1
        R["never_blocked"] = {
            "pairs": int(len(nb_idx)),
            "primary_cause_share": {k2: v / max(len(nb_idx), 1) for k2, v in sorted(causes.items(), key=lambda x: -x[1])},
            "rates": {k2: frac([d[k2] for d in nb_desc]) for k2 in (nb_desc[0].keys() if nb_desc else [])},
        }
        # comparison: descriptors of true pairs that WERE found
        found_idx = np.flatnonzero(in_top)
        smp = rng.sample(list(found_idx), min(50_000, len(found_idx)))
        fd = [descr(gt_s1_row[i], gt_pool_row[i]) for i in smp]
        R["found_true_pairs_rates"] = {k2: frac([d[k2] for d in fd]) for k2 in (fd[0].keys() if fd else [])}
        # by country for never-blocked
        R["never_blocked_by_country"] = {}
        for c in sorted(set(country_pos.tolist())):
            sel = [j for j, i in enumerate(nb_idx) if country_pos[gt_pos[i]] == c]
            cc = {}
            for j in sel:
                cz = primary_miss_cause(nb_desc[j])
                cc[cz] = cc.get(cz, 0) + 1
            R["never_blocked_by_country"][c] = {"pairs": len(sel), "primary_cause_share": {k2: v / max(len(sel), 1) for k2, v in sorted(cc.items(), key=lambda x: -x[1])}}

        # b) true pairs in candidates but scored below threshold
        lt = np.flatnonzero(low_true)
        smp = rng.sample(list(lt), min(50_000, len(lt)))
        ld = [descr(s1r[i], plr[i]) for i in smp]
        R["true_below_threshold"] = {
            "pairs": int(len(lt)),
            "rates": {k2: frac([d[k2] for d in ld]) for k2 in (ld[0].keys() if ld else [])},
            "score_quantiles": {q: float(np.quantile(score[lt], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if len(lt) else {},
            "feature_means": {c: float(v[lt].mean()) for c, v in feats.items()},
        }
        R["true_above_threshold_feature_means"] = {c: float(v[pred & is_true].mean()) for c, v in feats.items()}

        # c) false positives
        fpi = np.flatnonzero(fp)
        owner = owner_of_pool[plr[fpi]]
        R["false_positives"] = {
            "pairs": int(len(fpi)),
            "candidate_is_matched_to_another_S1": frac(owner >= 0),
            "candidate_is_unmatched_distractor": frac(owner < 0),
            "owner_is_also_a_val_S1": frac(is_val_s1[owner[owner >= 0]]) if (owner >= 0).any() else float("nan"),
            "feature_means": {c: float(v[fpi].mean()) for c, v in feats.items()},
            "score_quantiles": {q: float(np.quantile(score[fpi], q)) for q in (0.1, 0.5, 0.9)} if len(fpi) else {},
        }
        smp = rng.sample(list(fpi), min(50_000, len(fpi)))
        fdsc = [descr(s1r[i], plr[i]) for i in smp]
        R["false_positives"]["rates"] = {k2: frac([d[k2] for d in fdsc]) for k2 in (fdsc[0].keys() if fdsc else [])}
        # d) singleton false positives: what did we match?
        sfp_ent = single & (k > 0)
        sfp_pairs = fp & sfp_ent[pos]
        R["singleton_false_positive_pairs"] = {
            "entities": int(sfp_ent.sum()),
            "pairs": int(sfp_pairs.sum()),
            "candidate_matched_to_another_S1": frac(owner_of_pool[plr[sfp_pairs]] >= 0),
            "name_norm_eq_rate": frac(feats["name_norm_eq"][sfp_pairs]),
            "addr_norm_eq_rate": frac(feats["addr_norm_eq"][sfp_pairs]),
        }

    # ---------------- 4. what-if simulations on existing scores (diagnostic only)
    sims = {}
    # (a) top-1 fallback: entities with no prediction take their best candidate if score >= t2
    order = np.lexsort((-score, pos))
    first = np.ones(len(order), bool)
    first[1:] = pos[order][1:] != pos[order][:-1]
    top1 = order[first]
    for t2 in (0.50, 0.55, 0.60, 0.65):
        p2 = pred.copy()
        add = top1[(k[pos[top1]] == 0) & (score[top1] >= t2)]
        p2[add] = True
        k2 = np.bincount(pos[p2], minlength=m)
        tp2 = np.bincount(pos[p2 & is_true], minlength=m)
        sims[f"top1_fallback_t2={t2:.2f}"] = float(f05_from_counts(tp2, k2, n_true).mean())
    # (b) exclusivity among validation S1 only (underestimates conflicts: train-fold S1 not scored)
    po = np.flatnonzero(pred)
    best = {}
    for i in po:
        pr = int(plr[i])
        if pr not in best or score[i] > score[best[pr]]:
            best[pr] = i
    p3 = np.zeros_like(pred)
    p3[list(best.values())] = True
    k3 = np.bincount(pos[p3], minlength=m)
    tp3 = np.bincount(pos[p3 & is_true], minlength=m)
    sims["exclusivity_among_val_s1"] = float(f05_from_counts(tp3, k3, n_true).mean())
    sims["pool_records_predicted_for_2plus_val_s1"] = int(len(po) - len(best))
    # (c) perfect matcher on current candidates (= oracle), and perfect blocking + current decision is N/A
    sims["oracle_perfect_decision_on_current_candidates"] = float(f05_from_counts(c_tp, c_tp, n_true).mean())
    # (d) per-source thresholds (S2 vs S3) — tuned jointly on the same fold: optimistic
    is_s3 = plr >= n_s2
    best_src = (0.0, None)
    for t2s in np.arange(0.55, 0.80, 0.01):
        for t3s in np.arange(0.55, 0.80, 0.01):
            p4 = np.where(is_s3, score >= t3s, score >= t2s)
            k4 = np.bincount(pos[p4], minlength=m)
            tp4 = np.bincount(pos[p4 & is_true], minlength=m)
            v = float(f05_from_counts(tp4, k4, n_true).mean())
            if v > best_src[0]:
                best_src = (v, (round(float(t2s), 2), round(float(t3s), 2)))
    sims["per_source_threshold_best_in_sample"] = {"f05": best_src[0], "t_s2_t_s3": best_src[1]}
    R["what_if_simulations"] = sims
    R["score_separation"] = {
        "true_pairs_score_quantiles": {q: float(np.quantile(score[is_true], q)) for q in (0.05, 0.25, 0.5, 0.75)},
        "false_pairs_score_quantiles": {q: float(np.quantile(score[~is_true], q)) for q in (0.5, 0.9, 0.99, 0.999)},
    }

    # ---------------- 5. test behaviour by country vs validation
    with stage("test diagnostics"):
        tparts = sorted(paths.scored_dir("test", tag).glob("*.parquet"))
    if not tparts:  # no test inference for this experiment yet -> skip (never mix experiments)
        R["test_by_country"] = "not available (no test parts for this experiment)"
    else:
            tt = pa.concat_tables([pq.read_table(p, columns=["s1_row", "score", "cheap_rank"]) for p in tparts])
            ts1 = tt.column("s1_row").to_numpy().astype(np.int64)
            tsc = tt.column("score").to_numpy()
            del tt
            n_test_s1 = len(_ids(paths, "test", "source1"))
            tcountry = np.empty(n_test_s1, dtype=object)
            for key in load_manifest(paths, "test")["source1"]["countries"]:
                rows = read_country(paths, "test", "source1", key, ["row"]).column("row").to_numpy()
                tcountry[rows] = key
            n_c = np.bincount(ts1, minlength=n_test_s1)
            n_p = np.bincount(ts1[tsc >= t_best], minlength=n_test_s1)
            mx = np.full(n_test_s1, -1.0)
            np.maximum.at(mx, ts1, tsc)
            R["test_by_country"] = {}
            for c in sorted(set(tcountry.tolist())):
                sel = tcountry == c
                R["test_by_country"][c] = {
                    "s1": int(sel.sum()),
                    "candidates_per_s1": float(n_c[sel].mean()),
                    "no_candidates_share": frac(n_c[sel] == 0),
                    "predicted_per_s1": float(n_p[sel].mean()),
                    "empty_prediction_share": frac(n_p[sel] == 0),
                    "top1_score_median": float(np.median(mx[sel & (n_c > 0)])),
                }
            vmx = np.full(m, -1.0)
            np.maximum.at(vmx, pos, score)
            R["val_by_country_same_stats"] = {}
            for c in sorted(set(country_pos.tolist())):
                sel = country_pos == c
                R["val_by_country_same_stats"][c] = {
                    "s1": int(sel.sum()),
                    "candidates_per_s1": float(np.bincount(pos, minlength=m)[sel].mean()),
                    "no_candidates_share": frac(np.bincount(pos, minlength=m)[sel] == 0),
                    "predicted_per_s1": float(k[sel].mean()),
                    "empty_prediction_share": frac(k[sel] == 0),
                    "true_matches_per_s1": float(n_true[sel].mean()),
                    "top1_score_median": float(np.median(vmx[sel & (vmx >= 0)])),
                }

    write_json(out_dir / "analysis.json", R)

    # ---------------- examples (a few per category, original strings)
    with stage("examples"):
        ex = {
            "never blocked true pair": [(int(gt_s1_row[i]), int(gt_pool_row[i])) for i in rng.sample(list(nb_idx), min(a.examples, len(nb_idx)))],
            "true pair scored below threshold": [(int(s1r[i]), int(plr[i])) for i in rng.sample(list(lt), min(a.examples, len(lt)))],
            "false positive (predicted, not true)": [(int(s1r[i]), int(plr[i])) for i in rng.sample(list(fpi), min(a.examples, len(fpi)))],
        }
        s1_need = {s for v in ex.values() for s, _ in v}
        p_need = {p for v in ex.values() for _, p in v}
        r1 = load_raw(paths, "train", "source1", s1_need)
        r2 = load_raw(paths, "train", "source2", {p for p in p_need if p < n_s2})
        r3 = load_raw(paths, "train", "source3", {p - n_s2 for p in p_need if p >= n_s2})
        lines = [f"# {exp} validation error examples (random, seed {cfg['seed']}; raw organizer data — do not publish)\n"]
        for title, pairs in ex.items():
            lines.append(f"\n## {title}\n\n| S1 name | S1 address | candidate name | candidate address |\n|---|---|---|---|")
            for s, p in pairs:
                a1 = r1.get(s, ("?", "?", "?", "?"))
                a2 = r2.get(p) if p < n_s2 else r3.get(p - n_s2)
                a2 = a2 or ("?", "?", "?", "?")
                esc = lambda x: (x or "∅").replace("|", "/")
                lines.append(f"| {esc(a1[1])} | {esc(a1[2])} | {esc(a2[1])} | {esc(a2[2])} |")
        (out_dir / "examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    LOG.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
