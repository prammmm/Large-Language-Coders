"""EXP-B-09 (name-view token x house-number key) analysis. Deterministic, seed 42. Read-only w.r.t. EXP-B-06/B-08/S-01/S-02
artefacts and root output/.

    python scripts/analysis/exp_b09.py preflight | blockdiag

preflight   reference reproduction check: the B-06 `train` stage re-run under PYTHONNOUSERSITE=1 (scored tag
            preflight_B06_nousersite) vs the stored B-06 validation parts, field by field; the re-learned cross-script map;
            the recomputed B-06 test decision hash.
blockdiag   blocker-only kill gate on the frozen validation fold: B-09 blocker run 1000 deep vs B-06 (stored top-20
            validation lists + FINAL_RESEARCH_AUDIT/fates_B06_k1000.npz). Recall@20, uncapped recall, new / displaced
            candidates and true pairs, name_num block-cap loss, India / US, candidate counts, runtime, peak memory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_s02_shadow as X  # noqa: E402

from ber.blocking.blocker import PASS_ORDER, CountryBlocker  # noqa: E402
from ber.config import Paths, load_config  # noqa: E402
from ber.data.normalize_records import load_manifest, read_country  # noqa: E402
from ber.data.split import load_val_ids  # noqa: E402
from ber.pipeline.run_baseline import VIEW_COLS, LazyPool, _ids, resolve_xmap, with_xlit  # noqa: E402
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, write_json  # noqa: E402

RUN = Path("experiments/runs/EXP-B-10")
OUT = RUN / "analysis"
CFG = "configs/exp_b10_name_pair3.yaml"
CFG_B06 = "configs/exp_b06_name_pair.yaml"
SHIFT = np.int64(1) << np.int64(24)
NN_BIT = 1 << PASS_ORDER.index("name_num")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def load_parts(tag: str, split: str = "train") -> pa.Table:
    paths = Paths(load_config(CFG_B06))
    t = pa.concat_tables([pq.read_table(p) for p in sorted(paths.scored_dir(split, tag).glob("*.parquet"))])
    code = pc.add(pc.multiply(t.column("s1_row").cast(pa.int64()), int(SHIFT)), t.column("pool_row").cast(pa.int64()))
    return t.append_column("_code", code).sort_by("_code")


# ----------------------------------------------------------------------------- preflight
def cmd_preflight(c):
    import numpy
    ref, new = load_parts("scored_EXP-B-06"), load_parts("preflight_B06_nousersite")
    rep = {"numpy_used_for_rerun": "2.2.6 (env, PYTHONNOUSERSITE=1)", "numpy_used_for_references": "2.2.4 (user site)",
           "this_process_numpy": numpy.__version__, "rows": {"stored": ref.num_rows, "rerun": new.num_rows}, "columns": {}}
    same_rows = ref.num_rows == new.num_rows and ref.column("_code").equals(new.column("_code"))
    rep["pair_set_identical"] = bool(same_rows)
    if same_rows:
        for col in ref.column_names:
            a, b = ref.column(col).to_numpy(), new.column(col).to_numpy()
            if np.issubdtype(a.dtype, np.floating):
                eq = np.array_equal(a, b, equal_nan=True)
                rep["columns"][col] = {"identical": bool(eq), "max_abs_diff": float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if not eq else 0.0}
            else:
                rep["columns"][col] = {"identical": bool(np.array_equal(a, b))}
    pre = Path("experiments/runs/EXP-B-09_preflight_B06")
    rep["xlit_map_identical"] = (pre / "xlit_map.json").read_bytes() == Path("experiments/runs/EXP-B-06/xlit_map.json").read_bytes() if (pre / "xlit_map.json").exists() else "no map written"
    m_ref = json.loads(Path("experiments/runs/EXP-B-06/metrics.json").read_text(encoding="utf-8"))["at_threshold_used"]["f05"]
    m_new = json.loads((pre / "metrics.json").read_text(encoding="utf-8"))["at_threshold_used"]["f05"]
    rep["validation_f05"] = {"stored": m_ref, "rerun": m_new, "identical": m_ref == m_new}
    rep["test_output_sha256"] = {f: {"stored": sha(Path("experiments/runs/EXP-B-06/output") / f), "rerun": sha(pre / "output" / f)}
                                 for f in ("matching_results.tsv", "candidate_pairs.tsv")}
    rep["test_output_identical"] = all(v["stored"] == v["rerun"] for v in rep["test_output_sha256"].values())
    rep["reproduces_exactly"] = bool(same_rows and all(v["identical"] for v in rep["columns"].values()) and rep["validation_f05"]["identical"]
                                     and rep["test_output_identical"] and rep["xlit_map_identical"] is True)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "preflight_reproduction.json", rep)
    LOG.info("preflight: reproduces_exactly=%s | %s", rep["reproduces_exactly"], {k: v for k, v in rep.items() if k != "columns"})


# ----------------------------------------------------------------------------- blocking-only diagnostic (kill gate)
def run_blocker(cfg, paths, val_ids, top_k):
    bcfg = copy.deepcopy(cfg["blocking"])
    bcfg["top_k"] = top_k
    xmap = resolve_xmap(cfg, paths)
    man = load_manifest(paths, "train")
    n_s2 = int(man["source2"]["rows"])
    val_arr = pa.array(val_ids, pa.string())
    codes, ranks, flags, scores, stats = [], [], [], [], {}
    for key in sorted(man["source1"]["countries"]):
        s1 = read_country(paths, "train", "source1", key, VIEW_COLS)
        s1 = s1.filter(pc.is_in(s1.column("entity_id"), value_set=val_arr))
        prow = np.concatenate([read_country(paths, "train", src, key, ["row"]).column("row").to_numpy().astype(np.int64) + (n_s2 if src == "source3" else 0)
                               for src in ("source2", "source3")])
        srow = s1.column("row").to_numpy().astype(np.int64)
        b = CountryBlocker(with_xlit(s1, xmap), LazyPool(paths, "train", key, xmap), bcfg)
        stats[key] = b.stats["passes"]
        for lo, hi in b.chunk_bounds():
            cc = b.candidates(lo, hi)
            codes.append(srow[cc["s1_local"]] * SHIFT + prow[cc["pool_local"]])
            ranks.append(cc["cheap_rank"].astype(np.int32))
            flags.append(cc["pass_flags"].astype(np.uint16))
            scores.append(cc["cheap_score"])
        del b, s1
    return np.concatenate(codes), np.concatenate(ranks), np.concatenate(flags), np.concatenate(scores), stats


def cmd_blockdiag(c):
    cfg = load_config(CFG)
    paths = Paths(cfg)
    val_ids = load_val_ids(paths)
    t0 = time.perf_counter()
    with PeakMemory() as pm:
        code, rank, flag, cs, stats = run_blocker(cfg, paths, val_ids, 1000)
    secs = time.perf_counter() - t0
    f06 = np.load("experiments/runs/EXP-B-09/analysis/fates_B09_k1000.npz")
    gcode, r06, gctry = f06["gcode"].astype(np.int64), f06["g_rank"], f06["g_country"]
    o = np.argsort(code)
    cs_, rk_, fl_ = code[o], rank[o], flag[o]
    pos = np.minimum(np.searchsorted(cs_, gcode), len(cs_) - 1)
    found = cs_[pos] == gcode
    r09 = np.where(found, rk_[pos], 10**6)
    f09 = np.where(found, fl_[pos], 0)
    # B-06 stored top-20 validation lists (exact candidate sets the B-06 model scored)
    ref = load_parts("scored_EXP-B-09")
    c06 = ref.column("_code").to_numpy()
    top09 = np.sort(code[rank < 20])
    new = np.setdiff1d(top09, c06)
    disp = np.setdiff1d(c06, top09)
    gts = np.sort(gcode)
    istrue = lambda x: np.isin(x, gts)  # noqa: E731
    val = np.zeros(c.n, bool)
    val[pc.index_in(pa.array(val_ids, pa.string()), value_set=_ids(paths, "train", "source1")).to_numpy(zero_copy_only=False).astype(np.int64)] = True
    nval = int(val.sum())
    rep = {"population": "frozen validation fold (220,682 S1; 763,556 true pairs)", "runtime_s": round(secs, 1), "peak_gb": round(pm.peak_gb, 2),
           "B09_reference(named B06 in keys below)": "stored scored_EXP-B-09 top-20 lists + EXP-B-09/analysis/fates_B09_k1000.npz"}
    for lab, m in (("all", np.ones(len(gcode), bool)), ("india", gctry == "india"), ("us", gctry == "us")):
        rep[f"recall_{lab}"] = {"B06@20": float((r06[m] < 20).mean()), "B09@20": float((r09[m] < 20).mean()),
                                "B06@30": float((r06[m] < 30).mean()), "B09@30": float((r09[m] < 30).mean()),
                                "B06_uncapped(any rank<1000)": float((r06[m] < 1000).mean()), "B09_uncapped(any rank<1000)": float((r09[m] < 1000).mean()),
                                "delta@20": float((r09[m] < 20).mean() - (r06[m] < 20).mean())}
    nk = r06 >= 1000
    rep["no_key_pairs(B06)"] = {"pairs": int(nk.sum()), "now_generated_any_rank": int((r09[nk] < 1000).sum()), "now_in_top20": int((r09[nk] < 20).sum()),
                                "generated_by_name_num": int(((f09[nk] & NN_BIT) > 0).sum())}
    rep["top20_lists"] = {"B06_candidates": int(len(c06)), "B09_candidates": int(len(top09)), "new_candidates": int(len(new)), "new_true": int(istrue(new).sum()),
                          "new_false": int(len(new) - istrue(new).sum()), "displaced_existing": int(len(disp)), "displaced_true": int(istrue(disp).sum()),
                          "net_true": int(istrue(new).sum() - istrue(disp).sum()), "new_per_s1": len(new) / nval, "displaced_per_s1": len(disp) / nval}
    per_s1 = np.bincount((code[rank < 20] // SHIFT), minlength=c.n)[val]
    rep["candidate_count_distribution_top20"] = {"mean": float(per_s1.mean()), "median": float(np.median(per_s1)), "p95": float(np.quantile(per_s1, 0.95)),
                                                 "max": int(per_s1.max()), "zero": int((per_s1 == 0).sum())}
    uncapped = np.bincount(code // SHIFT, minlength=c.n)[val]
    rep["candidate_count_uncapped(rank<1000)"] = {"mean": float(uncapped.mean()), "B06_mean(B09 here)": None}
    rep["name_num_pass_stats"] = {k: v.get("name_num") for k, v in stats.items()}
    rep["name_num_true_pairs_in_top20"] = int((((f09 & NN_BIT) > 0) & (r09 < 20)).sum())
    rep["name_num_only_true_pairs_in_top20"] = int(((f09 == NN_BIT) & (r09 < 20)).sum())
    d = rep["recall_all"]["delta@20"]
    rep["kill_gate"] = {"rule": "STOP before retrain if blocking recall@20 gain over B-09 < +0.003 (pre-declared)", "delta@20": d, "pass": bool(d >= 0.003)}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "blocking_forensics.json", rep)
    np.savez_compressed(OUT / "fates_B10_k1000.npz", gcode=gcode, g_rank=r09, g_flags=f09, g_country=gctry)
    LOG.info("blockdiag: recall@20 %.4f -> %.4f (delta %+.4f) | top20 new %d (true %d) displaced %d (true %d) | %.0fs peak %.2f GB | gate %s",
             rep["recall_all"]["B06@20"], rep["recall_all"]["B09@20"], d, rep["top20_lists"]["new_candidates"], rep["top20_lists"]["new_true"],
             rep["top20_lists"]["displaced_existing"], rep["top20_lists"]["displaced_true"], secs, pm.peak_gb, rep["kill_gate"]["pass"])


CMDS = {"preflight": cmd_preflight, "blockdiag": cmd_blockdiag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=list(CMDS))
    a = ap.parse_args()
    setup_logging()
    configure_arrow()
    t0 = time.perf_counter()
    CMDS[a.cmd](X.Ctx())
    LOG.info("%s done in %.0fs", a.cmd, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
