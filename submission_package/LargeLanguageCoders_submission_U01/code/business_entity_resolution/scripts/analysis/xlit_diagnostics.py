"""EXP-N-02 Stage 2: feasibility diagnostics of the train-learned cross-script map (READ-ONLY).

Learns (or loads) the FINAL map from the 200k fit S1 (validation excluded, enforced), learns the
five cross-fit maps (sizes only), then MEASURES coverage on frozen-validation true pairs whose pool
name is non-Latin / mixed. Validation labels are used only for measurement, never for learning.

    python scripts/analysis/xlit_diagnostics.py --config configs/exp_n02_transliteration.yaml \
        [--b04-fates <blocking_forensics --dump of B-04>.npz]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from ber.config import Paths, load_config
from ber.data.normalize_records import load_manifest, read_country
from ber.data.split import load_val_ids
from ber.normalization.crossscript import apply_map_one, pair_tokens, token_script
from ber.pipeline.run_baseline import _ids, fit_s1_subset
from ber.pipeline.xlit import crossfit_plan, ensure_final_map, learn_map_for, train_pairs
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, write_json


def jac(a: str, b: str) -> float:
    x, y = set(a.split()), set(b.split())
    return len(x & y) / max(len(x | y), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_n02_transliteration.yaml")
    ap.add_argument("--b04-fates", default=None)
    ap.add_argument("--out", default="experiments/runs/EXP-N-02/xlit_feasibility.json")
    ap.add_argument("--skip-folds", action="store_true")
    a = ap.parse_args()
    setup_logging()
    configure_arrow()
    cfg = load_config(a.config)
    paths = Paths(cfg)
    with PeakMemory() as pm:
        fm = ensure_final_map(cfg, paths)
    m = fm.mapping
    rep = {"final_map": {k: v for k, v in fm.meta.items()}, "final_map_learn_seconds": round(pm.seconds, 1), "peak_rss_gb": round(pm.peak_gb, 2)}
    if not a.skip_folds:
        fit = fit_s1_subset(cfg, paths)
        rep["crossfit_maps"] = []
        for f, scored, learn in crossfit_plan(cfg, fit):
            x = learn_map_for(cfg, paths, learn, f"diag fold {f}")
            rep["crossfit_maps"].append({"fold": f, "scored_s1": len(scored), "learned_from_s1": len(learn), "mappings": len(x.mapping)})

    # ---- measurement on validation true pairs (non-Latin / mixed pool names)
    val = load_val_ids(paths)
    country_of = {}
    s1_all = _ids(paths, "train", "source1")
    for k in load_manifest(paths, "train")["source1"]["countries"]:
        ids = read_country(paths, "train", "source1", k, ["entity_id"]).column("entity_id").to_pylist()
        country_of.update(dict.fromkeys(ids, k))
    tot_tok = cov_tok = 0
    tok_types, cov_types = set(), set()
    per = Counter()
    jb, ja = [], []
    for sid, s1n, pn in train_pairs(paths, val):
        c = country_of.get(sid, "?")
        lat, non = pair_tokens(s1n, pn)
        per[f"{c}|pairs"] += 1
        if not non:
            continue
        toks = [t for t in pn.split() if token_script(t) == "nonlatin"]
        tot_tok += len(toks)
        hit = [t for t in toks if t in m]
        cov_tok += len(hit)
        tok_types.update(toks)
        cov_types.update(hit)
        x = apply_map_one(pn, m)
        b0, b1 = jac(s1n, pn), jac(s1n, x)
        jb.append(b0)
        ja.append(b1)
        per[f"{c}|any_token_mapped"] += bool(hit)
        per[f"{c}|all_nonlatin_mapped"] += len(hit) == len(toks)
        per[f"{c}|xlit_shares_token_with_s1"] += b1 > 0
        per[f"{c}|xlit_equals_s1_core"] += x == s1n
    n = len(jb)
    rep["validation_measurement"] = {
        "label": "measurement only (validation labels never used for learning)",
        "val_true_pairs_with_nonlatin_pool_name": n,
        "nonlatin_token_occurrences": tot_tok, "token_coverage": cov_tok / max(tot_tok, 1),
        "nonlatin_token_types": len(tok_types), "token_type_coverage": len(cov_types) / max(len(tok_types), 1),
        "pair_any_token_mapped": sum(v for k, v in per.items() if k.endswith("any_token_mapped")) / max(n, 1),
        "pair_xlit_shares_token_with_s1": sum(v for k, v in per.items() if k.endswith("shares_token_with_s1")) / max(n, 1),
        "pair_xlit_equals_s1_core": sum(v for k, v in per.items() if k.endswith("equals_s1_core")) / max(n, 1),
        "name_core_jaccard_mean_before": float(np.mean(jb)) if jb else None,
        "name_core_jaccard_mean_after": float(np.mean(ja)) if ja else None,
        "by_country_counts": dict(per),
    }
    # ---- known unreachable script-mismatch pairs (B-04 never retrieved)
    if a.b04_fates and Path(a.b04_fates).exists():
        F = np.load(a.b04_fates)
        never = F["g_rank"] >= 10**6
        gs, gp = F["gs"][never], F["gp"][never]
        n_s2 = int(load_manifest(paths, "train")["source2"]["rows"])
        from ber.pipeline.xlit import _read_names

        s1n = _read_names(paths, "train", "source1", gs, False)
        p2 = _read_names(paths, "train", "source2", gp[gp < n_s2], True)
        p3 = _read_names(paths, "train", "source3", gp[gp >= n_s2] - n_s2, True)
        cnt = Counter()
        for s, p in zip(gs.tolist(), gp.tolist()):
            pn = p2.get(p) if p < n_s2 else p3.get(p - n_s2)
            if pn is None:
                continue  # Latin pool name: not a script-mismatch pair
            a_ = s1n.get(s, "")
            if not pair_tokens(a_, pn)[0]:
                continue
            cnt["script_mismatch_never_retrieved"] += 1
            x = apply_map_one(pn, m)
            cnt["any_token_mapped"] += x != pn
            cnt["xlit_shares_token_with_s1"] += jac(a_, x) > 0
            cnt["xlit_equals_s1_core"] += x == a_
        rep["b04_never_retrieved_script_mismatch"] = dict(cnt)
    rep["examples"] = sorted(m.items())[:40]  # gitignored run dir only
    write_json(Path(a.out), rep)
    LOG.info("final map: %d mappings (%s rejected) | val token coverage %.3f | pair any-mapped %.3f | shares token %.3f | jaccard %.3f -> %.3f",
             len(m), fm.meta.get("rejected"), rep["validation_measurement"]["token_coverage"], rep["validation_measurement"]["pair_any_token_mapped"],
             rep["validation_measurement"]["pair_xlit_shares_token_with_s1"], rep["validation_measurement"]["name_core_jaccard_mean_before"] or 0,
             rep["validation_measurement"]["name_core_jaccard_mean_after"] or 0)
    if "b04_never_retrieved_script_mismatch" in rep:
        LOG.info("B-04 never-retrieved script-mismatch pairs: %s", rep["b04_never_retrieved_script_mismatch"])
    if "crossfit_maps" in rep:
        LOG.info("cross-fit maps: %s", [(x["fold"], x["learned_from_s1"], x["mappings"]) for x in rep["crossfit_maps"]])


if __name__ == "__main__":
    sys.exit(main())
