"""Compare the fate of every validation true pair under two blocker configs (READ-ONLY).

Inputs: two .npz dumps from scripts/analysis/blocking_forensics.py --dump (same frozen validation
split, so the true-pair arrays are identical and aligned). Reports, overall and per country:
  recovered   never retrieved by A (any rank)  -> retrieved by B (any rank / inside top-K)
  entering    outside A's top-K                -> inside B's top-K
  pushed out  inside A's top-K                 -> outside B's top-K (top-K displacement)
  net top-K change, and the script-mismatch share of recovered / still-unreachable pairs.

    python scripts/analysis/compare_blocking_fates.py --a A.npz --b B.npz --top-k 20 --out X.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_exp_b01 import load_views  # noqa: E402

from ber.config import Paths, load_config  # noqa: E402
from ber.data.normalize_records import load_manifest  # noqa: E402
from ber.utils.runtime import configure_arrow, setup_logging, write_json  # noqa: E402

NEVER = 10**6


def script_mismatch(paths, gs, gp, n_s2):
    """Latin S1 name vs non-Latin/mixed pool name, per pair (name_script 0 vs 1/2)."""
    v1 = load_views(paths, "train", "source1", np.unique(gs))
    v2 = load_views(paths, "train", "source2", np.unique(gp[gp < n_s2]))
    v3 = load_views(paths, "train", "source3", np.unique(gp[gp >= n_s2] - n_s2))
    out = np.zeros(len(gs), bool)
    for i, (s, p) in enumerate(zip(gs.tolist(), gp.tolist())):
        x = v1.get(s)
        y = v2.get(p) if p < n_s2 else v3.get(p - n_s2)
        out[i] = bool(x and y and x[2] == 0 and y[2] in (1, 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--config", default="configs/exp_b04_blocking_v2.yaml")
    ap.add_argument("--out", required=True)
    x = ap.parse_args()
    setup_logging()
    configure_arrow()
    A, B = np.load(x.a), np.load(x.b)
    if not np.array_equal(A["gcode"], B["gcode"]):
        raise SystemExit("true-pair arrays differ: dumps are not from the same validation split")
    ra, rb, country = A["g_rank"], B["g_rank"], A["g_country"]
    K = x.top_k
    fa, fb = ra < NEVER, rb < NEVER
    ka, kb = ra < K, rb < K
    sets = {"recovered_any_rank": ~fa & fb, "recovered_into_topk": ~fa & kb, "lost_any_rank": fa & ~fb,
            "entering_topk": ~ka & kb, "pushed_out_of_topk": ka & ~kb,
            "never_retrieved_A": ~fa, "never_retrieved_B": ~fb, "in_topk_A": ka, "in_topk_B": kb}
    paths = Paths(load_config(x.config))
    n_s2 = int(load_manifest(paths, "train")["source2"]["rows"])
    interesting = sets["recovered_any_rank"] | sets["never_retrieved_B"] | sets["entering_topk"]
    sm = np.zeros(len(ra), bool)
    idx = np.flatnonzero(interesting)
    sm[idx] = script_mismatch(paths, A["gs"][idx], A["gp"][idx], n_s2)
    rep = {"a": x.a, "b": x.b, "top_k": K, "true_pairs": int(len(ra))}
    for c in ["all"] + sorted(set(country.tolist())):
        sel = np.ones(len(ra), bool) if c == "all" else country == c
        r = {k: int((v & sel).sum()) for k, v in sets.items()}
        r["net_topk_change"] = r["entering_topk"] - r["pushed_out_of_topk"]
        r["topk_recall_A"] = float(ka[sel].mean())
        r["topk_recall_B"] = float(kb[sel].mean())
        r["uncapped_recall_A"] = float(fa[sel].mean())
        r["uncapped_recall_B"] = float(fb[sel].mean())
        r["script_mismatch"] = {
            "recovered_any_rank": int((sets["recovered_any_rank"] & sel & sm).sum()),
            "entering_topk": int((sets["entering_topk"] & sel & sm).sum()),
            "never_retrieved_B": int((sets["never_retrieved_B"] & sel & sm).sum()),
            "never_retrieved_B_share": float((sets["never_retrieved_B"] & sel & sm).sum() / max((sets["never_retrieved_B"] & sel).sum(), 1)),
            "share_of_all_true_pairs_unreachable_by_script": float((sets["never_retrieved_B"] & sel & sm).sum() / max(sel.sum(), 1)),
        }
        rep[c] = r
    Path(x.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(x.out), rep)
    for c in ("all", "india", "us"):
        if c in rep:
            r = rep[c]
            print(c, {k: r[k] for k in ("recovered_any_rank", "recovered_into_topk", "entering_topk", "pushed_out_of_topk", "net_topk_change")},
                  round(r["topk_recall_A"], 4), "->", round(r["topk_recall_B"], 4), "| uncapped", round(r["uncapped_recall_A"], 4), "->",
                  round(r["uncapped_recall_B"], 4), "| script", r["script_mismatch"])


if __name__ == "__main__":
    main()
