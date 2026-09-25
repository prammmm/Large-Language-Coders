"""EXP-B-02 blocking sweep: one controlled change per variant, blocking only, frozen
validation split. Each variant runs as its own process (clean memory measurement) and
writes experiments/runs/EXP-B-02-<variant>/blocking_eval.json; this script then writes a
summary table to experiments/runs/EXP-B-02/blocking_sweep.{json,md}.

    python scripts/experiments/run_blocking_sweep.py                 # all variants
    python scripts/experiments/run_blocking_sweep.py --only v0 a b   # a subset
    python scripts/experiments/run_blocking_sweep.py --variants-file my.json   # custom combos
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BASE = ["blocking.top_k=30"]  # rank-ordering is independent of K: recall@20 == the K=20 system
VARIANTS: dict[str, tuple[str, list[str]]] = {
    "v0": ("baseline blocking (EXP-B-01 keys)", []),
    "a": ("+ exact name_core key", ["blocking.passes.exact_core.enabled=true"]),
    "b": ("+ sorted name_core key", ["blocking.passes.sorted_core.enabled=true"]),
    "c": ("+ strip leading zeros in address numbers", ["blocking.strip_leading_zeros=true"]),
    "d": ("+ name_token pass (DF<=200)", ["blocking.passes.name_token.enabled=true"]),
    "e100": ("addr_num_word max_block_size 50->100", ["blocking.passes.addr_num_word.max_block_size=100"]),
    "e150": ("addr_num_word max_block_size 50->150", ["blocking.passes.addr_num_word.max_block_size=150"]),
    "e200": ("addr_num_word max_block_size 50->200", ["blocking.passes.addr_num_word.max_block_size=200"]),
    "f3": ("max_nums 2->3", ["blocking.passes.addr_num_word.max_nums=3"]),
    "g6": ("max_words 4->6", ["blocking.passes.addr_num_word.max_words=6"]),
}


def summarise(runs: Path, names: list[str], variants: dict) -> tuple[list[dict], str]:
    rows = []
    base = None
    for v in names:
        f = runs / f"EXP-B-02-{v}" / "blocking_eval.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        row = {"variant": v, "change": variants[v][0], "runtime_s": r["runtime_seconds"], "peak_rss_gb": r["peak_rss_gb"]}
        for k in ("20", "30"):
            a = r["at_k"].get(k)
            if a:
                row[f"recall@{k}"] = a["blocking_recall"]
                row[f"cand_mean@{k}"] = a["candidates_per_s1"]["mean"]
                row[f"cand_p99@{k}"] = a["candidates_per_s1"]["p99"]
                row[f"pairs@{k}"] = a["candidate_pairs"]
                row[f"true_found@{k}"] = a["true_pairs_found"]
                for c, cv in a["by_country"].items():
                    row[f"{c}_recall@{k}"] = cv["blocking_recall"]
        row["contribution"] = r["pass_contribution_at_top_k"]
        if v == "v0":
            base = row
        rows.append(row)
    lines = ["| variant | change | recall@20 | Δ vs v0 | India@20 | US@20 | cand/S1@20 | p99@20 | new true pairs@20 | extra cand pairs@20 | recall@30 | cand/S1@30 | runtime s | peak GB |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d = r["recall@20"] - base["recall@20"] if base else float("nan")
        nt = r["true_found@20"] - base["true_found@20"] if base else 0
        xp = r["pairs@20"] - base["pairs@20"] if base else 0
        lines.append(
            f"| {r['variant']} | {r['change']} | {r['recall@20']:.4f} | {d:+.4f} | {r.get('india_recall@20', float('nan')):.4f} | "
            f"{r.get('us_recall@20', float('nan')):.4f} | {r['cand_mean@20']:.2f} | {r['cand_p99@20']:.0f} | {nt:+,} | {xp:+,} | "
            f"{r.get('recall@30', float('nan')):.4f} | {r.get('cand_mean@30', float('nan')):.2f} | {r['runtime_s']:.0f} | {r['peak_rss_gb']:.2f} |"
        )
    return rows, "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_v0.yaml")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--variants-file", default=None, help="JSON {name: [description, [--set items]]} to add/override variants")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    variants = dict(VARIANTS)
    if a.variants_file:
        variants.update({k: (v[0], v[1]) for k, v in json.loads(Path(a.variants_file).read_text(encoding="utf-8")).items()})
    names = a.only or list(variants)
    if "v0" not in names:
        names = ["v0"] + names
    if not a.summary_only:
        for v in names:
            sets = BASE + variants[v][1] + [f"experiment.id=EXP-B-02-{v}"]
            cmd = [sys.executable, "-m", "ber.pipeline.run_baseline", "block-eval", "--config", a.config]
            for s in sets:
                cmd += ["--set", s]
            print(">>", v, variants[v][0], flush=True)
            subprocess.run(cmd, check=True)
    runs = Path("experiments/runs")
    rows, md = summarise(runs, names, variants)
    out = runs / "EXP-B-02"
    out.mkdir(parents=True, exist_ok=True)
    (out / "blocking_sweep.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (out / "blocking_sweep.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
