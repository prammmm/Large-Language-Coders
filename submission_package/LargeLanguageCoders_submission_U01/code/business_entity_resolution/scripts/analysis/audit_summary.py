"""READ-ONLY audit: collect the authoritative measured numbers of every experiment from its run
artifacts (metrics.json, full-competition JSONs, fit reports, submission summaries, output hashes),
so reports can be checked against evidence instead of memory.

    python scripts/analysis/audit_summary.py --out experiments/runs/FINAL/audit_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RUNS = Path("experiments/runs")


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def sha(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:16]


def pick(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/runs/FINAL/audit_summary.json")
    a = ap.parse_args()
    out = {}
    for exp in ("EXP-B-01", "EXP-F-01b", "EXP-M-03", "EXP-B-03a", "EXP-B-04", "EXP-B-05", "EXP-N-02", "EXP-N-03", "EXP-B-07", "EXP-B-06"):
        d = RUNS / exp
        m = load(d / "metrics.json")
        u = pick(m, "at_threshold_used") or pick(m, "at_tuned_threshold") or {}
        model = load(d / "model.json")
        rec = {"validation_no_competition": {k: u.get(k) for k in ("f05", "macro_precision", "macro_recall", "blocking_recall", "oracle_f05",
                                                                  "false_merges", "singleton_false_positive", "predicted_per_s1")},
               "threshold_used": pick(m, "threshold_used"), "model_threshold": pick(model, "meta", "threshold"),
               "by_country": {c: v.get("f05") for c, v in (u.get("by_country") or {}).items()}}
        for name in ("full_competition_unrestricted.json", "full_competition.json"):
            fc = load(d / name)
            if fc:
                rec["full_competition_file"] = name
                rec["E_unrestricted_f05"] = pick(fc, "variants", "unrestricted", "E", "f05")
                rec["E_D01_f05"] = pick(fc, "variants", "best_claimant_margin_0.00", "E", "f05")
                rec["E_D01_precision"] = pick(fc, "variants", "best_claimant_margin_0.00", "E", "macro_precision")
                rec["sanity"] = fc.get("sanity_validation_unrestricted_equals_production")
                break
        cmp_ = load(d / "full_competition_comparison.json")
        if cmp_:
            rec["comparison_arms"] = list(cmp_.get("arms", {}))
            res = cmp_.get("results", {})
            rec["comparison_E"] = {k: {x: v.get(x) for x in ("f05", "macro_precision", "macro_recall", "blocking_recall", "oracle_f05",
                                                             "false_merges", "singleton_false_positive")} for k, v in res.get("E", {}).items()}
            rec["comparison_halves"] = {p: {k: res[p][k]["f05"] for k in res[p]} for p in res if p != "E"}
        s = load(d / "submission_summary.json")
        if s:
            rec["submission_summary"] = {"threshold": s.get("threshold"), "matched_ids": pick(s, "matching_file", "ids"),
                                         "strict_errors": s.get("strict_validation_errors"), "competition_after": pick(s, "competition", "after", "conflicted_records")}
        for od in ("output", "output_t067"):
            if (d / od / "matching_results.tsv").exists():
                rec[f"{od}_sha"] = {f: sha(d / od / f) for f in ("candidate_pairs.tsv", "matching_results.tsv")}
        out[exp] = rec
    ed = load(RUNS / "EXP-D-01" / "full_competition_eval.json")
    if ed:
        out["EXP-D-01"] = {"E_control": pick(ed, "results", "E", "control_B04", "f05"), "E_D01": pick(ed, "results", "E", "EXP_D01", "f05"),
                           "E_D01_precision": pick(ed, "results", "E", "EXP_D01", "macro_precision")}
    out["root_output_sha"] = {f: sha(Path("output") / f) for f in ("candidate_pairs.tsv", "matching_results.tsv")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for k, v in out.items():
        if isinstance(v, dict) and "validation_no_competition" in v:
            print(f"{k:10s} val(noComp) {v['validation_no_competition'].get('f05')} | E unrestr {v.get('E_unrestricted_f05')} | E+D01 {v.get('E_D01_f05')} "
                  f"(P {v.get('E_D01_precision')}) | t_used {v.get('threshold_used')} | sub_t {pick(v, 'submission_summary', 'threshold')}")
        else:
            print(k, v)


if __name__ == "__main__":
    main()
