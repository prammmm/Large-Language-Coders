"""Independent check of EXP-D-01 test outputs against the EXP-B-04 control outputs (streamed).

Invariants: candidate files byte-identical; same S1 rows in the same order; every EXP-D-01 match
list is an order-preserving subset of the B-04 list and of the candidates; no S2/S3 id in two
rows; files are UTF-8 without BOM with LF line endings.

    python scripts/analysis/verify_d01_outputs.py --control output --exp experiments/runs/EXP-D-01/output
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rows(p: Path):
    with open(p, "rb") as f:
        for line in f:
            s1, _, ids = line.rstrip(b"\n").partition(b"\t")
            yield s1, [x for x in ids.split(b",") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", default="output")
    ap.add_argument("--exp", default="experiments/runs/EXP-D-01/output")
    ap.add_argument("--out", default="experiments/runs/EXP-D-01/output_invariants.json")
    a = ap.parse_args()
    c, e = Path(a.control), Path(a.exp)
    rep = {"control": str(c), "exp": str(e),
           "sha256": {f"{d}/{n}": sha(d / n) for d in (c, e) for n in ("candidate_pairs.tsv", "matching_results.tsv")}}
    rep["candidate_files_identical"] = rep["sha256"][f"{c}/candidate_pairs.tsv"] == rep["sha256"][f"{e}/candidate_pairs.tsv"]
    for p in (e / "candidate_pairs.tsv", e / "matching_results.tsv"):
        head = p.read_bytes()[:3] if p.stat().st_size < 10 else open(p, "rb").read(3)
        with open(p, "rb") as f:
            cr = any(b"\r" in blk for blk in iter(lambda: f.read(1 << 22), b""))
        rep[f"{p.name}_bom"] = head == b"\xef\xbb\xbf"
        rep[f"{p.name}_has_CR"] = cr
    n = same_rows = subset_ctrl = subset_cand = order_ok = 0
    removed = kept = 0
    seen: set[bytes] = set()
    dup = 0
    for (s_c, m_c), (s_e, m_e), (s_k, cand) in zip(rows(c / "matching_results.tsv"), rows(e / "matching_results.tsv"), rows(e / "candidate_pairs.tsv")):
        n += 1
        same_rows += s_c == s_e == s_k
        if n == 1:  # header row: the second column names differ by design
            subset_ctrl, subset_cand, order_ok = subset_ctrl + 1, subset_cand + 1, order_ok + 1
            continue
        sc, sk = set(m_c), set(cand)
        subset_ctrl += all(x in sc for x in m_e)
        subset_cand += all(x in sk for x in m_e)
        keep = set(m_e)
        order_ok += [x for x in m_c if x in keep] == m_e
        removed += len(m_c) - len(m_e)
        if n > 1:  # skip header
            kept += len(m_e)
            for x in m_e:
                if x in seen:
                    dup += 1
                seen.add(x)
    rep.update({"rows_incl_header": n, "same_s1_rows_in_same_order": same_rows == n, "matches_subset_of_control": subset_ctrl == n,
                "matches_subset_of_candidates": subset_cand == n, "kept_order_preserved": order_ok == n,
                "pairs_removed_vs_control": removed, "pairs_kept": kept, "ids_assigned_to_two_rows": dup})
    rep["ALL_OK"] = (rep["candidate_files_identical"] and rep["same_s1_rows_in_same_order"] and rep["matches_subset_of_control"]
                     and rep["matches_subset_of_candidates"] and rep["kept_order_preserved"] and dup == 0
                     and not any(v for k, v in rep.items() if k.endswith(("_bom", "_has_CR"))))
    Path(a.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
