"""EXP-N-01 feasibility probe (READ-ONLY, diagnostic): can a transliteration token map be
learned from the PROVIDED training pairs alone, and would it cover validation pairs?

Learning (train-fold S1 only; validation excluded):
  for true pairs where the S1 name is Latin and the S2/S3 name is non-Latin and both
  name_core have the same number of tokens, align tokens by position and count
  (non-Latin token -> Latin token). Keep the most frequent Latin target per non-Latin token
  if it has support >= MIN_SUPPORT and share >= MIN_SHARE.
Evaluation (validation S1 true pairs with the same script mismatch):
  token coverage, and name_core Jaccard before vs after mapping.
No external dictionaries, no transliteration libraries, no internet.

    python scripts/analysis/translit_feasibility.py --config configs/baseline_v0.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_exp_b01 import load_views  # noqa: E402

from ber.config import Paths, load_config  # noqa: E402
from ber.data.normalize_records import load_manifest  # noqa: E402
from ber.data.split import load_val_ids  # noqa: E402
from ber.pipeline.run_baseline import _ids, pool_ids  # noqa: E402
from ber.utils.runtime import configure_arrow, write_json  # noqa: E402

MIN_SUPPORT, MIN_SHARE = 3, 0.6


def gt_rows(paths, s1_subset: pa.Array):
    s1_ids, p_ids = _ids(paths, "train", "source1"), pool_ids(paths, "train")
    gt = pq.read_table(paths.interim_file("train", "ground_truth"))
    gt = gt.filter(pc.is_in(gt.column("source1_entity_id"), value_set=s1_subset))
    lists = pc.split_pattern(gt.column("matched_entity_ids"), ",")
    flat, parent = pc.list_flatten(lists), pc.list_parent_indices(lists)
    keep = pc.not_equal(flat, "")
    s = pc.index_in(pc.take(gt.column("source1_entity_id"), pc.filter(parent, keep)), value_set=s1_ids).to_numpy(zero_copy_only=False)
    p = pc.index_in(pc.filter(flat, keep), value_set=p_ids).to_numpy(zero_copy_only=False)
    return s.astype(np.int64), p.astype(np.int64)


def mismatch_pairs(paths, s, p, n_s2):
    v1 = load_views(paths, "train", "source1", np.unique(s))
    v2 = load_views(paths, "train", "source2", p[p < n_s2])
    v3 = load_views(paths, "train", "source3", p[p >= n_s2] - n_s2)
    out = []
    for a, b in zip(s, p):
        x = v1.get(int(a))
        y = v2.get(int(b)) if b < n_s2 else v3.get(int(b - n_s2))
        if x and y and x[2] == 0 and y[2] in (1, 2):  # S1 Latin, pool non-Latin / mixed
            out.append((x[1], y[1]))  # name_core
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline_v0.yaml")
    ap.add_argument("--train-s1", type=int, default=150_000)
    a = ap.parse_args()
    configure_arrow()
    cfg = load_config(a.config)
    paths = Paths(cfg)
    n_s2 = int(load_manifest(paths, "train")["source2"]["rows"])
    val = set(load_val_ids(paths))
    all_s1 = _ids(paths, "train", "source1").to_pylist()
    rng = np.random.default_rng(cfg["seed"] + 7)
    fold = [i for i in all_s1 if i not in val]
    fold = [fold[i] for i in rng.choice(len(fold), min(a.train_s1, len(fold)), replace=False)]

    # ---- learn
    s, p = gt_rows(paths, pa.array(fold, pa.string()))
    train_pairs = mismatch_pairs(paths, s, p, n_s2)
    counts: dict[str, Counter] = defaultdict(Counter)
    aligned = 0
    for lat, non in train_pairs:
        lt, nt = lat.split(), non.split()
        if len(lt) == len(nt):
            aligned += 1
            for x, y in zip(nt, lt):
                if not x.isascii():
                    counts[x][y] += 1
    table = {}
    for tok, c in counts.items():
        tgt, n = c.most_common(1)[0]
        if n >= MIN_SUPPORT and n / sum(c.values()) >= MIN_SHARE:
            table[tok] = tgt

    # ---- evaluate on validation
    s, p = gt_rows(paths, pa.array(sorted(val), pa.string()))
    val_pairs = mismatch_pairs(paths, s, p, n_s2)

    def jac(x, y):
        a_, b_ = set(x.split()), set(y.split())
        return len(a_ & b_) / max(len(a_ | b_), 1)

    tok_total = tok_cov = 0
    before, after, exact_after = [], [], 0
    for lat, non in val_pairs:
        mapped = []
        for t in non.split():
            if t.isascii():
                mapped.append(t)
            else:
                tok_total += 1
                if t in table:
                    tok_cov += 1
                mapped.append(table.get(t, t))
        m = " ".join(mapped)
        before.append(jac(lat, non))
        after.append(jac(lat, m))
        exact_after += m == lat
    report = {
        "train_fold_s1_sampled": len(fold),
        "train_script_mismatch_true_pairs": len(train_pairs),
        "train_pairs_alignable_equal_token_count": aligned,
        "dictionary_size": len(table),
        "min_support": MIN_SUPPORT, "min_share": MIN_SHARE,
        "val_script_mismatch_true_pairs": len(val_pairs),
        "val_non_latin_token_coverage": tok_cov / max(tok_total, 1),
        "val_name_core_jaccard_before": float(np.mean(before)) if before else None,
        "val_name_core_jaccard_after": float(np.mean(after)) if after else None,
        "val_name_core_exact_after": exact_after / max(len(val_pairs), 1),
        "examples": list(table.items())[:15],
    }
    out = paths.run_dir("EXP-N-01") / "feasibility.json"
    write_json(out, report)
    for k, v in report.items():
        if k != "examples":
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
