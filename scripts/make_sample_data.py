"""Build a small, proportional sample of the real data for fast development runs.

Raw files under datasets/ are only READ (streamed in batches, low memory). Output:
data/sample/datasets/{train,test}/ in the official TSV format, used by configs/sample_dev.yaml.

train: a random fraction of S1 + ALL their true S2/S3 matches + the same fraction of the
       unmatched (distractor) S2/S3 records -> a mini-universe with realistic density.
test:  the same fraction of S1, S2 and S3 at random (no labels exist).

Numbers measured on this sample are NOT full-data benchmarks (smaller pools -> easier
blocking); they are for checking code paths, runtime and memory only.

    python scripts/make_sample_data.py --fraction 0.02
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from ber.data.ingest import GT_COLUMNS, SOURCE_COLUMNS, open_tsv
from ber.utils.runtime import configure_arrow


def id_keys(ids) -> np.ndarray:
    """'S2-000123' -> int64 key encoding source digit, digit count (leading zeros matter)
    and value, so S2/S3 ids never collide."""
    src = pc.cast(pc.utf8_slice_codeunits(ids, 1, 2), pa.int64()).to_numpy(zero_copy_only=False)
    digits = pc.utf8_slice_codeunits(ids, 3)
    val = pc.cast(digits, pa.int64()).to_numpy(zero_copy_only=False)
    length = pc.utf8_length(digits).to_numpy(zero_copy_only=False).astype(np.int64)
    return val + length * 10**10 + src * 10**12


class TsvOut:
    def __init__(self, path: Path, header):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "wb")
        self.f.write(("\t".join(header) + "\n").encode("utf-8"))
        self.rows = 0

    def write(self, table: pa.Table):
        cols = [table.column(c).to_pylist() for c in table.column_names]
        lines = ["\t".join('"' + v.replace('"', '""') + '"' if '"' in v else v for v in row) for row in zip(*cols)]
        if lines:
            self.f.write(("\n".join(lines) + "\n").encode("utf-8"))
        self.rows += len(lines)

    def close(self):
        self.f.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="datasets")
    ap.add_argument("--out", default="data/sample/datasets")
    ap.add_argument("--fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    configure_arrow()
    raw, out = Path(a.raw), Path(a.out)
    rng = np.random.default_rng(a.seed)

    # ---- train S1 sample
    o = TsvOut(out / "train/train_source1.tsv", SOURCE_COLUMNS)
    sel_keys = []
    for b in open_tsv(raw / "train/train_source1.tsv", SOURCE_COLUMNS):
        t = pa.Table.from_batches([b]).filter(pa.array(rng.random(b.num_rows) < a.fraction))
        o.write(t)
        sel_keys.append(id_keys(t.column("entity_id")))
    o.close()
    sel_keys = np.sort(np.concatenate(sel_keys))
    print(f"train source1: {o.rows} rows")

    # ---- ground truth: rows of selected S1; collect wanted + all matched ids as int keys
    o = TsvOut(out / "train/train_ground_truth.tsv", GT_COLUMNS)
    wanted, matched = [], []
    for b in open_tsv(raw / "train/train_ground_truth.tsv", GT_COLUMNS):
        t = pa.Table.from_batches([b])
        flat = pc.list_flatten(pc.split_pattern(t.column("matched_entity_ids"), ","))
        flat = pc.filter(flat, pc.not_equal(flat, ""))
        matched.append(id_keys(flat))
        keep = np.isin(id_keys(t.column("source1_entity_id")), sel_keys)
        t = t.filter(pa.array(keep))
        o.write(t)
        f2 = pc.list_flatten(pc.split_pattern(t.column("matched_entity_ids"), ","))
        wanted.append(id_keys(pc.filter(f2, pc.not_equal(f2, ""))))
    o.close()
    wanted = np.unique(np.concatenate(wanted))
    matched = np.unique(np.concatenate(matched))
    print(f"train ground_truth: {o.rows} rows, {len(wanted)} wanted matches")

    for s in ("source2", "source3"):
        o = TsvOut(out / f"train/train_{s}.tsv", SOURCE_COLUMNS)
        for b in open_tsv(raw / f"train/train_{s}.tsv", SOURCE_COLUMNS):
            t = pa.Table.from_batches([b])
            k = id_keys(t.column("entity_id"))
            mask = (np.isin(k, wanted) | (~np.isin(k, matched) & (rng.random(t.num_rows) < a.fraction)))
            o.write(t.filter(pa.array(mask)))
        o.close()
        print(f"train {s}: {o.rows} rows")

    # ---- test: uniform sample of every file
    for s in ("source1", "source2", "source3"):
        o = TsvOut(out / f"test/test_{s}.tsv", SOURCE_COLUMNS)
        for b in open_tsv(raw / f"test/test_{s}.tsv", SOURCE_COLUMNS):
            t = pa.Table.from_batches([b])
            o.write(t.filter(pa.array(rng.random(t.num_rows) < a.fraction)))
        o.close()
        print(f"test {s}: {o.rows} rows")


if __name__ == "__main__":
    main()
