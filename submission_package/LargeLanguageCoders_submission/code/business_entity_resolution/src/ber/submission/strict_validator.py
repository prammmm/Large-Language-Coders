"""Strict local submission validator (DEC-011). Stricter than the official script:

Errors (exit 1):
  * BOM, non-UTF-8 bytes, CR characters (LF only), missing final newline
  * header not exactly ``source1_entity_id<TAB><list column>``
  * malformed row (!= 2 fields), spaces or quotes anywhere, empty list element
  * duplicate / missing / extra S1 rows vs. test_source1.tsv
  * ids without S2-/S3- prefix, S1 ids in lists, duplicate ids inside a list
  * ids not present in test_source2/3.tsv (always checked, memory-light via Arrow)
  * ids present in train_source2/3.tsv (with --train-dir)
  * matched pair not in candidate_pairs.tsv
  * candidate pairs != pairs actually scored by the pipeline (with --scored-dir)

IDs are read straight from the raw test TSVs (read-only), independently of the pipeline.
Usage (repo root):
  python -m ber.submission.strict_validator --matching output/matching_results.tsv \
      --candidate output/candidate_pairs.tsv --test-dir datasets/test \
      [--train-dir datasets/train] [--scored-dir data/processed/test/scored]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ber.data.ingest import SOURCE_COLUMNS, open_tsv
from ber.submission.writer import CANDIDATE_HEADER, MATCHING_HEADER
from ber.utils.runtime import configure_arrow

BATCH_IDS = 2_000_000
MAX_EXAMPLES = 5


def read_id_column(path: Path) -> pa.Array:
    reader = open_tsv(path, SOURCE_COLUMNS)
    chunks = [b.column("entity_id") for b in reader]
    return pa.concat_arrays(chunks) if chunks else pa.array([], pa.string())


def pair_hash(s1: list[str], ids: list[str]) -> np.ndarray:
    keys = np.array([f"{a}|{b}" for a, b in zip(s1, ids)], dtype=object)
    return pd.util.hash_array(keys, categorize=False).astype(np.uint64)


class _Issues:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, msg: str, examples=None):
        if examples:
            ex = sorted(examples)[:MAX_EXAMPLES]
            msg = f"{msg} ({len(examples)} total, e.g. {', '.join(map(str, ex))})"
        self.errors.append(msg)


def _check_file(path: Path, header: tuple[str, str], required: set[str], valid_ids: pa.Array, train_ids: pa.Array | None, issues: _Issues):
    """Stream one output file. Returns (sorted unique pair hashes, stats)."""
    name = path.name
    if not path.is_file():
        issues.err(f"{name}: file not found: {path}")
        return None, {}
    seen: set[str] = set()
    dup_rows, bad_rows, spaces, empty_elems, intra_dup, self_ids, bad_prefix = set(), [], set(), set(), set(), set(), set()
    unknown_count = 0
    unknown_ex: set[str] = set()
    train_count = 0
    train_ex: set[str] = set()
    hashes: list[np.ndarray] = []
    buf_s1: list[str] = []
    buf_id: list[str] = []
    n_rows = n_empty = n_ids = 0

    def flush():
        nonlocal unknown_count, train_count
        if not buf_id:
            return
        arr = pa.array(buf_id, pa.string())
        ok = pc.is_in(arr, value_set=valid_ids).to_numpy(zero_copy_only=False)
        if not ok.all():
            bad = np.flatnonzero(~ok)
            unknown_count += len(bad)
            for i in bad[:MAX_EXAMPLES]:
                unknown_ex.add(buf_id[i])
        if train_ids is not None:
            tr = pc.is_in(arr, value_set=train_ids).to_numpy(zero_copy_only=False)
            if tr.any():
                bad = np.flatnonzero(tr)
                train_count += len(bad)
                for i in bad[:MAX_EXAMPLES]:
                    train_ex.add(buf_id[i])
        hashes.append(pair_hash(buf_s1, buf_id))
        buf_s1.clear()
        buf_id.clear()

    with open(path, "rb") as f:
        first = f.readline()
        if first.startswith(b"\xef\xbb\xbf"):
            issues.err(f"{name}: starts with a UTF-8 BOM (must be UTF-8 without BOM)")
            first = first[3:]
        if b"\r" in first:
            issues.err(f"{name}: CRLF line endings in header (must be LF only)")
        expected = ("\t".join(header) + "\n").encode()
        if first.replace(b"\r", b"") != expected:
            issues.err(f"{name}: header {first!r} != expected {expected!r}")
            return None, {}
        cr_lines = 0
        line_no = 1
        last_line = first
        for bline in f:
            line_no += 1
            last_line = bline
            if b"\r" in bline:
                cr_lines += 1
                bline = bline.replace(b"\r", b"")
            try:
                line = bline.decode("utf-8")
            except UnicodeDecodeError:
                issues.err(f"{name}: invalid UTF-8 at line {line_no}")
                continue
            if line.endswith("\n"):
                line = line[:-1]
            if line == "":
                bad_rows.append(line_no)
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                bad_rows.append(line_no)
                continue
            s1, lst = parts
            if " " in line or '"' in line:
                spaces.add(s1)
            n_rows += 1
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)
            if lst == "":
                n_empty += 1
                continue
            ids = lst.split(",")
            if "" in ids:
                empty_elems.add(s1)
                ids = [x for x in ids if x]
            if len(set(ids)) != len(ids):
                intra_dup.add(s1)
            for x in ids:
                if x.startswith("S1-"):
                    self_ids.add(x)
                elif not x.startswith(("S2-", "S3-")):
                    bad_prefix.add(x)
            n_ids += len(ids)
            buf_s1.extend([s1] * len(ids))
            buf_id.extend(ids)
            if len(buf_id) >= BATCH_IDS:
                flush()
        flush()
        if cr_lines:
            issues.err(f"{name}: {cr_lines} line(s) contain CR (CRLF); LF only is required")
        if last_line and not last_line.endswith(b"\n"):
            issues.err(f"{name}: missing final newline")

    col = header[1]
    if bad_rows:
        issues.err(f"{name}: malformed rows (need exactly one TAB, no blank lines) at lines", [str(x) for x in bad_rows])
    if spaces:
        issues.err(f"{name}: spaces or quotes in rows", spaces)
    if dup_rows:
        issues.err(f"{name}: duplicate source1_entity_id rows", dup_rows)
    if empty_elems:
        issues.err(f"{name}: empty element in {col} (stray comma)", empty_elems)
    if intra_dup:
        issues.err(f"{name}: duplicate ids inside a {col} list", intra_dup)
    if self_ids:
        issues.err(f"{name}: Source-1 ids inside {col}", self_ids)
    if bad_prefix:
        issues.err(f"{name}: ids without S2-/S3- prefix", bad_prefix)
    if unknown_count:
        issues.err(f"{name}: {unknown_count} ids not in test_source2/3", unknown_ex)
    if train_count:
        issues.err(f"{name}: {train_count} ids found in TRAIN source2/3", train_ex)
    missing = required - seen
    if missing:
        issues.err(f"{name}: missing test S1 rows", missing)
    extra = seen - required
    if extra:
        issues.err(f"{name}: rows for S1 ids not in test_source1", extra)
    h = np.unique(np.concatenate(hashes)) if hashes else np.zeros(0, np.uint64)
    stats = {"rows": n_rows, "empty_rows": n_empty, "non_empty_rows": n_rows - n_empty, "ids": n_ids}
    print(f"  {name}: {n_rows} rows ({n_empty} empty), {n_ids} ids")
    return h, stats


def scored_pair_hashes(scored_dir: Path, s1_ids: pa.Array, pool_ids: pa.Array) -> np.ndarray:
    parts = sorted(Path(scored_dir).glob("*.parquet"))
    hashes = []
    for p in parts:
        t = pq.read_table(p, columns=["s1_row", "pool_row"])
        for off in range(0, t.num_rows, BATCH_IDS):
            sl = t.slice(off, BATCH_IDS)
            a = s1_ids.take(sl.column("s1_row").combine_chunks()).to_pylist()
            b = pool_ids.take(sl.column("pool_row").combine_chunks()).to_pylist()
            hashes.append(pair_hash(a, b))
    return np.unique(np.concatenate(hashes)) if hashes else np.zeros(0, np.uint64)


def validate(matching: Path, candidate: Path, test_dir: Path, train_dir: Path | None = None, scored_dir: Path | None = None):
    issues = _Issues()
    test_dir = Path(test_dir)
    s1_ids = read_id_column(test_dir / "test_source1.tsv")
    s2_ids = read_id_column(test_dir / "test_source2.tsv")
    s3_ids = read_id_column(test_dir / "test_source3.tsv")
    pool_ids = pa.concat_arrays([s2_ids, s3_ids])
    required = set(s1_ids.to_pylist())
    print(f"  required S1 entities: {len(required)}; valid S2/S3 ids: {len(pool_ids)}")
    train_ids = None
    if train_dir is not None:
        train_dir = Path(train_dir)
        train_ids = pa.concat_arrays([read_id_column(train_dir / "train_source2.tsv"), read_id_column(train_dir / "train_source3.tsv")])

    m_hash, m_stats = _check_file(Path(matching), MATCHING_HEADER, required, pool_ids, train_ids, issues)
    c_hash, c_stats = _check_file(Path(candidate), CANDIDATE_HEADER, required, pool_ids, train_ids, issues)
    if m_hash is not None and c_hash is not None:
        outside = int((~np.isin(m_hash, c_hash, assume_unique=True)).sum())
        if outside:
            issues.err(f"{outside} matched pair(s) are not in candidate_pairs.tsv (matches must be a subset of candidates)")
    if scored_dir is not None and c_hash is not None:
        s_hash = scored_pair_hashes(Path(scored_dir), s1_ids, pool_ids)
        if not np.array_equal(s_hash, c_hash):
            only_c = int((~np.isin(c_hash, s_hash, assume_unique=True)).sum())
            only_s = int((~np.isin(s_hash, c_hash, assume_unique=True)).sum())
            issues.err(f"candidate_pairs.tsv != scored pairs in {scored_dir}: {only_c} candidate-only, {only_s} scored-only pairs")
        else:
            print(f"  candidate set == scored set ({len(s_hash)} pairs)")
    elif scored_dir is None:
        issues.warnings.append("--scored-dir not given: candidate == scored-set check skipped")
    return issues.errors, issues.warnings, {"matching": m_stats, "candidate": c_stats}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Strict local validator (stricter than the official one).")
    ap.add_argument("--matching", default="output/matching_results.tsv")
    ap.add_argument("--candidate", default="output/candidate_pairs.tsv")
    ap.add_argument("--test-dir", default="datasets/test")
    ap.add_argument("--train-dir", default=None, help="also reject ids that exist in train source2/3")
    ap.add_argument("--scored-dir", default=None, help="pipeline scored-pairs dir; checks candidates == scored set")
    a = ap.parse_args(argv)
    configure_arrow()
    print("Strict local validator")
    errors, warnings, _ = validate(a.matching, a.candidate, a.test_dir, a.train_dir, a.scored_dir)
    for w in warnings:
        print(f"WARNING: {w}")
    if errors:
        print(f"FAIL: {len(errors)} issue(s):")
        for i, e in enumerate(errors, 1):
            print(f"  {i}. {e}")
        return 1
    print("PASS (strict)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
