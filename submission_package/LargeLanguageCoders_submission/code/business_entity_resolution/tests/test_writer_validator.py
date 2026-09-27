import numpy as np
import pyarrow as pa
import pytest

from ber.submission.strict_validator import validate
from ber.submission.writer import CANDIDATE_HEADER, MATCHING_HEADER, write_id_lists

COLS = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


@pytest.fixture
def test_dir(tmp_path):
    d = tmp_path / "test"
    d.mkdir()
    (d / "test_source1.tsv").write_bytes((COLS + "S1-1\ta\tb\tUS\nS1-2\ta\tb\tFrance\nS1-3\ta\tb\tIndia\n").encode())
    (d / "test_source2.tsv").write_bytes((COLS + "S2-1\ta\tb\tUS\nS2-2\ta\tb\tFrance\n").encode())
    (d / "test_source3.tsv").write_bytes((COLS + "S3-1\ta\tb\tIndia\n").encode())
    tr = tmp_path / "train"
    tr.mkdir()
    (tr / "train_source2.tsv").write_bytes((COLS + "S2-900\ta\tb\tUS\n").encode())
    (tr / "train_source3.tsv").write_bytes((COLS + "S3-900\ta\tb\tUS\n").encode())
    return d


def write_pair(tmp_path, cand_rows, match_rows):
    c = tmp_path / "candidate_pairs.tsv"
    m = tmp_path / "matching_results.tsv"
    c.write_bytes(("source1_entity_id\tcandidate_entity_ids\n" + "".join(r + "\n" for r in cand_rows)).encode())
    m.write_bytes(("source1_entity_id\tmatched_entity_ids\n" + "".join(r + "\n" for r in match_rows)).encode())
    return m, c


GOOD_C = ["S1-1\tS2-1,S3-1", "S1-2\tS2-2", "S1-3\t"]
GOOD_M = ["S1-1\tS2-1", "S1-2\t", "S1-3\t"]


def test_writer_output_format(tmp_path):
    p = tmp_path / "out" / "candidate_pairs.tsv"
    st = write_id_lists(p, CANDIDATE_HEADER, ["S1-1", "S1-2", "S1-3"], np.array([0, 0, 2]), ["S2-1", "S3-1", "S2-2"], block_rows=2)
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert raw == b"source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S3-1\nS1-2\t\nS1-3\tS2-2\n"
    assert st == {"path": str(p), "rows": 3, "empty_rows": 1, "ids": 3}


def test_writer_index_lookup_mode(tmp_path):
    p = tmp_path / "m.tsv"
    table = pa.array(["S2-1", "S2-2", "S3-1"])
    write_id_lists(p, MATCHING_HEADER, ["S1-1", "S1-2"], np.array([1, 1]), (np.array([2, 0]), table))
    assert p.read_bytes() == b"source1_entity_id\tmatched_entity_ids\nS1-1\t\nS1-2\tS3-1,S2-1\n"


def test_writer_rejects_bad_input(tmp_path):
    p = tmp_path / "x.tsv"
    with pytest.raises(ValueError):
        write_id_lists(p, MATCHING_HEADER, ["S1-1"], np.array([0, 0]), ["S2-1", "S2-1"])  # duplicate id
    with pytest.raises(ValueError):
        write_id_lists(p, MATCHING_HEADER, ["S1-1"], np.array([0]), ["S1-9"])  # S1 id in list
    with pytest.raises(ValueError):
        write_id_lists(p, MATCHING_HEADER, ["S1-1", "S1-2"], np.array([1, 0]), ["S2-1", "S2-2"])  # unsorted
    with pytest.raises(ValueError):
        write_id_lists(p, MATCHING_HEADER, ["S1-1", "S1-1"], np.array([], np.int64), [])  # duplicate S1


def test_validator_passes_good_files(tmp_path, test_dir):
    m, c = write_pair(tmp_path, GOOD_C, GOOD_M)
    errors, _, stats = validate(m, c, test_dir, train_dir=tmp_path / "train")
    assert errors == []
    assert stats["candidate"]["ids"] == 3


def test_writer_output_passes_validator(tmp_path, test_dir):
    s1 = ["S1-1", "S1-2", "S1-3"]
    c = write_id_lists(tmp_path / "c.tsv", CANDIDATE_HEADER, s1, np.array([0, 0, 1]), ["S2-1", "S3-1", "S2-2"])
    m = write_id_lists(tmp_path / "m.tsv", MATCHING_HEADER, s1, np.array([0]), ["S3-1"])
    errors, _, _ = validate(m["path"], c["path"], test_dir)
    assert errors == []


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda b: b"\xef\xbb\xbf" + b, "BOM"),
        (lambda b: b.replace(b"\n", b"\r\n"), "CR"),
        (lambda b: b.replace(b"source1_entity_id", b"source1_id", 1), "header"),
        (lambda b: b + b"S1-1\tS2-1\n", "duplicate source1_entity_id"),
        (lambda b: b.replace(b"S1-3\t\n", b""), "missing test S1"),
        (lambda b: b + b"S1-99\t\n", "not in test_source1"),
        (lambda b: b.replace(b"S1-1\tS2-1,S3-1", b"S1-1\tS2-1,S2-1"), "duplicate ids inside"),
        (lambda b: b.replace(b"S1-1\tS2-1,S3-1", b"S1-1\tS2-1,S1-2"), "Source-1 ids"),
        (lambda b: b.replace(b"S1-1\tS2-1,S3-1", b"S1-1\tS2-1,S2-404"), "not in test_source2/3"),
        (lambda b: b.replace(b"S1-1\tS2-1,S3-1", b"S1-1\tS2-1,S2-900"), "TRAIN"),
        (lambda b: b.replace(b"S1-1\tS2-1,S3-1", b"S1-1\tS2-1,"), "stray comma"),
        (lambda b: b.replace(b"S1-1\tS2-1,S3-1", b"S1-1\tS2-1, S3-1"), "spaces"),
        (lambda b: b.replace(b"S1-2\tS2-2", b"S1-2\tS2-2\textra"), "malformed"),
        (lambda b: b.replace(b"S1-2\tS2-2", b"S1-2\t\xff"), "UTF-8"),
        (lambda b: b.rstrip(b"\n"), "final newline"),
    ],
)
def test_validator_rejects(tmp_path, test_dir, mutate, needle):
    m, c = write_pair(tmp_path, GOOD_C, GOOD_M)
    c.write_bytes(mutate(c.read_bytes()))
    errors, _, _ = validate(m, c, test_dir, train_dir=tmp_path / "train")
    assert errors, "expected a failure"
    assert any(needle in e for e in errors), errors


def test_validator_matches_not_in_candidates(tmp_path, test_dir):
    m, c = write_pair(tmp_path, GOOD_C, ["S1-1\tS2-1", "S1-2\t", "S1-3\tS3-1"])
    errors, _, _ = validate(m, c, test_dir)
    assert any("not in candidate_pairs" in e for e in errors)


def test_validator_scored_set_mismatch(tmp_path, test_dir):
    import pyarrow.parquet as pq

    m, c = write_pair(tmp_path, GOOD_C, GOOD_M)
    sd = tmp_path / "scored"
    sd.mkdir()
    # scored set = (S1-1,S2-1),(S1-1,S3-1),(S1-2,S2-2) -> equal to GOOD_C
    pq.write_table(pa.table({"s1_row": pa.array([0, 0, 1], pa.int32()), "pool_row": pa.array([0, 2, 1], pa.int32())}), sd / "part-0.parquet")
    errors, _, _ = validate(m, c, test_dir, scored_dir=sd)
    assert errors == []
    pq.write_table(pa.table({"s1_row": pa.array([0], pa.int32()), "pool_row": pa.array([0], pa.int32())}), sd / "part-0.parquet")
    errors, _, _ = validate(m, c, test_dir, scored_dir=sd)
    assert any("scored" in e for e in errors)
