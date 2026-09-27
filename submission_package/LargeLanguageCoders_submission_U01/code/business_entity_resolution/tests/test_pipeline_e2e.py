"""Small synthetic end-to-end run of every pipeline stage + the official validator."""

import subprocess
import sys
from pathlib import Path

import pytest

from ber.config import Paths, load_config
from ber.data.ingest import count_data_lines
from ber.pipeline.run_baseline import main
from ber.utils.runtime import read_json

from synthetic import make_synthetic_repo

REPO = Path(__file__).resolve().parents[1]
OFFICIAL = REPO / "student_resource" / "utils" / "validate_submission.py"


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_repo")
    info = make_synthetic_repo(root)
    raw_before = {p: p.read_bytes() for p in (root / "datasets").rglob("*.tsv")}
    cfg = str(info["config"])
    for stage in ("preprocess", "split", "train", "infer", "submit"):
        assert main([stage, "--config", cfg]) == 0
    info["root"] = root
    info["raw_before"] = raw_before
    return info


def test_raw_data_untouched(repo):
    for p, b in repo["raw_before"].items():
        assert p.read_bytes() == b


def test_ingest_row_counts(repo):
    import pyarrow.parquet as pq

    paths = Paths(load_config(repo["config"]))
    for split in ("train", "test"):
        for s in ("source1", "source2", "source3"):
            assert pq.ParquetFile(paths.interim_file(split, s)).metadata.num_rows == count_data_lines(paths.raw_file(split, s))


def test_quoted_field_parsed(repo):
    import pyarrow.parquet as pq

    paths = Paths(load_config(repo["config"]))
    addrs = pq.read_table(paths.interim_file("train", "source2")).column("business_address").to_pylist()
    assert any(a.endswith(', "C" Wing') for a in addrs)


def test_open_set_countries(repo):
    paths = Paths(load_config(repo["config"]))
    m = read_json(paths.processed_dir("test") / "manifest.json")
    assert set(m["source1"]["countries"]) == {"us", "india", "france"}
    m_tr = read_json(paths.processed_dir("train") / "manifest.json")
    assert "france" not in m_tr["source1"]["countries"]


def test_split_is_deterministic(repo):
    paths = Paths(load_config(repo["config"]))
    first = paths.val_ids.read_bytes()
    with pytest.raises(FileExistsError):
        main(["split", "--config", str(repo["config"])])
    assert main(["split", "--config", str(repo["config"]), "--force"]) == 0
    assert paths.val_ids.read_bytes() == first


def test_train_metrics(repo):
    paths = Paths(load_config(repo["config"]))
    r = read_json(paths.run_dir("EXP-B-01") / "metrics.json")
    best = r["at_threshold_used"]
    assert r["evaluated_s1"] > 0
    assert 0 <= best["f05"] <= 1 and 0 <= best["blocking_recall"] <= 1
    assert best["f05"] >= r["all_empty_f05"]  # tuned threshold is never worse than predicting nothing
    assert best["blocking_recall"] > 0.8  # synthetic variants share address/name evidence
    assert best["f05"] > 0.6
    assert (paths.run_dir("EXP-B-01") / "tuned_threshold.json").exists()


def test_outputs_cover_all_test_s1_and_france(repo):
    paths = Paths(load_config(repo["config"]))
    for name, header in (("candidate_pairs.tsv", "candidate_entity_ids"), ("matching_results.tsv", "matched_entity_ids")):
        raw = (paths.output / name).read_bytes()
        assert raw.startswith(f"source1_entity_id\t{header}\n".encode())
        assert b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
        rows = raw.decode().splitlines()[1:]
        assert [r.split("\t")[0] for r in rows] == repo["test_s1"]  # every S1, test-file order
    summary = read_json(paths.run_dir("EXP-B-01") / "submission_summary.json")
    assert summary["strict_validation_errors"] == []


def test_official_validator_passes(repo):
    paths = Paths(load_config(repo["config"]))
    res = subprocess.run(
        [sys.executable, str(OFFICIAL), "-m", str(paths.output / "matching_results.tsv"), "-c", str(paths.output / "candidate_pairs.tsv"),
         "-t", str(paths.raw / "test"), "--check-ids"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "PASS" in res.stdout
