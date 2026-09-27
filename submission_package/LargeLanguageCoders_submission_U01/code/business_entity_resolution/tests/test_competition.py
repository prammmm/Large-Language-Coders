"""EXP-D-01: best-claimant (full-competition) decision rule — unit tests + synthetic end-to-end."""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from ber.config import Paths, load_config
from ber.decision.competition import best_claimant_keep, claim_stats, claimants_per_pair, winner_score_per_pair
from ber.pipeline.run_baseline import main
from ber.utils.runtime import read_json

from synthetic import make_synthetic_repo

REPO = Path(__file__).resolve().parents[1]
OFFICIAL = REPO / "student_resource" / "utils" / "validate_submission.py"


def _run(pool, s1, score, pred=None, margin=0.0):
    pool, s1 = np.array(pool), np.array(s1)
    score = np.array(score, np.float32)
    pred = np.ones(len(pool), bool) if pred is None else np.array(pred, bool)
    return best_claimant_keep(pool, s1, score, pred, margin)


def test_single_claimant_kept():
    assert _run([7], [0], [0.9]).tolist() == [True]
    # different records, one claimant each -> nothing removed
    assert _run([1, 2, 3], [0, 0, 1], [0.7, 0.8, 0.9]).all()


def test_two_claimants_winner_kept_loser_removed():
    assert _run([5, 5], [0, 1], [0.70, 0.95]).tolist() == [False, True]


def test_three_and_more_claimants():
    keep = _run([5, 5, 5], [2, 0, 1], [0.8, 0.9, 0.85])
    assert keep.tolist() == [False, True, False]
    keep = _run([5] * 6 + [6, 6], [0, 1, 2, 3, 4, 5, 0, 1], [0.7, 0.71, 0.99, 0.8, 0.9, 0.98, 0.9, 0.8])
    assert keep.tolist() == [False, False, True, False, False, False, True, False]


def test_equal_score_tie_goes_to_lowest_s1_row_and_is_order_independent():
    a = _run([5, 5, 5], [9, 3, 4], [0.9, 0.9, 0.9])
    assert a.tolist() == [False, True, False]
    rng = np.random.default_rng(0)
    for _ in range(20):  # permuting the input never changes the kept set
        p = rng.permutation(3)
        b = _run(np.array([5, 5, 5])[p], np.array([9, 3, 4])[p], np.array([0.9, 0.9, 0.9])[p])
        assert np.array_equal(b, a[p])


def test_only_predicted_pairs_compete_and_rule_only_removes():
    # the high-scoring pair for record 5 is NOT predicted (e.g. below threshold) -> it does not compete
    keep = _run([5, 5, 5], [0, 1, 2], [0.99, 0.8, 0.7], pred=[False, True, True])
    assert keep.tolist() == [False, True, False]
    rng = np.random.default_rng(1)
    pool, s1 = rng.integers(0, 50, 2000), rng.integers(0, 300, 2000)
    score = rng.random(2000).astype(np.float32)
    pred = score >= 0.5
    keep = best_claimant_keep(pool, s1, score, pred)
    assert not (keep & ~pred).any()  # never adds a prediction
    assert claim_stats(pool, s1, keep)["conflicted_records"] == 0  # exactly one claimant per record
    assert set(pool[keep]) == set(pool[pred])  # every claimed record keeps exactly its winner


def test_empty_and_all_unpredicted():
    e = np.zeros(0, np.int64)
    assert best_claimant_keep(e, e, np.zeros(0, np.float32), np.zeros(0, bool)).shape == (0,)
    assert not _run([1, 1], [0, 1], [0.1, 0.2], pred=[False, False]).any()
    assert claim_stats(e, e, np.zeros(0, bool))["conflicted_records"] == 0


def test_margin_sensitivity_mode():
    keep = _run([5, 5, 5], [0, 1, 2], [0.95, 0.93, 0.70], margin=0.05)
    assert keep.tolist() == [True, True, False]  # 0.93 is within 0.05 of the winner -> kept


def test_stats_helpers():
    pool, s1 = np.array([5, 5, 5, 6, 6, 7]), np.array([0, 1, 2, 0, 3, 4])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.95, 0.9], np.float32)
    pred = np.ones(6, bool)
    st = claim_stats(pool, s1, pred)
    assert st == {"predicted_pairs": 6, "claimed_records": 3, "conflicted_records": 2, "pairs_on_conflicted_records": 5,
                  "losers_lower_bound": 3, "conflict_size": {"2": 1, "3": 1, "4": 0, "5+": 0}}
    assert claimants_per_pair(pool, pred).tolist() == [3, 3, 3, 2, 2, 1]
    assert np.allclose(winner_score_per_pair(pool, s1, score, pred), [0.9, 0.9, 0.9, 0.95, 0.95, 0.9])


# ----------------------------------------------------------------------------- synthetic end-to-end
def _read_tsv(p):
    with open(p, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    return rows[0], {r[0]: [x for x in r[1].split(",") if x] for r in rows[1:]}


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("comp_repo")
    info = make_synthetic_repo(root, n_train=200, n_test=90)
    cfg = yaml.safe_load(Path(info["config"]).read_text(encoding="utf-8"))
    cfg["experiment"]["id"] = "EXP-TEST-CTRL"
    # a low threshold makes several S1 claim the same record, so the rule has work to do
    cfg["decision"] = {"threshold_from": "config", "threshold": 0.05}
    ctrl = root / "ctrl.yaml"
    ctrl.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    for st in ("preprocess", "split", "train", "infer", "submit"):
        assert main([st, "--config", str(ctrl)]) == 0, st
    exp = {"inherit": str(ctrl), "experiment": {"id": "EXP-TEST-D01", "allowed_stages": ["submit"]},
           "paths": {"output_dir": "out_d01"},
           "decision": {"competition": {"enabled": True, "rule": "best_claimant", "margin": 0.0}}}
    d01 = root / "d01.yaml"
    d01.write_text(yaml.safe_dump(exp, sort_keys=False), encoding="utf-8")
    assert main(["submit", "--config", str(d01)]) == 0
    info.update({"ctrl": ctrl, "d01": d01})
    return info


def test_e2e_invariants(repo):
    pc, pd = Paths(load_config(repo["ctrl"])), Paths(load_config(repo["d01"]))
    assert pc.output != pd.output
    assert (pc.output / "candidate_pairs.tsv").read_bytes() == (pd.output / "candidate_pairs.tsv").read_bytes()
    hc, mc = _read_tsv(pc.output / "matching_results.tsv")
    hd, md = _read_tsv(pd.output / "matching_results.tsv")
    _, cand = _read_tsv(pd.output / "candidate_pairs.tsv")
    assert hc == hd and list(mc) == list(md)  # same S1 rows, same order (complete coverage)
    seen = set()
    for s, ids in md.items():
        assert set(ids) <= set(mc[s]) and set(ids) <= set(cand[s])  # only removals; matches ⊆ candidates
        assert [x for x in mc[s] if x in set(ids)] == ids  # kept ids keep their order
        assert not (set(ids) & seen)  # no record assigned to two S1
        seen |= set(ids)
    s = read_json(pd.run_dir("EXP-TEST-D01") / "submission_summary.json")
    comp = s["competition"]
    assert s["strict_validation_errors"] == [] and comp["after"]["conflicted_records"] == 0
    assert comp["before"]["conflicted_records"] > 0, "synthetic data produced no conflicts; test would be vacuous"
    removed = sum(len(v) for v in mc.values()) - sum(len(v) for v in md.values())
    assert removed == comp["before"]["predicted_pairs"] - comp["after"]["predicted_pairs"] == comp["before"]["losers_lower_bound"]
    # singletons / empty predictions stay valid rows
    assert any(len(v) == 0 for v in md.values())


def test_e2e_deterministic_rerun(repo):
    pd = Paths(load_config(repo["d01"]))
    first = (pd.output / "matching_results.tsv").read_bytes()
    assert main(["submit", "--config", str(repo["d01"])]) == 0
    assert (pd.output / "matching_results.tsv").read_bytes() == first


def test_stage_guard(repo):
    with pytest.raises(SystemExit):
        main(["infer", "--config", str(repo["d01"])])


def test_official_validator_passes(repo):
    pd = Paths(load_config(repo["d01"]))
    res = subprocess.run(
        [sys.executable, str(OFFICIAL), "-m", str(pd.output / "matching_results.tsv"), "-c", str(pd.output / "candidate_pairs.tsv"),
         "-t", str(pd.raw / "test"), "--check-ids"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert res.returncode == 0 and "PASS" in res.stdout, res.stdout + res.stderr
