import numpy as np
import pytest

from ber.evaluation.metrics import entity_f05, evaluate, f05_from_counts, threshold_sweep


def test_official_example():
    # PS: predict [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812] -> 0.714
    f = entity_f05(["S2-00047", "S2-00193", "S3-00812"], ["S2-00047", "S3-00812"])
    assert f == pytest.approx(0.7142857, abs=1e-6)


def test_closed_form_equals_pr_form():
    for tp, k, n in [(1, 1, 4), (3, 4, 4), (2, 5, 3), (4, 4, 4)]:
        p, r = tp / k, tp / n
        assert f05_from_counts(tp, k, n) == pytest.approx(1.25 * p * r / (0.25 * p + r))


def test_singleton_rules():
    assert entity_f05([], []) == 1.0
    assert entity_f05(["S2-1"], []) == 0.0


def test_empty_prediction_on_positive_is_zero():
    assert entity_f05([], ["S2-1"]) == 0.0
    assert entity_f05(["S2-9"], ["S2-1"]) == 0.0


def test_evaluate_macro_and_breakdowns():
    truth = {"S1-a": ["S2-1", "S3-1"], "S1-b": [], "S1-c": ["S2-2"], "S1-d": []}
    pred = {"S1-a": ["S2-1", "S2-9", "S3-1"], "S1-b": [], "S1-c": [], "S1-d": ["S2-5"]}
    cands = {"S1-a": ["S2-1", "S2-9", "S3-1"], "S1-b": ["S2-7"], "S1-c": ["S2-2", "S2-3"], "S1-d": ["S2-5"]}
    country = {"S1-a": "us", "S1-b": "us", "S1-c": "india", "S1-d": "france"}
    r = evaluate(pred, truth, cands, country)
    expected = (0.7142857 + 1.0 + 0.0 + 0.0) / 4
    assert r["f05"] == pytest.approx(expected, abs=1e-6)
    assert r["singleton_correct_empty"] == 1 and r["singleton_false_positive"] == 1
    assert r["empty_on_positive"] == 1
    assert r["blocking_recall"] == pytest.approx(3 / 3)
    assert r["oracle_f05"] == pytest.approx(1.0)
    assert r["candidates_per_s1"]["mean"] == pytest.approx(7 / 4)
    assert r["candidates_per_s1"]["max"] == 3
    assert r["matches_not_in_candidates"] == 0
    assert set(r["by_country"]) == {"us", "india", "france"}
    assert r["by_country"]["us"]["f05"] == pytest.approx((0.7142857 + 1.0) / 2, abs=1e-6)
    assert set(r["by_match_count"]) == {"0", "1", "2-3"}
    # macro precision over entities with predictions: a=2/3, d=0
    assert r["macro_precision"] == pytest.approx((2 / 3 + 0) / 2)
    # macro recall over positives: a=1, c=0
    assert r["macro_recall"] == pytest.approx(0.5)


def test_blocking_recall_counts_misses():
    truth = {"S1-a": ["S2-1", "S2-2"]}
    r = evaluate({"S1-a": ["S2-1"]}, truth, {"S1-a": ["S2-1"]})
    assert r["blocking_recall"] == pytest.approx(0.5)
    assert r["oracle_f05"] == pytest.approx(1.25 / (0.5 + 1))


def test_all_empty_scores_singleton_rate():
    truth = {f"S1-{i}": ([] if i % 4 == 0 else ["S2-x"]) for i in range(100)}
    assert evaluate({}, truth)["f05"] == pytest.approx(0.25)


def test_threshold_sweep_matches_dict_evaluation():
    s1 = np.array([0, 0, 0, 1, 2])
    true = np.array([True, False, True, False, True])
    score = np.array([0.9, 0.6, 0.4, 0.8, 0.7], np.float32)
    n_true = np.array([2, 0, 2])
    rows = threshold_sweep(s1, true, score, n_true, [0.5, 0.85])
    # t=0.5 -> e0: preds{0.9 T,0.6 F} tp1 k2 n2; e1: pred 0.8 on singleton ->0; e2: tp1 k1 n2
    exp = (float(f05_from_counts(1, 2, 2)) + 0 + float(f05_from_counts(1, 1, 2))) / 3
    assert rows[0]["f05"] == pytest.approx(exp)
    # t=0.85 -> e0 tp1 k1; e1 empty singleton -> 1; e2 empty -> 0
    assert rows[1]["f05"] == pytest.approx((float(f05_from_counts(1, 1, 2)) + 1 + 0) / 3)
