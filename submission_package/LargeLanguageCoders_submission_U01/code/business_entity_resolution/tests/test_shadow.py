"""POST-S-01 forensics: shadow benchmark definitions, weighting, leakage checks, and the stage-2
feature builder's competition-drop mask (diagnostics only)."""

import numpy as np
import pytest
from ber.evaluation.metrics import evaluate_arrays
from ber.forensics.shadow import (cand_bin, chain_orphan_mask, leakage_checks, match_weights, orphan_fraction, orphan_mask, shadow_split, u01,
                                  weighted_metrics)

IDS = [f"S1-{i}" for i in range(20000)]


def test_u01_is_deterministic_uniform_and_salted():
    a, b = u01(IDS, "shadow"), u01(IDS, "shadow")
    np.testing.assert_array_equal(a, b)
    assert 0.47 < (a < 0.5).mean() < 0.53 and a.min() >= 0 and a.max() < 1
    assert not np.array_equal(a, u01(IDS, "orphan")) and not np.array_equal(a, u01(IDS, "shadow", seed=7))


def test_shadow_split_is_inside_E_half_B_and_disjoint():
    rng = np.random.default_rng(0)
    eb = rng.random(len(IDS)) < 0.5
    sp = shadow_split(eb, IDS)
    assert not (sp["shadow_base"] & ~eb).any() and not (sp["gate_B"] & ~eb).any()
    assert not (sp["shadow_base"] & sp["gate_B"]).any() and ((sp["shadow_base"] | sp["gate_B"]) == eb).all()


def test_orphan_fraction_and_mask():
    f = orphan_fraction(train_pool=10_320_219, train_s1=2_206_821, test_pool=10_019_000, test_s1=1_732_544)
    assert 0.18 < f < 0.20
    m = orphan_mask(IDS, 0.2)
    assert 0.18 < m.mean() < 0.22
    np.testing.assert_array_equal(m, orphan_mask(IDS, 0.2))
    assert orphan_fraction(100, 10, 50, 10) == 0.0  # a sparser test never removes S1


def test_match_weights_reproduce_target_mix():
    cells = np.array([0] * 80 + [1] * 20)
    w = match_weights(cells, {0: 0.5, 1: 0.5, 2: 0.3})  # cell 2 absent in the source: renormalised away
    assert abs(w.mean() - 1) < 1e-9
    assert abs(w[cells == 0].sum() / w.sum() - 0.5) < 1e-9 and abs(w[cells == 1].sum() / w.sum() - 0.5) < 1e-9
    np.testing.assert_array_equal(cand_bin(np.array([1, 3, 4, 7, 8, 11, 50])), [0, 0, 1, 2, 3, 4, 4])


def test_weighted_metrics_equal_evaluate_arrays_when_unweighted():
    rng = np.random.default_rng(1)
    n = rng.integers(0, 6, 500)
    k = rng.integers(0, 6, 500)
    tp = np.minimum(np.minimum(k, n), rng.integers(0, 6, 500))
    ref = evaluate_arrays(tp, k, n)
    got = weighted_metrics(tp, k, n)
    assert got["f05"] == pytest.approx(ref["f05"]) and got["precision"] == pytest.approx(ref["macro_precision"])
    assert got["recall"] == pytest.approx(ref["macro_recall"])
    w = np.ones(500) * 3.0
    assert weighted_metrics(tp, k, n, w)["f05"] == pytest.approx(ref["f05"])


def test_leakage_checks_report_overlaps():
    a = np.array([1, 1, 0, 0], bool)
    r = leakage_checks(a, fit=np.array([0, 0, 1, 1], bool), stage2_train=np.array([0, 1, 0, 0], bool))
    assert r == {"shadow_in_fit": 0, "shadow_in_stage2_train": 1}


# ----------------------------------------------------------------------------- diagnostics helpers
from ber.forensics.diagnostics import binned, cell_ids, expected_precision, flip_codes, flip_counts, psi, summary  # noqa: E402


def test_flip_codes_and_counts():
    c = np.array([0, 0, 1, 1], bool)
    n = np.array([0, 1, 0, 1], bool)
    assert flip_codes(c, n).tolist() == [0, 1, 2, 3]
    assert flip_counts(flip_codes(c, n)) == {"unchanged_0": 1, "added_0_to_1": 1, "removed_1_to_0": 1, "unchanged_1": 1}


def test_psi_zero_on_same_distribution_and_large_on_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20000)
    assert psi(a, a) < 1e-6
    assert psi(a, rng.normal(size=20000)) < 0.02
    assert psi(a, rng.normal(1.0, 1, 20000)) > 0.25
    b = a.copy()
    b[:5000] = np.nan  # missingness is a shift too
    assert psi(a, b) > 0.1


def test_summary_binned_cells():
    s = summary(np.array([1.0, 2.0, np.nan, 4.0]))
    assert s["missing_rate"] == 0.25 and s["median"] == 2.0 and s["n"] == 4
    assert binned(np.array([0.1, 0.5, 0.9, np.nan]), [0.3, 0.8]).tolist() == [0, 1, 2, 3]
    c = cell_ids(np.array([0, 1, 1]), np.array([2, 2, 3]))
    assert len(set(c.tolist())) == 3


def test_expected_precision_uses_train_cells_and_reports_coverage():
    train_cells = np.array([0] * 100 + [1] * 100)
    train_true = np.array([1] * 90 + [0] * 10 + [1] * 50 + [0] * 50)
    r = expected_precision(train_cells, train_true, np.array([0, 0, 1, 7]))
    assert r["covered"] == 3 and abs(r["expected_precision_on_covered"] - (0.9 + 0.9 + 0.5) / 3) < 1e-9
    assert abs(r["coverage"] - 0.75) < 1e-9



def test_shadow_split_excludes_validation():
    rng = np.random.default_rng(3)
    eb = rng.random(len(IDS)) < 0.5
    val = rng.random(len(IDS)) < 0.1
    sp = shadow_split(eb, IDS, exclude=val)
    assert not (sp["shadow_base"] & val).any() and not (sp["shadow_base"] & ~eb).any()
    assert ((sp["shadow_base"] | sp["gate_B"]) == eb).all()


def test_chain_orphan_mask_only_removes_unprotected_chain_members():
    names = np.array(["a", "a", "a", "b", "c", "c", "d"] * 3000, dtype=object)
    ids = [f"S1-{i}" for i in range(len(names))]
    protect = np.zeros(len(names), bool)
    protect[::5] = True
    m = chain_orphan_mask(ids, names, protect, frac=0.5)
    assert not (m & protect).any()
    single = np.isin(names, ["b", "d"]) & (np.arange(len(names)) < 7)  # first block: b and d occur once there, but repeat later
    uniq_names = np.array([f"u{i}" for i in range(50)], dtype=object)
    assert not chain_orphan_mask([f"X{i}" for i in range(50)], uniq_names, np.zeros(50, bool), frac=1.0).any()  # unique names never removed
    assert 0.3 < m[~protect].mean() < 0.55
    np.testing.assert_array_equal(m, chain_orphan_mask(ids, names, protect, frac=0.5))


def test_floor_decision_limits_only_additions():
    from ber.forensics.diagnostics import floor_decision
    control = np.array([1, 1, 0, 0, 0], bool)
    stage2 = np.array([1, 0, 1, 1, 0], bool)
    score = np.array([0.9, 0.8, 0.6, 0.2, 0.1], np.float32)
    assert floor_decision(control, stage2, score, 0.0).tolist() == stage2.tolist()
    assert floor_decision(control, stage2, score, 0.5).tolist() == [True, False, True, False, False]  # removal kept, weak addition dropped
    assert floor_decision(control, stage2, score, 2.0).tolist() == [True, False, False, False, False]  # removals only


# ----------------------------------------------------------------------------- v2: locked sets + density tiers
from ber.forensics.shadow import assert_disjoint, locked_split, lookalike_pairs, tier_orphan_mask  # noqa: E402


def test_locked_split_disjoint_cover_and_deterministic():
    rng = np.random.default_rng(5)
    shadow = rng.random(len(IDS)) < 0.4
    sp = locked_split(shadow, IDS)
    assert_disjoint(**sp)
    assert ((sp["dev"] | sp["locked_A"] | sp["locked_B"]) == shadow).all()
    assert 0.45 < sp["dev"].sum() / shadow.sum() < 0.55 and 0.2 < sp["locked_A"].sum() / shadow.sum() < 0.3
    again = locked_split(shadow, IDS)
    for k in sp:
        np.testing.assert_array_equal(sp[k], again[k])
    with pytest.raises(AssertionError):
        assert_disjoint(a=np.array([1, 0], bool), b=np.array([1, 1], bool))


def test_tier_orphan_mask_protects_is_nested_and_chain_first():
    rng = np.random.default_rng(6)
    chain = rng.random(len(IDS)) < 0.5
    protect = rng.random(len(IDS)) < 0.2
    prev = np.zeros(len(IDS), bool)
    for q in (0.0, 0.3, 0.7, 1.0, 1.4, 2.0):
        m = tier_orphan_mask(IDS, chain, protect, q)
        assert not (m & protect).any() and (m >= prev).all()  # never an evaluated S1; nested in q
        if q <= 1.0:
            assert not (m & ~chain).any()  # chain members first
        prev = m
    assert not tier_orphan_mask(IDS, chain, protect, 0.0).any()
    np.testing.assert_array_equal(tier_orphan_mask(IDS, chain, protect, 2.0), ~protect)


def test_lookalike_pairs_follow_claimants_after_drop():
    s1 = np.array([0, 1, 2, 2])
    pool = np.array([10, 10, 11, 12])
    score = np.array([0.3, 0.9, 0.2, 0.4], np.float32)
    peer = np.array([0.9, 0.9, np.nan, 0.85], np.float32)
    assert lookalike_pairs(s1, pool, score, peer).tolist() == [False, False, False, True]  # record 10 contested; NaN peer never
    drop = np.array([False, True, False])
    assert lookalike_pairs(s1, pool, score, peer, drop).tolist() == [True, False, False, True]  # S1 1 removed -> 10 uncontested
