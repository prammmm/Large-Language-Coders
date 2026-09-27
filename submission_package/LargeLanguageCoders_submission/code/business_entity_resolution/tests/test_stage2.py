"""EXP-S-01: unit tests of the second-stage feature construction (ber.decision.stage2)."""

import numpy as np
import pytest

from ber.decision.stage2 import (CONTEXT_FEATURES, PEER_SIMS, VIEW_COLUMNS, context_features, feature_names, peer_feature_names,
                                 peer_features, peer_index, record_similarity)


def arr(*x, dt=np.int64):
    return np.array(x, dt)


def test_context_s1_level_rank_gap_second_and_ties():
    s1 = arr(0, 0, 0, 1)
    pool = arr(10, 11, 12, 10)
    sc = np.array([0.9, 0.95, 0.9, 0.3], np.float32)
    f = context_features(s1, pool, sc)
    assert f["s1_rank"].tolist() == [1, 0, 2, 0]  # tie 0.9/0.9 -> lower pool row first
    assert f["s1_n_cand"].tolist() == [3, 3, 3, 1]
    assert f["s1_best"].tolist() == pytest.approx([0.95, 0.95, 0.95, 0.3])
    assert f["s1_second"][:3].tolist() == pytest.approx([0.9] * 3) and np.isnan(f["s1_second"][3])
    assert f["s1_gap_best"].tolist() == pytest.approx([0.05, 0.0, 0.05, 0.0], abs=1e-6)
    assert f["s1_n_ge_08"].tolist() == [3, 3, 3, 0] and f["s1_rel_pos"].tolist() == pytest.approx([0.5, 0, 1, 0])


def test_context_record_level_claimants_rank_and_gap():
    # record 10 claimed by S1 0 (0.9) and S1 1 (0.3); record 11 only by S1 0
    s1, pool = arr(0, 1, 0), arr(10, 10, 11)
    sc = np.array([0.9, 0.3, 0.6], np.float32)
    f = context_features(s1, pool, sc)
    assert f["rec_n_claim"].tolist() == [2, 2, 1] and f["rec_rank"].tolist() == [0, 1, 0]
    assert f["rec_is_best"].tolist() == [1, 0, 1]
    assert f["rec_gap_best_other"][:2].tolist() == pytest.approx([0.6, -0.6], abs=1e-6)  # vs the best OTHER claimant
    assert np.isnan(f["rec_gap_best_other"][2]) and np.isnan(f["rec_second"][2])


def test_context_empty_and_permutation_invariance():
    f = context_features(arr(), arr(), np.zeros(0, np.float32))
    assert all(len(v) == 0 for v in f.values()) and set(f) == set(CONTEXT_FEATURES)
    rng = np.random.default_rng(0)
    s1, pool = rng.integers(0, 30, 300), rng.integers(0, 60, 300)
    code = np.unique(s1 * 100 + pool)
    s1, pool = code // 100, code % 100
    sc = rng.random(len(s1)).astype(np.float32)
    ref = context_features(s1, pool, sc)
    p = rng.permutation(len(s1))
    got = context_features(s1[p], pool[p], sc[p])
    for k in CONTEXT_FEATURES:
        np.testing.assert_array_equal(got[k], ref[k][p])


def test_peer_index_excludes_self_orders_by_score_and_pads():
    s1 = arr(0, 0, 0, 0, 1)
    pool = arr(1, 2, 3, 4, 9)
    sc = np.array([0.5, 0.9, 0.7, 0.8], np.float32)
    sc = np.append(sc, np.float32(0.4))
    p = peer_index(s1, pool, sc, 3)
    # order in S1 0: pool2 (0.9, idx1) > pool4 (0.8, idx3) > pool3 (0.7, idx2) > pool1 (0.5, idx0)
    assert [p[0][1], p[1][1], p[2][1]] == [3, 2, 0]  # the top pair's peers skip itself
    assert [p[0][2], p[1][2], p[2][2]] == [1, 3, 0]
    assert [p[0][0], p[1][0], p[2][0]] == [1, 3, 2]
    assert [p[0][4], p[1][4], p[2][4]] == [-1, -1, -1]  # single-candidate S1: no peers
    for j in range(3):
        assert all(p[j][i] != i for i in range(5))  # never its own peer
        assert all(p[j][i] == -1 or s1[p[j][i]] == s1[i] for i in range(5))  # peers come from the SAME S1 only


def _views_factory(table):
    def views(rows):
        return {c: [table[int(r)][i] for r in rows] for i, c in enumerate(VIEW_COLUMNS)}
    return views


def test_record_similarity_empty_duplicates_and_zero_strip():
    A = {"name_core": ["sharma medical", "", "a a b"], "name_xlit": ["sharma medical", "", "a a b"], "addr_norm": ["12 mg road pune", "", "x"],
         "addr_nums": ["012", "", ""], "addr_words": ["mg road pune", "", "near road"]}
    B = {"name_core": ["sharma medical", "x", "a b"], "name_xlit": ["sharma medical", "x", "a b"], "addr_norm": ["12 mg road pune", "y", "x"],
         "addr_nums": ["12", "5", ""], "addr_words": ["mg road pune", "z", "road near"]}
    r = record_similarity(A, B)
    assert r["name_jacc"].tolist() == [1.0, 0.0, 1.0]  # duplicates collapse (set semantics)
    assert r["num_jacc"][0] == 1.0  # "012" == "12" after leading-zero strip
    assert r["name_tokset"][1] == 0.0 and r["addr_tokset"][1] == 0.0  # empty side -> 0
    assert r["word_jacc"][2] == 0.0  # only stop-words -> no useful words


def test_peer_features_names_nan_padding_and_determinism():
    table = {1: ("sharma medical", "sharma medical", "12 mg road pune", "12", "mg pune"),
             2: ("sharma medical", "sharma medical", "12 mg road pune", "12", "mg pune"),
             3: ("other shop", "other shop", "5 lake road delhi", "5", "lake delhi"),
             9: ("solo", "solo", "", "", "")}
    s1, pool = arr(0, 0, 0, 1), arr(1, 2, 3, 9)
    sc = np.array([0.9, 0.8, 0.3, 0.5], np.float32)
    peers = peer_index(s1, pool, sc, 3)
    f = peer_features(pool, sc, peers, _views_factory(table))
    assert list(f) == peer_feature_names(3)
    assert f["p1_name_jacc"][1] == 1.0 and f["p1_score"][1] == pytest.approx(0.9)  # record 2 looks like the S1's best record 1
    assert f["p1_name_jacc"][2] == 0.0  # record 3 does not look like record 1
    assert np.isnan(f["p1_score"][3]) and f["peer_n"][3] == 0 and np.isnan(f["peer_max_sim"][3])  # no peers
    assert np.isnan(f["p3_name_jacc"][0])  # S1 0 has only 2 other candidates
    g = peer_features(pool, sc, peers, _views_factory(table))
    for k in f:
        np.testing.assert_array_equal(f[k], g[k])


def test_feature_name_layout():
    names = feature_names(3)
    assert len(names) == len(set(names)) == len(CONTEXT_FEATURES) + 3 * (len(PEER_SIMS) + 1) + 4


# ----------------------------------------------------------------------------- EXP-S-02: restricted additions / veto
from ber.decision.competition import best_claimant_keep, claim_stats  # noqa: E402
from ber.decision.stage2 import restrict_additions  # noqa: E402


def test_veto_is_exact_intersection_and_subset_of_both():
    rng = np.random.default_rng(42)
    control, stage2 = rng.random(5000) < 0.4, rng.random(5000) < 0.5
    score = rng.random(5000).astype(np.float32)
    v = restrict_additions(control, stage2, score, np.inf)
    np.testing.assert_array_equal(v, control & stage2)
    assert not (v & ~control).any() and not (v & ~stage2).any()  # never adds, never keeps a stage-2 rejection
    np.testing.assert_array_equal(restrict_additions(control, stage2, score, 0.0), stage2)  # floor 0: stage 2 unchanged
    mid = restrict_additions(control, stage2, score, 0.5)
    np.testing.assert_array_equal(mid, (control & stage2) | (stage2 & ~control & (score >= 0.5)))


def test_veto_edge_cases_empty_all_false_nan_and_alignment():
    e = np.zeros(0, bool)
    assert restrict_additions(e, e, np.zeros(0, np.float32), np.inf).shape == (0,)
    f = np.zeros(4, bool)
    assert not restrict_additions(f, f, np.ones(4, np.float32), np.inf).any()
    # a NaN stage-1 score can never pass a floor; a control pair stage 2 keeps is kept whatever its score
    out = restrict_additions(np.array([0, 1], bool), np.array([1, 1], bool), np.array([np.nan, np.nan], np.float32), 0.0)
    assert out.tolist() == [False, True]
    with pytest.raises(ValueError):
        restrict_additions(np.ones(3, bool), np.ones(2, bool), np.ones(3, np.float32), np.inf)


def test_veto_claimant_conflict_never_reassigns_a_record():
    # record 100 is claimed by S1 0 (stage-1 0.9) and S1 1 (0.8): the control's D-01 keeps S1 0; stage 2
    # prefers S1 1. The veto removes S1 0's pair and must NOT hand the record to S1 1 (no addition).
    s1, pool = np.array([0, 1, 1, 2]), np.array([100, 100, 101, 102])
    score = np.array([0.9, 0.8, 0.95, 0.7], np.float32)
    control = best_claimant_keep(pool, s1, score, score >= 0.67)
    assert control.tolist() == [True, False, True, True]
    prob = np.array([0.4, 0.9, 0.9, 0.3], np.float32)  # stage 2: S1 1 wins record 100; rejects S1 2's pair
    stage2 = best_claimant_keep(pool, s1, prob, prob >= 0.65)
    v = restrict_additions(control, stage2, score, np.inf)
    assert v.tolist() == [False, False, True, False]
    assert claim_stats(pool, s1, v)["conflicted_records"] == 0
