"""EXP-D-04: removal-only per-S1 expected-F0.5 pruning.
A exact optimum vs brute force (all truth assignments, all subsets of the accepted set that are prefixes)
B removal-only (keep ⊆ acc), S1 without accepted pairs untouched  C k = 0 rule (singleton)  D worked examples
E records taken by another S1 count as p = 0  F determinism / order invariance  G config: one change, S-03 read-only."""

import itertools

import numpy as np
import pytest

from ber.config import load_config
from ber.decision.ef_prune import expected_f_prune


def f05(tp, k, t):
    if t == 0:
        return 1.0 if k == 0 else 0.0
    if k == 0 or tp == 0:
        return 0.0
    P, R = tp / k, tp / t
    return 1.25 * P * R / (0.25 * P + R)


def brute_expected(pa, pr, k):
    ps = list(pa) + list(pr)
    e = 0.0
    for truth in itertools.product((0, 1), repeat=len(ps)):
        w = np.prod([p if t else 1 - p for p, t in zip(ps, truth)])
        e += w * f05(sum(truth[:k]), k, sum(truth))
    return e


def brute_best_k(pa, pr):
    es = [brute_expected(pa, pr, k) for k in range(len(pa) + 1)]
    best = max(es)
    return max(k for k, e in enumerate(es) if e >= best - 1e-12)


def one_s1(pa, pr):
    n = len(pa) + len(pr)
    s1 = np.zeros(n, np.int64)
    pool = np.arange(n, dtype=np.int64)
    prob = np.array(list(pa) + list(pr), np.float32)
    acc = np.array([True] * len(pa) + [False] * len(pr))
    return s1, pool, prob, acc


def test_A_matches_brute_force():
    rng = np.random.default_rng(42)
    for _ in range(150):
        m, r = int(rng.integers(1, 5)), int(rng.integers(0, 5))
        pa = np.sort(rng.uniform(0.55, 1.0, m).astype(np.float32))[::-1]
        pr = rng.uniform(0.0, 0.7, r).astype(np.float32)
        keep = expected_f_prune(*one_s1(pa, pr))
        assert int(keep.sum()) == brute_best_k(pa.astype(float), pr.astype(float)), (pa, pr)
        assert keep[: len(pa)].tolist() == sorted(keep[: len(pa)].tolist(), reverse=True)  # a prefix by probability


def test_B_removal_only_and_untouched_s1():
    rng = np.random.default_rng(7)
    n = 400
    s1 = np.sort(rng.integers(0, 60, n)).astype(np.int64)
    pool = rng.permutation(n).astype(np.int64)
    prob = rng.uniform(0, 1, n).astype(np.float32)
    acc = prob > 0.6
    acc[s1 == 3] = False
    keep = expected_f_prune(s1, pool, prob, acc)
    assert not (keep & ~acc).any()
    assert (keep[s1 == 3] == acc[s1 == 3]).all()


def test_C_k0_singleton_rule():
    # one accepted pair p = 0.4 with nothing else: keep only if p >= 0.5 (E[keep] = p, E[empty] = 1 - p)
    assert expected_f_prune(*one_s1([0.4], []))[0] == False  # noqa: E712
    assert expected_f_prune(*one_s1([0.6], []))[0] == True  # noqa: E712
    assert expected_f_prune(*one_s1([0.5], []))[0] == True  # noqa: E712  (tie keeps)


def test_D_worked_examples():
    # 1 confident match + a borderline pair: break-even ~0.727
    assert expected_f_prune(*one_s1([1.0, 0.70], []))[1] == False  # noqa: E712
    assert expected_f_prune(*one_s1([1.0, 0.75], []))[1] == True  # noqa: E712
    # 3 confident matches + a borderline pair: break-even ~0.771
    assert expected_f_prune(*one_s1([1.0, 1.0, 1.0, 0.75], []))[3] == False  # noqa: E712
    assert expected_f_prune(*one_s1([1.0, 1.0, 1.0, 0.80], []))[3] == True  # noqa: E712


def test_E_taken_records_count_zero():
    # S1 0: accepted 1.0 and 0.72; a rejected 0.6 candidate whose record is accepted by S1 1 must not count
    s1 = np.array([0, 0, 0, 1], np.int64)
    pool = np.array([10, 11, 12, 12], np.int64)
    prob = np.array([1.0, 0.72, 0.6, 0.99], np.float32)
    acc = np.array([True, True, False, True])
    got = expected_f_prune(s1, pool, prob, acc)
    ref = expected_f_prune(s1[[0, 1, 3]], pool[[0, 1, 3]], prob[[0, 1, 3]], acc[[0, 1, 3]])
    assert got[[0, 1, 3]].tolist() == ref.tolist()


def test_F_deterministic_and_order_invariant():
    rng = np.random.default_rng(3)
    n = 3000
    s1 = rng.integers(0, 500, n).astype(np.int64)
    pool = rng.permutation(n).astype(np.int64)
    prob = rng.uniform(0, 1, n).astype(np.float32)
    acc = prob > 0.65
    a = expected_f_prune(s1, pool, prob, acc)
    assert np.array_equal(a, expected_f_prune(s1, pool, prob, acc))
    perm = rng.permutation(n)
    b = expected_f_prune(s1[perm], pool[perm], prob[perm], acc[perm])
    assert np.array_equal(a[perm], b)


def test_G_config_single_change_s03_read_only():
    s03, d04 = load_config("configs/exp_s03_veto_on_b09.yaml"), load_config("configs/exp_d04_ef_prune.yaml")
    s2a, s2b = dict(s03["stage2"]), dict(d04["stage2"])
    assert s2b.pop("ef_prune") is True and "ef_prune" not in s2a
    assert s2b.pop("model_from") == "EXP-S-03" and s2b.pop("features_from") == "EXP-S-03"
    assert s2a == s2b
    for sec in ("blocking", "matcher", "scoring", "features", "decision", "fit", "xlit", "seed"):
        assert d04.get(sec) == s03.get(sec), sec
    assert d04["paths"]["output_dir"] == "experiments/runs/EXP-D-04/output" and d04["experiment"]["id"] == "EXP-D-04"
    from ber.config import Paths
    from ber.pipeline.run_stage2 import model_dir, run_fit
    assert model_dir(d04, Paths(d04)).name == "EXP-S-03"
    with pytest.raises(RuntimeError):
        run_fit(d04, Paths(d04))  # the S-03 model is never refitted / overwritten
