"""EXP-B-11: name_token.pair weight 1.0 -> 2.0 on top of B-10's 3-token name-token-pair key (config only).
A 3-token key  B/C/D/P only the weight changes, exactly 2.0, other weights unchanged  E/F top-K and cap  G/H/O determinism, dedup
I/J cheap-score change is exactly the name-token contribution  K weight 1.0 reproduces B-10 exactly; B-09 config untouched
L fit-set guard  M D-01 / features / threshold procedure unchanged  N isolated paths."""

import numpy as np
import pytest

from ber.config import Paths, load_config
from ber.data.split import load_val_ids
from ber.models import model_path

from test_blocking_b10 import NT, brute, fixture, pairs, run
from test_blocking_b10 import cfg as nt_cfg

B09, B10, B11 = "configs/exp_b09_name_num.yaml", "configs/exp_b10_name_pair3.yaml", "configs/exp_b11_name_pair3_w2.yaml"
EN = 1 << 0  # exact_name bit


def cfg(k=3, w=1.0, exact_name=False):
    c = nt_cfg(k)
    c["passes"]["name_token"]["weight"] = w
    c["passes"]["exact_name"]["enabled"] = exact_name
    return c


def leaves(d, pre=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from leaves(v, pre + (k,))
        else:
            yield pre + (k,), v


def by_pair(r):
    return {(int(a), int(b)): (float(s), int(f)) for a, b, s, f in zip(r["s1_local"], r["pool_local"], r["cheap_score"], r["pass_flags"])}


def test_A_uses_b10_three_token_key():
    s1, pool = fixture()
    assert pairs(run(s1, pool, cfg(3, 2.0))) == brute(s1, pool, 3)
    b11 = load_config(B11)["blocking"]["passes"]["name_token"]
    assert b11["pair"] is True and b11["max_tokens"] == 3


def test_B_C_D_P_only_the_name_token_weight_changes():
    b10, b11 = dict(leaves(load_config(B10))), dict(leaves(load_config(B11)))
    diff = {k for k in set(b10) | set(b11) if b10.get(k) != b11.get(k)}
    # P: exactly one blocking leaf changes (plus the run identity / isolated paths, N)
    assert {k for k in diff if k[0] == "blocking"} == {("blocking", "passes", "name_token", "weight")}
    assert b10[("blocking", "passes", "name_token", "weight")] == 1.0 and b11[("blocking", "passes", "name_token", "weight")] == 2.0  # C
    assert {k[0] for k in diff} <= {"blocking", "experiment", "paths", "scoring", "_config_path"}, diff  # _config_path = loader metadata
    assert {k for k in diff if k[0] == "scoring"} <= {("scoring", "tag")}
    w10 = {p: v["weight"] for p, v in load_config(B10)["blocking"]["passes"].items() if p != "name_token"}
    w11 = {p: v["weight"] for p, v in load_config(B11)["blocking"]["passes"].items() if p != "name_token"}
    assert w10 == w11  # D


def test_E_F_topk_and_cap_unchanged():
    b09, b11 = load_config(B09)["blocking"], load_config(B11)["blocking"]
    assert b11["top_k"] == 20 == b09["top_k"]
    assert b11["passes"]["name_token"]["max_block_size"] == 50 == b09["passes"]["name_token"]["max_block_size"]
    assert {p: v.get("max_block_size") for p, v in b11["passes"].items()} == {p: v.get("max_block_size") for p, v in b09["passes"].items()}


def test_G_H_O_deterministic_and_deduplicated():
    s1, pool = fixture()
    a, b = run(s1, pool, cfg(3, 2.0, True)), run(s1, pool, cfg(3, 2.0, True))
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    codes = a["s1_local"] * 1000 + a["pool_local"]
    assert len(np.unique(codes)) == len(codes)
    # the candidate set never depends on the weight (only ranks / scores do) when nothing is cut by top-K
    assert pairs(run(s1, pool, cfg(3, 1.0, True))) == pairs(a)


def test_I_J_score_change_is_exactly_the_name_token_contribution():
    s1, pool = fixture()
    base = by_pair(run(s1, pool, cfg(3, 1.0)))  # name-token pass alone at weight 1.0 (exact_name hits identical names)
    only_en = cfg(3, 1.0, True)
    only_en["passes"]["name_token"]["enabled"] = False
    en = by_pair(run(s1, pool, only_en))
    w1, w2 = by_pair(run(s1, pool, cfg(3, 1.0, True))), by_pair(run(s1, pool, cfg(3, 2.0, True)))
    assert set(w1) == set(w2) and any(f & EN for _, f in w1.values()) and any(f & NT for _, f in w1.values())
    for p in w1:
        nt_part = w1[p][0] - en.get(p, (0.0, 0))[0]
        if w1[p][1] & NT:
            assert nt_part > 0 and w2[p][0] == pytest.approx(w1[p][0] + nt_part, rel=1e-6)  # I: doubled name-token term
            assert nt_part == pytest.approx(base[p][0], rel=1e-6)
        else:
            assert w2[p][0] == w1[p][0]  # J: non-name-token contributions unchanged
        assert w1[p][1] == w2[p][1]  # flags unchanged


def test_K_weight_one_reproduces_b10_and_b09_untouched():
    s1, pool = fixture()
    a, b10 = run(s1, pool, cfg(3, 1.0)), run(s1, pool, nt_cfg(3))  # weight 1.0 = the B-10 blocker, array for array
    for key in a:
        np.testing.assert_array_equal(a[key], b10[key])
    b09 = load_config(B09)["blocking"]["passes"]["name_token"]
    assert b09["weight"] == 1.0 and b09["max_tokens"] == 2  # B-09 behaviour is defined by its own config, unchanged


def test_L_fit_set_guard_holds_for_b11():
    from ber.pipeline.run_baseline import fit_s1_subset
    c = load_config(B11)
    paths = Paths(c)
    if not paths.val_ids.exists():
        pytest.skip("frozen split not present")
    fit = set(fit_s1_subset(c, paths))
    assert len(fit) == 200000 and not fit & set(load_val_ids(paths))
    assert fit == set(fit_s1_subset(load_config(B09), Paths(load_config(B09))))  # same fit set as the control


def test_M_N_decision_features_and_isolated_paths():
    b09, b11 = load_config(B09), load_config(B11)
    for sec in ("features", "decision", "fit", "xlit", "lightgbm", "split", "seed"):
        assert b11.get(sec) == b09.get(sec), sec
    assert b11["matcher"]["features"] == b09["matcher"]["features"]
    assert {k: v for k, v in b11["matcher"].items() if k != "model_path"} == {k: v for k, v in b09["matcher"].items() if k != "model_path"}
    p = model_path(b11, Paths(b11))
    assert p.parts[-2] == "EXP-B-11" and b11["scoring"]["tag"] == "scored_EXP-B-11"
    assert b11["paths"]["output_dir"] == "experiments/runs/EXP-B-11/output"
