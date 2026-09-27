"""EXP-B-10: name_token.pair widened from the 2 to the 3 rarest pool-present name-view tokens (config only), on top of B-09.
A brute force  B exactly 3 rarest  C/E 2-token mode and flags unchanged  D dedup  F/K/L/M config + paths  G/J/N determinism
H empty names  I partitions (fit set guard)."""

import itertools

import numpy as np
import pyarrow as pa
import pytest

from ber.blocking.blocker import PASS_ORDER, CountryBlocker
from ber.config import Paths, load_config
from ber.data.split import load_val_ids  # noqa: F401  (partition guard exercised in test_I)
from ber.models import model_path

from test_blocking_v2 import cfg as bcfg

NT = 1 << PASS_ORDER.index("name_token")


def tbl(names):
    return pa.table({"name_norm": names, "name_core": names, "name_concat": [n.replace(" ", "") for n in names], "addr_norm": [""] * len(names),
                     "addr_nums": [""] * len(names), "addr_words": [""] * len(names)})


def cfg(k):
    c = bcfg()
    c["passes"]["name_token"] = {"enabled": True, "pair": True, "max_tokens": k, "name_max_df": 10**9, "max_df": 200, "max_block_size": 50, "weight": 1.0}
    return c


def run(s1, pool, c):
    b = CountryBlocker(s1, pool, c)
    out = {k: [] for k in ("s1_local", "pool_local", "cheap_score", "cheap_rank", "pass_flags")}
    for lo, hi in b.chunk_bounds():
        cc = b.candidates(lo, hi)
        for key in out:
            out[key].append(cc[key])
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}


def pairs(r):
    return set(zip(r["s1_local"].tolist(), r["pool_local"].tolist()))


def brute(s1, pool, k, cap=50):
    ptok = [set(n.split()) for n in pool.column("name_core").to_pylist()]
    df = {t: sum(t in s for s in ptok) for t in set().union(*ptok)}
    keys = lambda toks: {tuple(sorted(p)) for p in itertools.combinations(sorted([t for t in toks if df.get(t, 0) >= 1], key=lambda t: df[t])[:k], 2)}  # noqa: E731
    pk = [keys(t) for t in ptok]
    size = {}
    for ks in pk:
        for key in ks:
            size[key] = size.get(key, 0) + 1
    return {(i, j) for i, n in enumerate(s1.column("name_core").to_pylist()) for j, ks in enumerate(pk) if any(size[x] <= cap for x in keys(set(n.split())) & ks)}


def fixture():
    # DF (pool): alpha 1, beta 2, gamma 3, delta 4 -> no ties inside a record
    s1 = tbl(["alpha beta gamma delta", "gamma delta", ""])
    pool = tbl(["alpha", "beta gamma", "beta delta", "gamma delta", "gamma delta x", "delta y"])
    return s1, pool


def test_A_three_token_key_matches_brute_force():
    s1, pool = fixture()
    for k in (2, 3):
        assert pairs(run(s1, pool, cfg(k))) == brute(s1, pool, k), k


def test_B_exactly_three_rarest_tokens():
    s1, pool = fixture()
    r2, r3 = pairs(run(s1, pool, cfg(2))), pairs(run(s1, pool, cfg(3)))
    # S1 0: rarest 2 = alpha, beta; rarest 3 adds gamma -> (beta, gamma) links pool 1; delta (4th) never used for S1 0
    assert (0, 1) not in r2 and (0, 1) in r3
    assert (0, 2) not in r3  # (beta, delta): delta is the S1's 4th rarest token
    assert r2 <= r3 or all(p[0] == 1 for p in r2 - r3)


def test_C_E_two_token_mode_unchanged_and_flags():
    s1, pool = fixture()
    a, b = run(s1, pool, cfg(2)), run(s1, pool, cfg(2))
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert a["pass_flags"].dtype == np.uint8 and set(np.unique(a["pass_flags"]).tolist()) <= {NT}
    r3 = run(s1, pool, cfg(3))
    assert set(np.unique(r3["pass_flags"]).tolist()) <= {NT}  # same pass bit: widening, not a new family


def test_D_G_H_J_N_dedup_determinism_empty():
    s1, pool = fixture()
    r = run(s1, pool, cfg(3))
    codes = r["s1_local"] * 1000 + r["pool_local"]
    assert len(np.unique(codes)) == len(codes)
    again = run(s1, pool, cfg(3))
    for key in r:
        np.testing.assert_array_equal(r[key], again[key])
    assert 2 not in set(r["s1_local"].tolist())  # empty name: no candidates, no crash


def test_F_K_L_M_config():
    b09, b10 = load_config("configs/exp_b09_name_num.yaml"), load_config("configs/exp_b10_name_pair3.yaml")
    assert b10["blocking"]["passes"]["name_token"]["max_tokens"] == 3 and b09["blocking"]["passes"]["name_token"]["max_tokens"] == 2
    nt09 = {k: v for k, v in b09["blocking"]["passes"]["name_token"].items() if k != "max_tokens"}
    nt10 = {k: v for k, v in b10["blocking"]["passes"]["name_token"].items() if k != "max_tokens"}
    assert nt09 == nt10
    others09 = {k: v for k, v in b09["blocking"]["passes"].items() if k != "name_token"}
    others10 = {k: v for k, v in b10["blocking"]["passes"].items() if k != "name_token"}
    assert others09 == others10 and others10["name_num"]["enabled"]  # M: name_num exactly as in B-09 (one entry, unchanged)
    assert {k: v for k, v in b10["blocking"].items() if k != "passes"} == {k: v for k, v in b09["blocking"].items() if k != "passes"}  # top-K 20 etc.
    for sec in ("features", "decision", "fit", "xlit", "lightgbm", "split", "seed"):
        assert b10.get(sec) == b09.get(sec), sec  # L: D-01 / threshold procedure unchanged
    assert b10["matcher"]["features"] == b09["matcher"]["features"]
    p = model_path(b10, Paths(b10))
    assert p.parts[-2] == "EXP-B-10" and b10["scoring"]["tag"] == "scored_EXP-B-10" and "EXP-B-10" in b10["paths"]["output_dir"]  # F


def test_I_fit_set_guard_holds_for_b10():
    from ber.pipeline.run_baseline import fit_s1_subset
    c = load_config("configs/exp_b10_name_pair3.yaml")
    paths = Paths(c)
    if not paths.val_ids.exists():
        pytest.skip("frozen split not present")
    fit = set(fit_s1_subset(c, paths))
    assert len(fit) == 200000 and not fit & set(load_val_ids(paths))
