"""EXP-B-09: name-view token x house-number blocking key (`name_num` pass). Written before the implementation.

A brute force   B only the 2 rarest name tokens   C zero-stripped numbers, up to 2 (rarest)   D dedup + cheap score
E block cap     F empty fields   G existing passes unchanged   H/I config + separate paths   J no validation in fit
K determinism   L D-01 unchanged."""

import csv
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from ber.blocking.blocker import PASS_ORDER, CountryBlocker
from ber.config import Paths, load_config
from ber.data.split import load_val_ids
from ber.models import model_path
from ber.pipeline.run_baseline import fit_s1_subset, main
from ber.utils.runtime import read_json

from synthetic import make_synthetic_repo
from test_blocking_v2 import cfg as bcfg

NN = 1 << PASS_ORDER.index("name_num") if "name_num" in PASS_ORDER else None


def tbl(names, nums, words=None):
    n = len(names)
    words = words or [""] * n
    return pa.table({"name_norm": names, "name_core": names, "name_concat": [x.replace(" ", "") for x in names], "addr_norm": [f"{a} {b}".strip() for a, b in zip(nums, words)],
                     "addr_nums": nums, "addr_words": words})


def cfg(name_num=True, cap=50, max_name_tokens=2, max_nums=2, others=False):
    c = bcfg()
    if others:
        c["passes"]["exact_core"]["enabled"] = True
        c["passes"]["addr_num_word"] = {"enabled": True, "max_block_size": 50, "max_nums": 2, "max_words": 4, "word_max_df": 100, "weight": 1.0}
    c["strip_leading_zeros"] = True
    c["passes"]["name_num"] = {"enabled": name_num, "max_name_tokens": max_name_tokens, "max_nums": max_nums, "max_block_size": cap, "weight": 1.0}
    return c


def run(s1, pool, c):
    b = CountryBlocker(s1, pool, c)
    out = {k: [] for k in ("s1_local", "pool_local", "cheap_score", "cheap_rank", "pass_flags")}
    for lo, hi in b.chunk_bounds():
        cc = b.candidates(lo, hi)
        for k in out:
            out[k].append(cc[k])
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}, b


def pairs(res):
    return set(zip(res["s1_local"].tolist(), res["pool_local"].tolist()))


def brute(s1, pool, k_tok=2, k_num=2, cap=50):
    """Reference: per record the k rarest pool-present name tokens x k rarest pool-present numbers (zero-stripped);
    fixtures avoid DF ties inside a record so 'rarest' is unambiguous."""
    strip = lambda x: str(int(x)) if x.isdigit() else x  # noqa: E731
    ptoks = [set(n.split()) for n in pool.column("name_core").to_pylist()]
    pnums = [{strip(x) for x in s.split()} for s in pool.column("addr_nums").to_pylist()]
    tdf = {t: sum(t in s for s in ptoks) for t in set().union(*ptoks)}
    ndf = {t: sum(t in s for s in pnums) for t in set().union(*pnums)}

    def keys(toks, nums):
        tk = sorted([t for t in toks if tdf.get(t, 0) >= 1], key=lambda t: tdf[t])[:k_tok]
        nk = sorted([x for x in nums if ndf.get(x, 0) >= 1], key=lambda x: ndf[x])[:k_num]
        return {(t, x) for t in tk for x in nk}

    pkeys = [keys(t, n) for t, n in zip(ptoks, pnums)]
    size = {}
    for ks in pkeys:
        for key in ks:
            size[key] = size.get(key, 0) + 1
    out = set()
    for i, (n, s) in enumerate(zip(s1.column("name_core").to_pylist(), s1.column("addr_nums").to_pylist())):
        ks = keys(set(n.split()), {strip(x) for x in s.split()})
        for j, pk in enumerate(pkeys):
            if any(size[key] <= cap for key in ks & pk):
                out.add((i, j))
    return out


def fixture():
    s1 = tbl(["sharma medical store", "gupta traders", "patel", ""], ["12 7", "0045", "", "9"])
    pool = tbl(["sharma medical hall", "sharma general store", "medical sharma", "gupta sons", "kumar traders", "patel bros", "sharma medical"] + ["common"] * 3,
               ["12", "7", "012", "45", "45", "", "99", "1", "1", "1"])
    return s1, pool


def test_A_matches_brute_force_and_is_registered():
    assert "name_num" in PASS_ORDER
    s1, pool = fixture()
    res, _ = run(s1, pool, cfg())
    assert pairs(res) == brute(s1, pool)
    assert pairs(res)  # the fixture really links something
    assert ((res["pass_flags"] & NN) > 0).all()  # only the name_num pass is enabled


def test_B_only_the_two_rarest_name_tokens():
    # record tokens: rare1 (DF 1), rare2 (DF 1 in another record), shared "store" (DF 3): the shared token is 3rd rarest
    s1 = tbl(["alpha beta store"], ["5"])
    pool = tbl(["alpha", "beta", "store gamma", "store delta", "store epsilon"], ["8", "9", "5", "6", "7"])
    res, _ = run(s1, pool, cfg())
    assert (0, 2) not in pairs(res)  # "store" x 5 is not a key: store is not among the S1's 2 rarest tokens
    res3, _ = run(s1, pool, cfg(max_name_tokens=3))
    assert (0, 2) in pairs(res3)


def test_C_zero_stripped_numbers_up_to_two_rarest():
    s1 = tbl(["zeta"], ["007 12 99"])
    pool = tbl(["zeta"] * 2 + ["zeta", "x", "x", "x"], ["7", "0012", "99", "99", "99", "99"])
    res, _ = run(s1, pool, cfg())
    got = pairs(res)
    assert (0, 0) in got and (0, 1) in got  # "007" == "7", "12" == "0012"
    assert (0, 2) not in got  # 99 is the S1's most common number (DF 4): outside the 2 rarest
    assert got == brute(s1, pool)


def test_D_duplicates_removed_and_scores_summed_deterministically():
    s1 = tbl(["alpha beta"], ["5"])
    pool = tbl(["alpha beta"], ["5"])  # linked by (alpha,5) and (beta,5)
    res, b = run(s1, pool, cfg())
    assert len(res["s1_local"]) == 1
    w = np.log1p(b.n_pool / 1.0)
    assert res["cheap_score"][0] == pytest.approx(2 * w, rel=1e-6)
    again, _ = run(s1, pool, cfg())
    for k in res:
        np.testing.assert_array_equal(res[k], again[k])


@pytest.mark.parametrize("n,linked", [(50, True), (51, False)])
def test_E_block_cap(n, linked):
    s1 = tbl(["omega"], ["3"])
    pool = tbl(["omega"] * n, ["3"] * n)
    res, b = run(s1, pool, cfg(cap=50))
    assert bool(len(res["s1_local"])) == linked
    if not linked:
        assert b.stats["passes"]["name_num"]["dropped_oversize_keys"] >= 1


def test_F_empty_names_or_numbers():
    s1 = tbl(["", "alpha", ""], ["5", "", ""])
    pool = tbl(["alpha", "", "beta"], ["5", "5", ""])
    res, _ = run(s1, pool, cfg())
    assert len(res["s1_local"]) == 0
    res, _ = run(tbl([], []), pool, cfg())
    assert len(res["s1_local"]) == 0


def test_G_existing_passes_unchanged_and_flag_dtype():
    s1, pool = fixture()
    ref, _ = run(s1, pool, cfg(name_num=False, others=True))
    base = cfg(name_num=False, others=True)
    del base["passes"]["name_num"]  # configs written before B-09 do not mention the pass at all
    ref2, _ = run(s1, pool, base)
    for k in ref:
        np.testing.assert_array_equal(ref[k], ref2[k])
    assert ref["pass_flags"].dtype == np.uint8  # unchanged storage type without the new pass
    new, _ = run(s1, pool, cfg(name_num=True, others=True))
    assert new["pass_flags"].dtype == np.uint16
    old = {p: (s, f) for p, s, f in zip(zip(ref["s1_local"].tolist(), ref["pool_local"].tolist()), ref["cheap_score"].tolist(), ref["pass_flags"].tolist())}
    for p, s, f in zip(zip(new["s1_local"].tolist(), new["pool_local"].tolist()), new["cheap_score"].tolist(), new["pass_flags"].tolist()):
        if p in old:
            assert (f & ~NN) == old[p][1]  # existing bits unchanged
            assert s >= old[p][0] - 1e-6  # cheap score only gains the new key's weight
        else:
            assert f == NN  # pairs new in the list come from the new key only


def test_H_I_L_b09_config():
    b06, b09 = load_config("configs/exp_b06_name_pair.yaml"), load_config("configs/exp_b09_name_num.yaml")
    nn = b09["blocking"]["passes"]["name_num"]
    assert nn == {"enabled": True, "max_name_tokens": 2, "max_nums": 2, "max_block_size": 50, "weight": 1.0}
    assert "name_num" not in b06["blocking"]["passes"] or not b06["blocking"]["passes"]["name_num"].get("enabled")
    other = {k: v for k, v in b09["blocking"]["passes"].items() if k != "name_num"}
    assert other == b06["blocking"]["passes"]
    assert {k: v for k, v in b09["blocking"].items() if k != "passes"} == {k: v for k, v in b06["blocking"].items() if k != "passes"}  # top_k 20 etc.
    for sec in ("features", "decision", "fit", "xlit", "lightgbm", "split", "seed"):  # L: D-01 (decision.competition) unchanged
        assert b09.get(sec) == b06.get(sec), sec
    assert b09["matcher"]["features"] == b06["matcher"]["features"]
    p = model_path(b09, Paths(b09))
    assert p.parts[-2] == "EXP-B-09" and "EXP-B-06" not in str(p)
    assert b09["scoring"]["tag"] == "scored_EXP-B-09" and "EXP-B-09" in b09["paths"]["output_dir"]


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    root = tmp_path_factory.mktemp("b09_repo")
    info = make_synthetic_repo(root, n_train=260, n_test=90)
    c = yaml.safe_load(Path(info["config"]).read_text(encoding="utf-8"))
    c["experiment"]["id"] = "SYN-B09"
    c["matcher"].update({"type": "lightgbm", "model_path": None})
    c["decision"] = {"threshold_from": "model", "threshold": 0.5, "competition": {"enabled": True, "rule": "best_claimant", "margin": 0.0}}
    c["fit"].update({"max_train_s1": None, "inner_tune_fraction": 0.3})
    c["lightgbm"] = {"num_boost_round": 40, "early_stopping_rounds": 10, "params": {"num_leaves": 7, "min_data_in_leaf": 5, "num_threads": 1}}
    c["scoring"]["tag"] = "scored_b09"
    c["blocking"]["strip_leading_zeros"] = True
    c["blocking"]["passes"]["name_num"] = {"enabled": True, "max_name_tokens": 2, "max_nums": 2, "max_block_size": 50, "weight": 1.0}
    p = root / "b09.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    for st in ("preprocess", "split", "fit", "train", "infer", "submit"):
        assert main([st, "--config", str(p)]) == 0, st
    return p


def test_J_K_pipeline_no_validation_in_fit_new_pass_trained_and_deterministic(syn):
    c = load_config(syn)
    paths = Paths(c)
    assert not set(fit_s1_subset(c, paths)) & set(load_val_ids(paths))
    fit_parts = sorted(paths.scored_dir("train", "scored_b09_fit").glob("*.parquet"))
    t = pa.concat_tables([pq.read_table(f, columns=["s1_row", "pass_flags"]) for f in fit_parts])
    ids = pq.read_table(paths.interim_file("train", "source1"), columns=["entity_id"]).column("entity_id").to_pylist()
    assert not {ids[i] for i in np.unique(t.column("s1_row").to_numpy())} & set(load_val_ids(paths))
    assert t.column("pass_flags").type == pa.uint16() and (t.column("pass_flags").to_numpy() & NN).any()  # trained on name_num candidates
    m = read_json(model_path(c, paths))
    assert m["meta"]["blocking"]["passes"]["name_num"]["enabled"]
    out = paths.output
    s = read_json(paths.run_dir("SYN-B09") / "submission_summary.json")
    assert s["strict_validation_errors"] == []
    before = {f: (out / f).read_bytes() for f in ("candidate_pairs.tsv", "matching_results.tsv")}
    assert main(["submit", "--config", str(syn)]) == 0
    assert all((out / f).read_bytes() == b for f, b in before.items())
    with open(out / "matching_results.tsv", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    assert len(rows) - 1 == len({r[0] for r in rows[1:]})  # one row per S1
