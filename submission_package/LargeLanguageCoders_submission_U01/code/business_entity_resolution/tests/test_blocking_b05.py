"""EXP-B-05: wider address-word window (3 -> 5) for the combined keys name_addr / addr_pair.

Unit tests on the window itself, backward compatibility of the 3-word setting, edge cases,
caps, config isolation, and a synthetic end-to-end run:
normalization -> B-05 blocking -> features -> LightGBM retrain -> scoring -> D-01 -> invariants."""

import csv
import itertools
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from ber.blocking.blocker import PASS_ORDER, cross_keys, rarest_per_record
from ber.config import Paths, load_config
from ber.data.normalize_records import load_manifest, read_country
from ber.data.split import load_val_ids
from ber.pipeline.run_baseline import _ids, fit_s1_subset, main
from ber.utils.runtime import read_json

from synthetic import make_synthetic_repo
from test_blocking_v2 import cfg, pairs, table

REPO = Path(__file__).resolve().parents[1]
AP = 1 << PASS_ORDER.index("addr_pair")
NA = 1 << PASS_ORDER.index("name_addr")


def _pool_with_df(target_name, target_addr, rare=("alpha", "bravo", "charlie"), common=("delta", "echo")):
    """Pool where the S1's words alpha/bravo/charlie are RARE (DF 1) and delta/echo COMMON (DF 3)."""
    names = [target_name] + [f"filler {i}" for i in range(7)]
    addrs = [target_addr] + [f"{w} zulu{i}" for i, w in enumerate(rare)] + [f"{common[0]} yankee{i}" for i in range(2)] + \
            [f"{common[1]} xray{i}" for i in range(2)]
    return table(names, addrs)


def test_rarest_per_record_window_sizes():
    rec = np.zeros(7, np.int32)
    tok = np.arange(7, dtype=np.int32)
    df = np.array([9, 1, 5, 2, 7, 3, 4])
    for k, want in ((3, {1, 3, 5}), (5, {1, 3, 5, 6, 2}), (10, set(range(7)))):
        _, t = rarest_per_record(rec, tok, df, k)
        assert set(t.tolist()) == want


def test_addr_pair_key_count_grows_3_to_10_pairs():
    rec = np.zeros(6, np.int32)
    tok = np.arange(6, dtype=np.int32)
    df = np.arange(1, 7)
    for k, n in ((3, 3), (5, 10)):
        r, kk = cross_keys(rec, tok, df, k, rec, tok, df, k, 6, 1000, same=True)
        assert len(kk) == n == len(set(kk.tolist())) == len(list(itertools.combinations(range(k), 2)))


def test_window5_recovers_pair_sharing_only_4th_and_5th_rarest_words():
    s1 = table(["Target Shop"], ["alpha bravo charlie delta echo"])
    pool = _pool_with_df("Unrelated Name", "delta echo")
    got3, _ = pairs(s1, pool, cfg(addr_pair=True, max_words=3))
    got5, _ = pairs(s1, pool, cfg(addr_pair=True, max_words=5))
    assert (0, 0) not in {(s, p) for s, p, _, _ in got3}  # shared words fall outside the rarest-3 window
    assert (0, 0) in {(s, p) for s, p, _, _ in got5}
    assert all(f & AP for s, p, f, _ in got5 if (s, p) == (0, 0))


def test_name_addr_window5_recovers_name_plus_4th_rarest_word():
    s1 = table(["Sunrise Dental"], ["alpha bravo charlie delta"])
    pool = _pool_with_df("Sunrise Clinic", "delta")
    got3, _ = pairs(s1, pool, cfg(name_addr=True, max_words=3))
    got5, _ = pairs(s1, pool, cfg(name_addr=True, max_words=5))
    assert (0, 0) not in {(s, p) for s, p, _, _ in got3} and (0, 0) in {(s, p) for s, p, _, _ in got5}


def test_window3_behaviour_unchanged_and_window5_is_a_superset():
    rng = np.random.default_rng(7)
    words = [f"w{i:02d}" for i in range(25)]
    names = ["Sun Moon", "Star Sun", "Moon Dental", "Sun Dental", "Star Clinic"]
    s1 = table([names[i % 5] for i in range(12)], [" ".join(rng.choice(words, 6, replace=False)) for _ in range(12)])
    pool = table([names[i % 5] for i in range(60)], [" ".join(rng.choice(words, 6, replace=False)) for _ in range(60)])
    # the default window of the test helper IS 3: explicit 3 must give the identical candidate list
    a, _ = pairs(s1, pool, cfg(name_addr=True, addr_pair=True))
    b, _ = pairs(s1, pool, cfg(name_addr=True, addr_pair=True, max_words=3))
    assert a == b
    c, _ = pairs(s1, pool, cfg(name_addr=True, addr_pair=True, max_words=5))
    # before top-K the key set only grows (top_k 20 here is never binding: <= 60 pool records, check it)
    assert {(s, p) for s, p, _, _ in a} <= {(s, p) for s, p, _, _ in c}


def test_empty_and_short_addresses_do_not_crash_or_match():
    s1 = table(["Sunrise Dental", "Moon Shop", "Star"], ["", "springfield", "near road"])
    pool = table(["Sunrise Dental", "Moon Shop", "Star"], ["", "springfield", "near road"])
    got, _ = pairs(s1, pool, cfg(name_addr=True, addr_pair=True, max_words=5))
    flags = {(s, p): f for s, p, f, _ in got}
    assert not any(f & AP for f in flags.values())  # < 2 useful words -> no address-pair key
    assert (1, 1) in flags and flags[(1, 1)] & NA  # name token + the single useful word still works
    assert (0, 0) not in flags and (2, 2) not in flags  # empty address / only stop-words


def test_generic_tokens_never_selected_even_with_window5():
    s1 = table(["A"], ["maple street road near springfield"])
    pool = table(["B", "C"], ["street road near", "maple springfield"])
    got, _ = pairs(s1, pool, cfg(addr_pair=True, max_words=5))
    assert {(s, p) for s, p, _, _ in got} == {(0, 1)}
    got, _ = pairs(s1, table(["B", "C"], ["maple springfield", "maple springfield"]), cfg(addr_pair=True, max_words=5, word_max_df=1))
    assert got == []  # words above word_max_df are not "useful" (token-quality rule unchanged)


def test_duplicate_keys_dedup_and_block_caps_with_window5():
    s1 = table(["Sunrise Dental"], ["maple springfield oak"])
    pool = table([f"Sunrise Shop {i}" for i in range(6)], ["maple springfield oak"] * 6)
    got, _ = pairs(s1, pool, cfg(name_addr=True, addr_pair=True, max_words=5))
    assert len({(s, p) for s, p, _, _ in got}) == len(got) == 6  # one row per pair across both passes
    assert all(f == (AP | NA) for _, _, f, _ in got)
    got, b = pairs(s1, pool, cfg(name_addr=True, addr_pair=True, block=5, max_words=5))
    assert got == [] and b.stats["passes"]["addr_pair"]["dropped_oversize_keys"] >= 1  # cap 5 < block of 6


def test_b05_config_changes_only_the_window():
    b04 = load_config(REPO / "configs" / "exp_b04_blocking_v2.yaml")
    b05 = load_config(REPO / "configs" / "exp_b05_addr_window5.yaml")
    assert b05["blocking"]["passes"]["addr_pair"]["max_words"] == 5 and b05["blocking"]["passes"]["name_addr"]["max_words"] == 5
    x = yaml.safe_load(yaml.safe_dump(b05["blocking"]))
    x["passes"]["addr_pair"]["max_words"] = x["passes"]["name_addr"]["max_words"] = 3
    assert x == b04["blocking"]  # nothing else in the blocker differs
    for k in ("features", "lightgbm", "split", "seed", "preprocess"):
        assert b05.get(k) == b04.get(k), k
    assert {k: v for k, v in b05["fit"].items()} == {k: v for k, v in b04["fit"].items()}
    assert b05["matcher"]["features"] == b04["matcher"]["features"]
    assert b05["scoring"]["tag"] != b04["scoring"]["tag"] and b05["matcher"].get("model_path") is None
    assert b05["paths"]["output_dir"] != "output"  # never the production output folder
    assert b05["decision"]["competition"] == {"enabled": True, "rule": "best_claimant", "margin": 0.0}


# ----------------------------------------------------------------------------- synthetic end-to-end
@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("b05_repo")
    info = make_synthetic_repo(root, n_train=220, n_test=90)
    c = yaml.safe_load(Path(info["config"]).read_text(encoding="utf-8"))
    c["experiment"]["id"] = "B05-SYN"
    c["matcher"]["type"] = "lightgbm"
    c["decision"] = {"threshold_from": "model", "threshold": 0.5,
                     "competition": {"enabled": True, "rule": "best_claimant", "margin": 0.0}}
    c["fit"].update({"max_train_s1": None, "inner_tune_fraction": 0.3})
    c["lightgbm"] = {"num_boost_round": 50, "early_stopping_rounds": 10, "params": {"num_leaves": 7, "min_data_in_leaf": 5, "num_threads": 1}}
    c["scoring"]["tag"] = "scored_b05"
    c["blocking"]["passes"]["name_addr"] = {"enabled": True, "max_name_tokens": 2, "max_words": 5, "word_max_df": 500000, "max_block_size": 50, "weight": 1.0}
    c["blocking"]["passes"]["addr_pair"] = {"enabled": True, "max_words": 5, "word_max_df": 500000, "max_block_size": 50, "weight": 1.0}
    p = root / "b05.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    for st in ("preprocess", "split", "block-eval", "fit", "train", "infer", "submit"):
        assert main([st, "--config", str(p)]) == 0, st
    info["cfg"] = p
    return info


def test_e2e_model_trained_on_b05_candidates_and_no_leakage(repo):
    cfg_ = load_config(repo["cfg"])
    paths = Paths(cfg_)
    fit_ids, val = set(fit_s1_subset(cfg_, paths)), set(load_val_ids(paths))
    assert fit_ids and val and not fit_ids & val
    model = read_json(paths.run_dir("B05-SYN") / "model.json")
    assert model["meta"]["blocking"] == cfg_["blocking"]  # blocker/model consistency
    assert model["meta"]["blocking"]["passes"]["addr_pair"]["max_words"] == 5
    t = pa.concat_tables([pq.read_table(f, columns=["s1_row", "pass_flags"]) for f in paths.scored_dir("train", "scored_b05_fit").glob("*.parquet")])
    rows_ids = set(_ids(paths, "train", "source1").take(pa.array(t.column("s1_row").to_numpy().astype(np.int64))).to_pylist())
    assert rows_ids <= fit_ids and not rows_ids & val  # training rows: train-fold only
    assert (t.column("pass_flags").to_numpy() & (AP | NA)).any()


def test_e2e_candidates_valid_country_isolated_and_topk(repo):
    cfg_ = load_config(repo["cfg"])
    paths = Paths(cfg_)
    m = load_manifest(paths, "test")
    s1c = np.empty(m["source1"]["rows"], dtype=object)
    for k in m["source1"]["countries"]:
        s1c[read_country(paths, "test", "source1", k, ["row"]).column("row").to_numpy()] = k
    n2 = m["source2"]["rows"]
    pc_ = np.empty(n2 + m["source3"]["rows"], dtype=object)
    for src, off in (("source2", 0), ("source3", n2)):
        for k in m[src]["countries"]:
            pc_[read_country(paths, "test", src, k, ["row"]).column("row").to_numpy() + off] = k
    t = pa.concat_tables([pq.read_table(p) for p in paths.scored_dir("test", "scored_b05").glob("*.parquet")])
    sr, pr = t.column("s1_row").to_numpy(), t.column("pool_row").to_numpy()
    assert (s1c[sr] == pc_[pr]).all()
    codes = sr.astype(np.int64) * len(pc_) + pr
    assert len(np.unique(codes)) == len(codes)
    assert np.bincount(sr).max() <= cfg_["blocking"]["top_k"] and (t.column("cheap_rank").to_numpy() < cfg_["blocking"]["top_k"]).all()


def test_e2e_d01_output_invariants(repo):
    paths = Paths(load_config(repo["cfg"]))
    s = read_json(paths.run_dir("B05-SYN") / "submission_summary.json")
    assert s["strict_validation_errors"] == [] and s["competition"]["after"]["conflicted_records"] == 0

    def read(p):
        with open(p, encoding="utf-8", newline="") as f:
            r = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        return {x[0]: [i for i in x[1].split(",") if i] for x in r[1:]}

    m, c = read(paths.output / "matching_results.tsv"), read(paths.output / "candidate_pairs.tsv")
    assert list(m) == list(c) and len(m) == load_manifest(paths, "test")["source1"]["rows"]
    seen = set()
    for s1, ids in m.items():
        assert set(ids) <= set(c[s1]) and not set(ids) & seen
        seen |= set(ids)
    raw = (paths.output / "matching_results.tsv").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw
