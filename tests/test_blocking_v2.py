"""EXP-B-04 Blocking V2: name_addr and addr_pair passes, caps, dedup, country isolation, and a
synthetic end-to-end run where LightGBM is RETRAINED on candidates from the same blocker."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from ber.blocking.blocker import PASS_ORDER, CountryBlocker, cross_keys, rarest_per_record, cross_within_records
from ber.config import Paths, load_config
from ber.data.normalize_records import load_manifest, read_country
from ber.data.split import load_val_ids
from ber.normalization.views import build_views
from ber.pipeline.run_baseline import fit_s1_subset, main
from ber.utils.runtime import read_json

from synthetic import make_synthetic_repo


def table(names, addrs):
    v = build_views(names, addrs)
    return pa.table({"row": pa.array(range(len(names)), pa.int32()), **{k: pa.array(x) for k, x in v.items()}})


def cfg(name_addr=False, addr_pair=False, block=50, word_max_df=1000, max_words=3, max_name_tokens=2):
    off = {"enabled": False, "max_block_size": 50, "weight": 1.0}
    passes = {k: dict(off) for k in ("exact_name", "exact_core", "sorted_core")}
    passes["addr_num_word"] = {**off, "max_nums": 2, "max_words": 4, "word_max_df": 100}
    passes["addr_rare_word"] = {**off, "max_df": 1}
    passes["name_token"] = {**off, "max_df": 5}
    passes["name_addr"] = {"enabled": name_addr, "max_block_size": block, "weight": 1.0, "max_name_tokens": max_name_tokens,
                           "max_words": max_words, "word_max_df": word_max_df}
    passes["addr_pair"] = {"enabled": addr_pair, "max_block_size": block, "weight": 1.0, "max_words": max_words, "word_max_df": word_max_df}
    return {"top_k": 20, "max_raw_pairs_per_chunk": 1000, "passes": passes}


def pairs(s1, pool, c):
    b = CountryBlocker(s1, pool, c)
    out = []
    for lo, hi in b.chunk_bounds():
        cc = b.candidates(lo, hi)
        out += list(zip(cc["s1_local"].tolist(), cc["pool_local"].tolist(), cc["pass_flags"].tolist(), cc["cheap_score"].tolist()))
    return out, b


def test_cross_keys_matches_reference_and_pairs_are_unordered():
    rng = np.random.default_rng(3)
    rec = np.sort(rng.integers(0, 30, 150)).astype(np.int32)
    tok = rng.integers(0, 40, 150).astype(np.int32)
    code = np.unique(rec.astype(np.int64) * 40 + tok)
    rec, tok = (code // 40).astype(np.int32), (code % 40).astype(np.int32)
    df = rng.integers(1, 9, 40)
    r, t = rarest_per_record(rec, tok, df, 3)
    ref_r, ref_k = cross_within_records(r, t, r, t, 40)
    keep = (ref_k // 40) < (ref_k % 40)
    ref = sorted(zip(ref_r[keep].tolist(), ref_k[keep].tolist()))
    for window in (1, 7, 1000):
        rr, kk = cross_keys(rec, tok, df, 3, rec, tok, df, 3, 40, window, same=True)
        assert sorted(zip(rr.tolist(), kk.tolist())) == ref
        assert ((kk // 40) < (kk % 40)).all()  # unordered, no self-pairs


def test_name_addr_key_links_common_name_word_plus_address_word():
    s1 = table(["Sunrise Dental Clinic"], ["Maple Avenue Springfield"])
    pool = table(["SUNRISE Medical", "Sunrise Pharma", "Moonrise Dental"], ["Springfield, Maple Ave", "Other Town", "Maple Springfield"])
    got, _ = pairs(s1, pool, cfg(name_addr=True))
    assert {(s, p) for s, p, _, _ in got} == {(0, 0), (0, 2)}  # needs a shared name token AND address word
    bit = 1 << PASS_ORDER.index("name_addr")
    assert all(f & bit for _, _, f, _ in got)


def test_addr_pair_key_requires_two_shared_useful_words_and_skips_stopwords():
    s1 = table(["A"], ["Maple Street Springfield"])
    pool = table(["B", "C", "D"], ["maple springfield", "maple street road", "springfield road near"])
    got, _ = pairs(s1, pool, cfg(addr_pair=True))
    # "street"/"road"/"near" are address stop-words: only record 0 shares TWO useful words
    assert {(s, p) for s, p, _, _ in got} == {(0, 0)}


def test_frequency_and_block_caps():
    s1 = table(["Sunrise Dental"], ["Maple Springfield"])
    pool = table([f"Sunrise Shop {i}" for i in range(6)], ["Maple Springfield"] * 6)
    got, b = pairs(s1, pool, cfg(name_addr=True, addr_pair=True, block=5))  # blocks of 6 > cap 5
    assert got == [] and b.stats["passes"]["name_addr"]["dropped_oversize_keys"] >= 1
    got, _ = pairs(s1, pool, cfg(name_addr=True, block=50, word_max_df=5))  # words with pool DF 6 > 5 excluded
    assert got == []
    got, _ = pairs(s1, pool, cfg(name_addr=True, addr_pair=True, block=50))
    assert len({(s, p) for s, p, _, _ in got}) == len(got) == 6  # dedup across both passes
    both = (1 << PASS_ORDER.index("name_addr")) | (1 << PASS_ORDER.index("addr_pair"))
    assert all(f == both for _, _, f, _ in got)


def test_cheap_score_reflects_new_keys():
    s1 = table(["Sunrise Dental"], ["Maple Springfield"])
    pool = table(["Sunrise Shop"], ["Maple Springfield"])
    a, _ = pairs(s1, pool, cfg(addr_pair=True))
    b, _ = pairs(s1, pool, cfg(name_addr=True, addr_pair=True))
    assert b[0][3] > a[0][3]  # an extra shared key raises cheap_score -> the model must be retrained


def test_pass_flags_fit_in_uint16():
    # EXP-B-09 added a 9th pass: flags are stored as uint16 when it is enabled (uint8 otherwise, see test_blocking_b09 G)
    assert len(PASS_ORDER) <= 16


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("b04_repo")
    info = make_synthetic_repo(root, n_train=220, n_test=90)
    c = yaml.safe_load(Path(info["config"]).read_text(encoding="utf-8"))
    c["experiment"]["id"] = "B04-SYN"
    c["matcher"]["type"] = "lightgbm"
    c["decision"] = {"threshold_from": "model", "threshold": 0.5}
    c["fit"].update({"max_train_s1": None, "inner_tune_fraction": 0.3})
    c["lightgbm"] = {"num_boost_round": 50, "early_stopping_rounds": 10, "params": {"num_leaves": 7, "min_data_in_leaf": 5, "num_threads": 1}}
    c["scoring"]["tag"] = "scored_b04"
    c["blocking"]["passes"]["name_addr"] = {"enabled": True, "max_name_tokens": 2, "max_words": 3, "word_max_df": 500000, "max_block_size": 50, "weight": 1.0}
    c["blocking"]["passes"]["addr_pair"] = {"enabled": True, "max_words": 3, "word_max_df": 500000, "max_block_size": 50, "weight": 1.0}
    p = root / "b04.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    for st in ("preprocess", "split", "block-eval", "fit", "train", "infer", "submit"):
        assert main([st, "--config", str(p)]) == 0, st
    info["cfg"] = p
    return info


def test_retrained_model_uses_same_blocker_and_no_leakage(repo):
    cfg_ = load_config(repo["cfg"])
    paths = Paths(cfg_)
    assert not set(fit_s1_subset(cfg_, paths)) & set(load_val_ids(paths))
    model = read_json(paths.run_dir("B04-SYN") / "model.json")
    assert model["type"] == "lightgbm"
    assert model["meta"]["blocking"]["passes"]["name_addr"]["enabled"] and model["meta"]["blocking"]["passes"]["addr_pair"]["enabled"]
    assert model["meta"]["blocking"] == cfg_["blocking"]  # model trained on candidates of THIS blocker
    fit_flags = pq.read_table(next(paths.scored_dir("train", "scored_b04_fit").glob("*.parquet")), columns=["pass_flags"]).column("pass_flags").to_numpy()
    new_bits = (1 << PASS_ORDER.index("name_addr")) | (1 << PASS_ORDER.index("addr_pair"))
    assert (fit_flags & new_bits).any()  # training candidates contain the new keys


def test_outputs_valid_and_country_isolated(repo):
    paths = Paths(load_config(repo["cfg"]))
    assert read_json(paths.run_dir("B04-SYN") / "submission_summary.json")["strict_validation_errors"] == []
    m = load_manifest(paths, "test")
    s1c = np.empty(m["source1"]["rows"], dtype=object)
    for k in m["source1"]["countries"]:
        s1c[read_country(paths, "test", "source1", k, ["row"]).column("row").to_numpy()] = k
    n2 = m["source2"]["rows"]
    pc_ = np.empty(n2 + m["source3"]["rows"], dtype=object)
    for src, off in (("source2", 0), ("source3", n2)):
        for k in m[src]["countries"]:
            pc_[read_country(paths, "test", src, k, ["row"]).column("row").to_numpy() + off] = k
    t = pa.concat_tables([pq.read_table(p) for p in paths.scored_dir("test", "scored_b04").glob("*.parquet")])
    sr, pr = t.column("s1_row").to_numpy(), t.column("pool_row").to_numpy()
    assert (s1c[sr] == pc_[pr]).all()
    codes = sr.astype(np.int64) * len(pc_) + pr
    assert len(np.unique(codes)) == len(codes)
    r = read_json(paths.run_dir("B04-SYN") / "metrics.json")
    assert r["matcher"] == "LightGBMModel" and 0 <= r["at_threshold_used"]["f05"] <= 1
