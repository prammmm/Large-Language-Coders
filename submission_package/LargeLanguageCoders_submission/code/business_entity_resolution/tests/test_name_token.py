"""EXP-B-03a: name-token blocking pass (key generation, generic-token filter, dedup, country
isolation, candidate validity, fixed-model scoring path)."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ber.blocking.blocker import PASS_ORDER, CountryBlocker
from ber.config import Paths, load_config
from ber.data.normalize_records import load_manifest, read_country
from ber.normalization.views import build_views
from ber.pipeline.run_baseline import main
from ber.utils.runtime import read_json

from synthetic import make_synthetic_repo

NT = np.uint8(1 << PASS_ORDER.index("name_token"))


def table(names, addrs):
    v = build_views(names, addrs)
    return pa.table({"row": pa.array(range(len(names)), pa.int32()), **{k: pa.array(x) for k, x in v.items()}})


def cfg(max_df=2, enabled=True):
    off = {"enabled": False, "max_block_size": 50, "weight": 1.0}
    return {"top_k": 20, "max_raw_pairs_per_chunk": 1000, "passes": {
        "exact_name": dict(off), "addr_num_word": {**off, "max_nums": 2, "max_words": 4, "word_max_df": 100},
        "addr_rare_word": {**off, "max_df": 1}, "exact_core": dict(off), "sorted_core": dict(off),
        "name_token": {"enabled": enabled, "max_df": max_df, "weight": 1.0}}}


def pairs(s1, pool, c):
    b = CountryBlocker(s1, pool, c)
    out = []
    for lo, hi in b.chunk_bounds():
        cc = b.candidates(lo, hi)
        out += list(zip(cc["s1_local"].tolist(), cc["pool_local"].tolist(), cc["pass_flags"].tolist()))
    return out


def test_name_token_keys_and_generic_filter():
    s1 = table(["Zephyrine Bakery", "Common Traders"], ["a", "b"])
    pool = table(["ZEPHYRINE Foods", "Common Motors", "Common Pharma", "Common Clinic", "Other"], ["c", "d", "e", "f", "g"])
    got = pairs(s1, pool, cfg(max_df=2))
    # rare token "zephyrine" (pool DF 1) -> candidate; "common" (pool DF 3 > 2) is generic -> no key
    assert {(s, p) for s, p, _ in got} == {(0, 0)}
    assert all(f == NT for _, _, f in got)
    assert pairs(s1, pool, cfg(enabled=False)) == []
    # raising max_df admits the generic token
    assert {(s, p) for s, p, _ in pairs(s1, pool, cfg(max_df=3))} == {(0, 0), (1, 1), (1, 2), (1, 3)}


def test_name_token_dedup_multiple_shared_tokens():
    s1 = table(["Zephyr Quasar Labs"], ["a"])
    pool = table(["Quasar Zephyr Holdings"], ["b"])  # two shared rare tokens -> ONE candidate
    got = pairs(s1, pool, cfg(max_df=5))
    assert len(got) == 1 and got[0][:2] == (0, 0)


def test_legal_forms_do_not_create_keys():
    s1 = table(["Alpha Pvt Ltd"], ["a"])
    pool = table(["Beta Private Limited"], ["b"])  # only legal forms shared; name_core drops them
    assert pairs(s1, pool, cfg(max_df=5)) == []


def test_pipeline_name_token_with_fixed_model(tmp_path_factory):
    """Synthetic end-to-end: fit a LightGBM model, then score a name-token config with that FIXED
    model file (the EXP-B-03a mechanism); candidates stay inside each country and are valid."""
    root = tmp_path_factory.mktemp("nt_repo")
    info = make_synthetic_repo(root, n_train=200, n_test=90)
    base = yaml.safe_load(Path(info["config"]).read_text(encoding="utf-8"))
    base["experiment"]["id"] = "BASE-LGBM"
    base["matcher"]["type"] = "lightgbm"
    base["decision"] = {"threshold_from": "model", "threshold": 0.5}
    base["fit"].update({"max_train_s1": None, "inner_tune_fraction": 0.3})
    base["lightgbm"] = {"num_boost_round": 50, "early_stopping_rounds": 10,
                        "params": {"num_leaves": 7, "min_data_in_leaf": 5, "num_threads": 1}}
    base["scoring"]["tag"] = "scored_base"
    p0 = root / "base.yaml"
    p0.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    for st in ("preprocess", "split", "fit", "train"):
        assert main([st, "--config", str(p0)]) == 0
    nt = yaml.safe_load(p0.read_text(encoding="utf-8"))
    nt["experiment"]["id"] = "NT"
    nt["blocking"]["passes"]["name_token"].update({"enabled": True, "max_df": 200})
    nt["scoring"]["tag"] = "scored_nt"
    nt["matcher"]["model_path"] = str(Paths(load_config(p0)).run_dir("BASE-LGBM") / "model.json")
    p1 = root / "nt.yaml"
    p1.write_text(yaml.safe_dump(nt, sort_keys=False), encoding="utf-8")
    for st in ("train", "infer", "submit"):
        assert main([st, "--config", str(p1)]) == 0
    paths = Paths(load_config(p1))
    r = read_json(paths.run_dir("NT") / "metrics.json")
    assert r["matcher"] == "LightGBMModel" and "BASE-LGBM" in r["threshold_source"]
    assert read_json(paths.run_dir("NT") / "submission_summary.json")["strict_validation_errors"] == []
    # country isolation + no duplicate pairs in the scored test candidates
    m = load_manifest(paths, "test")
    s1c = np.empty(m["source1"]["rows"], dtype=object)
    for k in m["source1"]["countries"]:
        s1c[read_country(paths, "test", "source1", k, ["row"]).column("row").to_numpy()] = k
    n2 = m["source2"]["rows"]
    n_pool = n2 + m["source3"]["rows"]
    pc_ = np.empty(n_pool, dtype=object)
    for src, off in (("source2", 0), ("source3", n2)):
        for k in m[src]["countries"]:
            pc_[read_country(paths, "test", src, k, ["row"]).column("row").to_numpy() + off] = k
    t = pa.concat_tables([pq.read_table(p) for p in paths.scored_dir("test", "scored_nt").glob("*.parquet")])
    sr, pr = t.column("s1_row").to_numpy(), t.column("pool_row").to_numpy()
    assert (s1c[sr] == pc_[pr]).all()
    codes = sr.astype(np.int64) * n_pool + pr
    assert len(np.unique(codes)) == len(codes)
