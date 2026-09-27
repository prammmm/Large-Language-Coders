"""Phase-2 components: blocking v1 keys, leading-zero handling, conflict features,
logistic matcher, config overrides, threshold sources, fit/evaluate leakage guards."""

import numpy as np
import pyarrow as pa
import pytest

from ber.blocking.blocker import CountryBlocker, sort_tokens, strip_leading_zeros
from ber.config import deep_update, load_config, parse_sets
from ber.decision.threshold import resolve_threshold, threshold_source
from ber.features.pair_features import legal_forms, pair_features, strip_zeros_list
from ber.models.logistic import LogisticModel, base_features, feature_matrix
from ber.normalization.views import build_views


def table(names, addrs):
    v = build_views(names, addrs)
    return pa.table({"row": pa.array(range(len(names)), pa.int32()), **{k: pa.array(x) for k, x in v.items()}})


def cfg_with(**passes):
    base = {
        "top_k": 10,
        "max_raw_pairs_per_chunk": 1000,
        "passes": {
            "exact_name": {"enabled": True, "max_block_size": 50, "weight": 2.0},
            "addr_num_word": {"enabled": False, "max_block_size": 50, "max_nums": 2, "max_words": 4, "word_max_df": 100, "weight": 1.0},
            "addr_rare_word": {"enabled": False, "max_df": 1, "weight": 1.0},
            "name_token": {"enabled": False, "max_df": 10, "weight": 1.0},
            "exact_core": {"enabled": False, "max_block_size": 50, "weight": 2.0},
            "sorted_core": {"enabled": False, "max_block_size": 50, "weight": 2.0},
        },
    }
    for k, v in passes.items():
        if k == "strip_leading_zeros":
            base[k] = v
        else:
            base["passes"][k]["enabled"] = v
    return base


def pairs(s1, pool, cfg):
    b = CountryBlocker(s1, pool, cfg)
    out = set()
    for lo, hi in b.chunk_bounds():
        c = b.candidates(lo, hi)
        out |= set(zip(c["s1_local"].tolist(), c["pool_local"].tolist()))
    return out


# ---------------------------------------------------------------- blocking v1
def test_transforms():
    got = strip_leading_zeros(pa.array(["04126 007 0 00 b04 045nd 100", "", "10 01"])).to_pylist()
    assert got == ["4126 7 0 0 b04 45nd 100", "", "10 1"]
    assert sort_tokens(pa.array(["parts advanced", "", "b a"])).to_pylist() == ["advanced parts", "", "a b"]


def test_exact_core_key_ignores_legal_suffix():
    s1 = table(["Surgical Physicians of Seabrook"], ["x"])
    pool = table(["Surgical Physicians of Seabrook Ltd", "Other Clinic"], ["y", "z"])
    assert pairs(s1, pool, cfg_with()) == set()  # name_norm differs by the suffix
    assert pairs(s1, pool, cfg_with(exact_core=True)) == {(0, 0)}


def test_sorted_core_key_is_word_order_insensitive():
    s1 = table(["Advanced Parts LLC"], ["x"])
    pool = table(["Advanced LLC Parts", "Parts Unlimited"], ["y", "z"])
    assert pairs(s1, pool, cfg_with(exact_core=True)) == {(0, 0)}  # core = "advanced parts" both
    s1b = table(["Parts Advanced"], ["x"])
    assert pairs(s1b, pool, cfg_with(exact_core=True)) == set()
    assert pairs(s1b, pool, cfg_with(sorted_core=True)) == {(0, 0)}


def test_leading_zero_option_in_address_number_keys():
    s1 = table(["A Co"], ["4126 Lakepointe Forest Drive Seabrook"])
    pool = table(["Zzz"], ["04126 Lakepointe Forest Dr Seabrook"])
    cfg = cfg_with(addr_num_word=True)
    cfg["passes"]["exact_name"]["enabled"] = False
    assert pairs(s1, pool, cfg) == set()
    cfg["strip_leading_zeros"] = True
    assert pairs(s1, pool, cfg) == {(0, 0)}


def test_new_pass_flags_distinct():
    s1 = table(["Parts Advanced"], ["x"])
    pool = table(["Advanced Parts Inc"], ["y"])
    b = CountryBlocker(s1, pool, cfg_with(sorted_core=True))
    c = b.candidates(0, 1)
    from ber.blocking.blocker import PASS_ORDER

    assert c["pass_flags"].tolist() == [1 << PASS_ORDER.index("sorted_core")]


# ---------------------------------------------------------------- features
def test_strip_zeros_list():
    assert strip_zeros_list(["04126 007", "", "0 00 0a 045nd"]) == ["4126 7", "", "0 0 0a 45nd"]


def test_legal_forms_canonical():
    assert legal_forms("sbr and sons private limited") == {"ltd"}
    assert legal_forms("sbr and sons llp") == {"llp"}
    assert legal_forms("acme ltd") == legal_forms("acme limited")
    assert legal_forms("acme co") == set()


def test_conflict_features_and_zero_option():
    s1 = table(["New Delhi Son LLP", "Laex Inc", "Acme Ltd"], ["628 10 Mohalla", "100 8th Street", "12 Main St"])
    pool = table(["New Delhi Son Limited", "Laex Inc", "Acme Limited"], ["628 13 Mohalla", "Pleasant Street", "012 Main St"])
    idx = np.arange(3)
    f = pair_features(s1, pool, idx, idx, options={"extra": ["num_conflict", "legal_form_conflict"]})
    assert f["legal_form_conflict"].tolist() == [1.0, 0.0, 0.0]  # LLP vs Limited; Ltd == Limited
    assert f["num_conflict"].tolist() == [0.0, 0.0, 1.0]  # shares 628 / one side has no number / 12 vs 012
    f2 = pair_features(s1, pool, idx, idx, options={"strip_leading_zeros": True, "extra": ["num_conflict"]})
    assert f2["num_conflict"].tolist() == [0.0, 0.0, 0.0]
    assert f2["addr_num_jaccard"][2] == 1.0
    with pytest.raises(ValueError):
        pair_features(s1, pool, idx, idx, options={"extra": ["country_onehot"]})


def test_default_features_unchanged_without_options():
    s1 = table(["Acme Ltd"], ["012 Main St"])
    pool = table(["Acme Limited"], ["12 Main St"])
    a = pair_features(s1, pool, np.array([0]), np.array([0]))
    b = pair_features(s1, pool, np.array([0]), np.array([0]), options={})
    assert set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)
    assert "num_conflict" not in a


# ---------------------------------------------------------------- logistic matcher
def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.random(n), rng.random(n)
    logit = -4 + 5 * x1 + 3 * x2
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    return {"x1": x1.astype(np.float32), "x2": x2.astype(np.float32)}, y


def test_logistic_recovers_signal_and_gradient_is_zero():
    feats, y = _toy()
    m = LogisticModel(["x1", "x2"]).fit(feature_matrix(feats, ["x1", "x2"]), y, l2=0.0)
    w = m.coef / m.std  # back to raw scale
    assert w[0] > 3 and w[1] > 1.5 and w[0] > w[1]
    X = (feature_matrix(feats, ["x1", "x2"]) - m.mean) / m.std
    p = 1 / (1 + np.exp(-(X @ m.coef + m.intercept)))
    grad = np.r_[np.mean(p - y), X.T @ (p - y) / len(y)]
    assert np.abs(grad).max() < 1e-6


def test_logistic_roundtrip_and_interactions(tmp_path):
    feats, y = _toy()
    names = ["x1", "x2", "x1*x2"]
    assert base_features(names) == ["x1", "x2"]
    m = LogisticModel(names, meta={"threshold": 0.5}).fit(feature_matrix(feats, names), y)
    m.save(tmp_path / "m.json")
    m2 = LogisticModel.load(tmp_path / "m.json")
    assert np.allclose(m.score(feats), m2.score(feats))
    assert m2.meta["threshold"] == 0.5 and m2.required_features == ["x1", "x2"]
    s = m.score(feats)
    assert s.dtype == np.float32 and (s >= 0).all() and (s <= 1).all()


def test_logistic_constant_column_is_safe():
    feats, y = _toy()
    feats["c"] = np.ones_like(feats["x1"])
    m = LogisticModel(["x1", "c"]).fit(feature_matrix(feats, ["x1", "c"]), y)
    assert np.isfinite(m.coef).all()


# ---------------------------------------------------------------- config / decision
def test_parse_sets_and_inherit(tmp_path):
    o = parse_sets(["blocking.top_k=30", "matcher.type=logistic", "features.extra=[num_conflict]", "a.b.c=0.5"])
    assert o == {"blocking": {"top_k": 30}, "matcher": {"type": "logistic"}, "features": {"extra": ["num_conflict"]}, "a": {"b": {"c": 0.5}}}
    (tmp_path / "base.yaml").write_text("x: 1\ny: {a: 1, b: 2}\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text("inherit: base.yaml\ny: {b: 3}\n", encoding="utf-8")
    c = load_config(tmp_path / "child.yaml")
    assert c["x"] == 1 and c["y"] == {"a": 1, "b": 3}
    with pytest.raises(ValueError):
        parse_sets(["novalue"])
    assert deep_update({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}


def test_threshold_sources(tmp_path):
    assert threshold_source({"use_tuned_threshold": True, "threshold": 0.7}) == "val_sweep"
    assert threshold_source({"threshold": 0.7}) == "config"
    mf = tmp_path / "model.json"
    LogisticModel(["x"], meta={"threshold": 0.42}).save(mf)
    t, src = resolve_threshold({"threshold_from": "model", "threshold": 0.7}, tmp_path / "none.json", mf)
    assert t == 0.42 and "train-fold" in src
    t, src = resolve_threshold({"threshold_from": "model", "threshold": 0.7}, tmp_path / "none.json", tmp_path / "missing.json")
    assert t == 0.7 and src == "config"
    with pytest.raises(ValueError):
        threshold_source({"threshold_from": "test_set"})
