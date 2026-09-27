import numpy as np
import pyarrow as pa
import pytest

from ber.blocking.blocker import (
    CountryBlocker,
    cross_within_records,
    encode_columns,
    explode_tokens,
    num_word_keys,
    ragged_gather,
    rank_within_groups,
    rarest_per_record,
    sort_by_key,
)
from ber.features.pair_features import FEATURE_NAMES, pair_features
from ber.models.weighted_rule import WeightedRuleScorer
from ber.normalization.views import build_views

BLOCK_CFG = {
    "top_k": 5,
    "max_raw_pairs_per_chunk": 3,
    "passes": {
        "exact_name": {"enabled": True, "max_block_size": 2, "weight": 2.0},
        "addr_num_word": {"enabled": True, "max_block_size": 5, "max_nums": 2, "max_words": 4, "word_max_df": 100, "weight": 1.0},
        "addr_rare_word": {"enabled": True, "max_df": 1, "weight": 1.0},
        "name_token": {"enabled": False, "max_df": 10, "weight": 1.0},
    },
}


def table(names, addrs):
    v = build_views(names, addrs)
    return pa.table({"row": pa.array(range(len(names)), pa.int32()), **{k: pa.array(x) for k, x in v.items()}})


def test_helpers():
    assert rank_within_groups(np.array([0, 0, 1, 1, 1, 3])).tolist() == [0, 1, 0, 1, 2, 0]
    assert ragged_gather(np.array([0, 3]), np.array([2, 1]), np.arange(10, 20)).tolist() == [10, 11, 13]
    rec, key = cross_within_records(np.array([0, 1]), np.array([7, 8]), np.array([0, 0, 1]), np.array([1, 2, 3]), 10)
    assert rec.tolist() == [0, 0, 1] and key.tolist() == [71, 72, 83]
    r, t, vocab = explode_tokens(pa.array(["a b a", "", "b c"]))
    pairs = sorted(zip(r.tolist(), [vocab[i].as_py() for i in t]))
    assert pairs == [(0, "a"), (0, "b"), (2, "b"), (2, "c")]  # dedup within record, empty skipped


def test_encode_columns_consistent_across_columns_and_chunks():
    a = pa.array(["x y", "z", ""])
    b = pa.chunked_array([pa.array(["y x x", "w"]), pa.array(["z w"])])
    for chunk in (1, 2, 100):
        rec, tok, vocab = encode_columns([a, b], split=True, chunk_rows=chunk)
        got = sorted((int(r), vocab[int(t)].as_py()) for r, t in zip(rec, tok))
        assert got == [(0, "x"), (0, "y"), (1, "z"), (3, "x"), (3, "y"), (4, "w"), (5, "w"), (5, "z")]
        assert list(rec) == sorted(rec)  # ascending records (required by num_word_keys)
    rec, tok, vocab = encode_columns([pa.array(["ab c", ""]), pa.array(["ab c"])], split=False, chunk_rows=1)
    assert rec.tolist() == [0, 2] and tok[0] == tok[1]


def test_sort_by_key_packed_and_fallback_agree():
    rng = np.random.default_rng(0)
    key = rng.integers(0, 50, 1000)
    rec = rng.permutation(1000)
    k1, r1 = sort_by_key(key, rec)
    order = np.lexsort((rec, key))
    assert k1.tolist() == key[order].tolist() and r1.tolist() == rec[order].tolist()
    big = key.astype(np.int64) + (1 << 55)  # forces the lexsort fallback
    k2, r2 = sort_by_key(big, rec)
    assert r2.tolist() == r1.tolist()


def test_num_word_keys_windowed_equals_global():
    rng = np.random.default_rng(1)
    n_rec = np.sort(rng.integers(0, 40, 120)).astype(np.int32)
    n_tok = rng.integers(0, 15, 120).astype(np.int32)
    w_rec = np.sort(rng.integers(0, 40, 200)).astype(np.int32)
    w_tok = rng.integers(0, 30, 200).astype(np.int32)
    code = np.unique(n_rec.astype(np.int64) * 15 + n_tok)
    n_rec, n_tok = (code // 15).astype(np.int32), (code % 15).astype(np.int32)
    code = np.unique(w_rec.astype(np.int64) * 30 + w_tok)
    w_rec, w_tok = (code // 30).astype(np.int32), (code % 30).astype(np.int32)
    n_df, w_df = rng.integers(1, 9, 15), rng.integers(1, 9, 30)
    ok = rng.random(30) < 0.8
    ref_n = rarest_per_record(n_rec, n_tok, n_df, 2)
    sel = ok[w_tok]
    ref_w = rarest_per_record(w_rec[sel], w_tok[sel], w_df, 3)
    ref = cross_within_records(*ref_n, *ref_w, 30)
    for window in (1, 7, 1000):
        r, k = num_word_keys(n_rec, n_tok, n_df, w_rec, w_tok, w_df, ok, 2, 3, 30, window)
        assert sorted(zip(r.tolist(), k.tolist())) == sorted(zip(ref[0].tolist(), ref[1].tolist()))


def _blocked(s1, pool, cfg=BLOCK_CFG):
    b = CountryBlocker(s1, pool, cfg)
    out = []
    for lo, hi in b.chunk_bounds():
        c = b.candidates(lo, hi)
        out += list(zip(c["s1_local"].tolist(), c["pool_local"].tolist(), c["cheap_rank"].tolist()))
    return b, out


def test_blocks_union_dedup_and_caps():
    s1 = table(["Apex Traders Inc", "Nova Foods"], ["12 Lake Road Pune", "99 Hill Drive Tulsa"])
    pool = table(
        [
            "APEX TRADERS",  # 0: address match (num+word) only
            "Apex Traders Inc",  # 1: exact name + address -> must appear once (dedup)
            "Zeta Motors",  # 2: unrelated
            "Nova Foods",  # 3: exact name, different address
            "Nova Foods",  # 4: exact name
            "Nova Foods",  # 5: exact name -> block of 3 > cap 2 -> dropped
        ],
        ["12 LAKE RD PUNE", "12 Lake Road, Pune", "5 Other Street", "1 Elm Lane", "2 Elm Lane", "3 Elm Lane"],
    )
    b, pairs = _blocked(s1, pool)
    got = {(s, p) for s, p, _ in pairs}
    assert len(got) == len(pairs), "duplicates in candidates"
    assert (0, 0) in got and (0, 1) in got
    assert (0, 2) not in got
    assert not any(s == 1 and p in (3, 4, 5) for s, p in got), "oversize exact-name block must be skipped"
    assert b.stats["passes"]["exact_name"]["dropped_oversize_keys"] == 1
    # both passes hit (0,1) -> it outranks (0,0)
    ranks = {(s, p): r for s, p, r in pairs}
    assert ranks[(0, 1)] == 0
    assert len(b.chunk_bounds()) >= 1


def test_top_k_cap_and_ranking_deterministic():
    s1 = table(["Solo"], ["7 Oak Street Springfield"])
    pool = table([f"Rec {i}" for i in range(4)], ["7 Oak Street Springfield"] * 4)
    cfg = {**BLOCK_CFG, "top_k": 2}
    _, pairs = _blocked(s1, pool, cfg)
    assert [p for _, p, _ in pairs] == [0, 1]  # ties broken by lower pool index
    _, again = _blocked(s1, pool, cfg)
    assert pairs == again


def test_empty_address_and_no_candidates():
    s1 = table(["Lonely Name"], [""])
    pool = table(["Other"], ["1 Road"])
    _, pairs = _blocked(s1, pool)
    assert pairs == []


def test_pair_features_ranges_and_values():
    s1 = table(["Apex Traders Inc", "Raj Pharma Pvt Ltd"], ["12 Lake Road Pune", "4 MG Road Indore"])
    pool = table(["APEX TRADERS", "राज फार्मा", "Zeta"], ["12 LAKE RD PUNE", "4 MG Road Indore", ""])
    f = pair_features(s1, pool, np.array([0, 1, 0]), np.array([0, 1, 2]))
    assert set(f) == set(FEATURE_NAMES)
    for k, v in f.items():
        assert v.dtype == np.float32 and len(v) == 3, k
    assert f["name_core_eq"][0] == 1.0 and f["name_norm_eq"][0] == 0.0
    assert f["addr_norm_eq"][0] == 1.0 and f["addr_jaccard"][0] == 1.0
    assert f["addr_num_jaccard"][1] == 1.0 and f["addr_num_shared"][1] == 1.0
    assert f["name_core_jaccard"][1] == 0.0  # script mismatch, no transliteration in v0
    assert f["addr_any_empty"].tolist() == [0.0, 0.0, 1.0]
    assert f["addr_ratio"][2] == 0.0
    for k in ("name_ratio", "name_token_set", "name_concat_jw", "addr_ratio", "addr_token_set", "name_core_jaccard", "addr_jaccard"):
        assert (f[k] >= 0).all() and (f[k] <= 1).all(), k


def test_weighted_rule_scorer():
    s = WeightedRuleScorer({"name_core_jaccard": 1.0, "addr_shared_useful": 1.0}, {"addr_shared_useful": 2})
    out = s.score({"name_core_jaccard": np.array([1.0, 0.0], np.float32), "addr_shared_useful": np.array([4.0, 1.0], np.float32)})
    assert out.tolist() == pytest.approx([1.0, 0.25])
    with pytest.raises(ValueError):
        WeightedRuleScorer({"country_onehot": 1.0})
