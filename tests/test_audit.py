"""FINAL RESEARCH AUDIT helpers: per-pair profile, key families and the emitted-pair estimator are
checked against brute-force Python on small tables."""

import numpy as np
import pyarrow as pa

from ber.normalization.views import ADDRESS_STOPWORDS

from ber.forensics.audit import (cardinality_bin, emit_pairs, oracle_gain, encode_views, family_keys, jaccard, numbers, pair_profile, script_class, tokens,
                                 useful_words)


def tbl(names, addrs, nums):
    words = [a.lower() for a in addrs]
    return pa.table({"name_xlit": names, "addr_words": words, "addr_nums": nums})


def test_script_tokens_numbers_and_jaccard():
    assert script_class("sharma medical") == "latin" and script_class("शर्मा") == "nonlatin"
    assert script_class("sharma मेडिकल") == "mixed" and script_class("123 &") == "empty"
    assert numbers("0012 45") == {"12", "45"} and tokens("a b a") == {"a", "b"}
    stop = sorted(ADDRESS_STOPWORDS)[0]
    assert useful_words(f"{stop} x zz 5a") == {"zz", "5a"}  # stopwords and 1-char words are not useful
    assert np.isnan(jaccard(set(), set())) and jaccard({"a"}, {"a", "b"}) == 0.5


def test_pair_profile_flags():
    s1 = {"name_norm": "sharma medical pvt ltd", "name_core": "sharma medical", "name_xlit": "sharma medical", "addr_norm": "12 mg nagar",
          "addr_nums": "12", "addr_words": "mg nagar"}
    rec = {"name_norm": "शर्मा मेडिकल", "name_core": "शर्मा मेडिकल", "name_xlit": "sharma medical", "addr_norm": "", "addr_nums": "", "addr_words": ""}
    p = pair_profile(s1, rec)
    assert p["script_mismatch"] and p["xlit_bridges_script"] and p["xlit_equal"] and p["rec_addr_empty"] and p["name_only"]
    assert not p["shared_number"] and p["xlit_shared"] == 2 and p["name_core_shared"] == 0
    q = pair_profile(s1, dict(rec, name_xlit="kumar stores", name_core="kumar stores", addr_nums="0012", addr_norm="12"))
    assert q["number_only"] and q["shared_number"] and q["address_only"] and not q["name_only"]


def brute_keys(recs_tokens, df, k):
    """rarest k tokens (ties by id) per record."""
    return [sorted(ts, key=lambda t: (df[t], t))[:k] for ts in recs_tokens]


def test_family_keys_and_emit_pairs_match_brute_force():
    s1 = tbl(["alpha beta", "gamma", "alpha delta"], ["mg nagar", "x", "park lane"], ["12", "", "7"])
    pool = tbl(["alpha beta co", "beta alpha", "gamma", "delta alpha", "alpha"], ["nagar mg", "", "zz", "lane park", "mg"], ["12", "12", "", "07", "12"])
    enc = encode_views(s1, pool)
    # tok_x_num with k 2 x 2: brute force over decoded tokens
    r, k = family_keys(enc, "tok_x_num", 2, 2)
    assert len(r) == len(np.unique(r * 10**6 + k))  # distinct (record, key)
    s_rec, p_rec, st = emit_pairs(r, k, enc["n_s1"], cap=50)
    got = set(zip(s_rec.tolist(), p_rec.tolist()))
    # expected: S1 0 (alpha beta, 12) links pool 0 (alpha beta, 12), pool 1 (beta alpha, 12), pool 4 (alpha, 12); S1 2 (alpha delta, 7) links pool 3 (07)
    assert got == {(0, 0), (0, 1), (0, 4), (2, 3)}
    # the cap drops blocks larger than cap
    s_rec2, _, st2 = emit_pairs(r, k, enc["n_s1"], cap=1)
    assert st2["keys_over_cap"] > 0 and (0, 1) not in set(zip(s_rec2.tolist(), _.tolist()))
    # token pair: S1 0 (alpha, beta) -> pool 0 and 1 (both contain alpha+beta)
    r, k = family_keys(enc, "tok_pair", 3)
    s_rec, p_rec, _ = emit_pairs(r, k, enc["n_s1"], cap=50)
    assert {(0, 0), (0, 1), (2, 3)} <= set(zip(s_rec.tolist(), p_rec.tolist()))
    # s1_mask restricts emitting S1
    s_rec, _, _ = emit_pairs(r, k, enc["n_s1"], cap=50, s1_mask=np.array([False, False, True]))
    assert set(s_rec.tolist()) <= {2}


def test_single_tok_respects_df_and_word_pair_unordered():
    s1 = tbl(["rare common"], ["aa bb cc"], [""])
    pool = tbl(["common"] * 5 + ["rare x"], ["bb aa"] * 3 + ["cc bb", "", ""], [""] * 6)
    enc = encode_views(s1, pool)
    r, k = family_keys(enc, "single_tok", max_df=1)
    s_rec, p_rec, _ = emit_pairs(r, k, enc["n_s1"], cap=50)
    assert set(zip(s_rec.tolist(), p_rec.tolist())) == {(0, 5)}  # only the rare token links
    r, k = family_keys(enc, "word_pair", 3)
    s_rec, p_rec, _ = emit_pairs(r, k, enc["n_s1"], cap=50)
    assert {(0, 0), (0, 1), (0, 2), (0, 3)} == set(zip(s_rec.tolist(), p_rec.tolist()))


def test_single_word_oracle_gain_and_cardinality():
    s1 = tbl(["x"], ["rareword common"], [""])
    pool = tbl(["y", "z"], ["rareword", "common"], ["", ""])
    enc = encode_views(s1, pool)
    r, k = family_keys(enc, "single_word", max_df=1)
    s_rec, p_rec, _ = emit_pairs(r, k, enc["n_s1"], cap=50)
    assert set(zip(s_rec.tolist(), p_rec.tolist())) == {(0, 0), (0, 1)}  # both words have pool DF 1
    tp, k_, n = np.array([1, 0, 0]), np.array([2, 1, 0]), np.array([2, 0, 1])
    # removing the false pair of S1 0 and the singleton FP of S1 1 improves F0.5; adding S1 2's true pair too
    assert oracle_gain(tp, k_, n, remove_false=np.array([1, 1, 0])) > 0
    assert oracle_gain(tp, k_, n, add_true=np.array([0, 0, 1])) > 0 and oracle_gain(tp, k_, n) == 0
    assert cardinality_bin(np.array([0, 1, 2, 3, 5, 6, 40])).tolist() == [0, 1, 2, 3, 3, 4, 4]


def test_gross_pairs_and_share_key_match_emit_pairs():
    from ber.forensics.audit import gross_pairs, pool_key_index, share_key
    s1 = tbl(["alpha beta", "gamma", "alpha delta"], ["mg nagar", "x", "park lane"], ["12", "", "7"])
    pool = tbl(["alpha beta co", "beta alpha", "gamma", "delta alpha", "alpha"], ["nagar mg", "", "zz", "lane park", "mg"], ["12", "12", "", "07", "12"])
    enc = encode_views(s1, pool)
    r, k = family_keys(enc, "tok_x_num", 2, 2)
    uk, cnt, sr, sk = pool_key_index(r, k, enc["n_s1"])
    g, over = gross_pairs(uk, cnt, sr, sk, cap=50)
    s_rec, p_rec, st = emit_pairs(r, k, enc["n_s1"], cap=50)
    assert int(g.sum()) == st["gross_pairs"] and over == 0
    pairs = [(0, 0), (0, 1), (0, 2), (2, 3), (1, 2)]
    got = share_key(r, k, enc["n_s1"], np.array([a for a, _ in pairs]), np.array([b for _, b in pairs]))
    emitted = set(zip(s_rec.tolist(), p_rec.tolist()))
    assert got.tolist() == [p in emitted for p in pairs]
    capped = share_key(r, k, enc["n_s1"], np.array([0]), np.array([1]), cap=1)  # (alpha,12) has pool block 3 > cap 1
    assert capped.tolist() == [((0, 1) in set(zip(*emit_pairs(r, k, enc["n_s1"], cap=1)[:2])))]
