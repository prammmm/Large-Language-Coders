"""EXP-B-07: (exact word-order-insensitive name) x (useful address word) key for generic names."""

from ber.blocking.blocker import PASS_ORDER

from test_blocking_v2 import cfg as bcfg, pairs, table

EC = 1 << PASS_ORDER.index("exact_core")
SC = 1 << PASS_ORDER.index("sorted_core")


def ccfg(with_words=True, block=5, max_words=30, sorted_core=True):
    c = bcfg()
    c["passes"]["exact_core"] = {"enabled": True, "with_addr_words": with_words, "max_words": max_words,
                                 "max_block_size": block, "weight": 1.0}
    c["passes"]["sorted_core"] = {"enabled": sorted_core, "max_block_size": block, "weight": 1.0}
    return c


def keyset(got):
    return {(s, p) for s, p, _, _ in got}


def _generic_pool():
    # 8 "Royal Ventures" records in different cities: the exact-name block (8) exceeds cap 5
    cities = ["delhi", "nagpur", "pune", "indore", "kolkata", "chennai", "jaipur", "surat"]
    return table(["Royal Ventures"] * 8, [f"{i} some street {c}" for i, c in enumerate(cities)])


def test_generic_name_found_through_name_x_city_when_name_block_is_capped():
    s1 = table(["Ventures Royal"], ["c 199 sarvodaya enclave nagpur"])
    pool = _generic_pool()
    base, _ = pairs(s1, pool, ccfg(with_words=False))
    assert base == []  # sorted/exact name blocks of 8 > cap 5: nothing
    got, _ = pairs(s1, pool, ccfg())
    assert keyset(got) == {(0, 1)} and all(f & EC for _, _, f, _ in got)  # only the Nagpur record


def test_name_must_match_exactly_and_words_must_be_useful():
    s1 = table(["Royal Traders"], ["near road nagpur"])
    got, _ = pairs(s1, _generic_pool(), ccfg())
    assert got == []  # different name
    s1b = table(["Royal Ventures"], ["near road street"])  # only stop-words shared
    got, _ = pairs(s1b, table(["Royal Ventures"], ["near road street"]), ccfg())
    assert not any(f & EC for _, _, f, _ in got)


def test_default_exact_core_unchanged_and_block_cap_applies():
    s1 = table(["Royal Ventures"], ["x"])
    pool = table(["Royal Ventures", "Other"], ["y", "z"])
    assert keyset(pairs(s1, pool, ccfg(with_words=False, block=50, sorted_core=False))[0]) == {(0, 0)}
    s1c = table(["Royal Ventures"], ["nagpur"])
    poolc = table(["Royal Ventures"] * 6, ["nagpur"] * 6)  # (name, nagpur) block of 6 > cap 5
    got, b = pairs(s1c, poolc, ccfg(sorted_core=False))
    assert got == [] and b.stats["passes"]["exact_core"]["dropped_oversize_keys"] >= 1
