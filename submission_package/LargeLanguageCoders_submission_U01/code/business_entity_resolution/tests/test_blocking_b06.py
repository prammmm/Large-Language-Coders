"""EXP-B-06: name-token-pair key (name_token pass, pair mode) for name-only records."""

from ber.blocking.blocker import PASS_ORDER
from ber.normalization.crossscript import apply_map

from test_blocking_v2 import cfg as bcfg, pairs, table

NT = 1 << PASS_ORDER.index("name_token")


def pcfg(pair=True, name_max_df=1000, block=50, max_tokens=2, max_df=5, name_view=None):
    c = bcfg()
    c["passes"]["name_token"] = {"enabled": True, "pair": pair, "max_tokens": max_tokens, "name_max_df": name_max_df,
                                 "max_df": max_df, "max_block_size": block, "weight": 1.0}
    if name_view:
        c["name_view"] = name_view
    return c


def keyset(got):
    return {(s, p) for s, p, _, _ in got}


def test_pair_key_links_name_only_records_with_empty_addresses():
    s1 = table(["Harbor Lotus Clinic"], [""])
    pool = table(["Lotus Harbor Clinic", "Harbor Bakery", "Lotus Foods"], ["", "", ""])
    got, _ = pairs(s1, pool, pcfg())
    # the S1's 2 rarest pool-present tokens are harbor + lotus (clinic is equally rare, ties by id);
    # only the record sharing BOTH tokens is linked
    assert keyset(got) == {(0, 0)} and all(f & NT for _, _, f, _ in got)


def test_single_token_names_and_generic_tokens_give_no_pair_key():
    s1 = table(["Nova", "Nova Traders"], ["", ""])
    pool = table(["Nova", "Nova Traders", "Nova Traders"], ["", "", ""])
    got, _ = pairs(s1, pool, pcfg(name_max_df=1))  # "nova"/"traders" DF > 1 -> not usable
    assert got == []
    got, _ = pairs(s1, pool, pcfg())
    assert (0, 0) not in keyset(got)  # a one-token name has no pair
    assert {(1, 1), (1, 2)} <= keyset(got)


def test_block_cap_and_default_mode_unchanged():
    s1 = table(["Harbor Lotus"], [""])
    pool = table(["Harbor Lotus"] * 6, [""] * 6)  # one (harbor, lotus) block of 6 records
    got, b = pairs(s1, pool, pcfg(block=5))
    assert got == [] and b.stats["passes"]["name_token"]["dropped_oversize_keys"] >= 1
    # pair=False keeps the EXP-B-02 single-token behaviour
    s1b, poolb = table(["Zenith Cedar"], ["x"]), table(["Zenith Motors"], ["y"])
    assert keyset(pairs(s1b, poolb, pcfg(pair=False, max_df=5))[0]) == {(0, 0)}
    assert keyset(pairs(s1b, poolb, pcfg(pair=True))[0]) == set()  # only ONE shared token


def test_pair_key_uses_the_configured_name_view():
    m = {"शर्मा": "sharma", "मेडिकल": "medical"}
    s1 = table(["Sharma Medical"], [""])
    s1 = s1.append_column("name_xlit", apply_map(s1.column("name_core"), m))
    pool = table(["शर्मा मेडिकल"], [""])
    pool = pool.append_column("name_xlit", apply_map(pool.column("name_core"), m))
    assert keyset(pairs(s1, pool, pcfg())[0]) == set()
    assert keyset(pairs(s1, pool, pcfg(name_view="name_xlit"))[0]) == {(0, 0)}
