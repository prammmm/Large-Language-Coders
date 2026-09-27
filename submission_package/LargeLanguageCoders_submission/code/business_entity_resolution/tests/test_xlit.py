"""EXP-N-02: train-pair-learned cross-script name view — unit tests, leakage guards, blocking,
features and a synthetic end-to-end run (map -> blocking -> features -> LightGBM retrain -> D-01)."""

import csv
import random
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from ber.blocking.blocker import PASS_ORDER
from ber.config import Paths, load_config
from ber.data.split import load_val_ids
from ber.features.pair_features import pair_features
from ber.normalization.crossscript import DROP, LeakageError, XlitMap, apply_map, apply_map_one, learn_legal_tokens, learn_token_map, pair_tokens, token_script
from ber.pipeline.run_baseline import _ids, fit_s1_subset, main
from ber.pipeline.xlit import crossfit_plan, fold_of, learn_map_for
from ber.utils.runtime import read_json

from synthetic import make_synthetic_repo
from test_blocking_v2 import cfg as bcfg, pairs, table

DEV_SHARMA, DEV_MEDICAL, DEV_STORE = "शर्मा", "मेडिकल", "स्टोर"
SC = 1 << PASS_ORDER.index("sorted_core")


# ----------------------------------------------------------------------------- script detection / tokens
def test_script_detection():
    assert token_script("sharma") == "latin" and token_script("café") == "latin" and token_script("ǆ") == "latin"
    assert token_script(DEV_SHARMA) == "nonlatin" and token_script("சென்னை") == "nonlatin" and token_script("mixedशर्मा") == "nonlatin"
    assert token_script("123") == "none" and token_script("") == "none"


def test_pair_token_extraction_uses_sets_and_scripts():
    lat, non = pair_tokens("sharma medical sharma 24", f"{DEV_SHARMA} {DEV_MEDICAL} {DEV_SHARMA} 24 store")
    assert lat == {"sharma", "medical"} and non == {DEV_SHARMA, DEV_MEDICAL}  # numbers / Latin pool tokens ignored
    assert pair_tokens("", DEV_SHARMA) == (set(), {DEV_SHARMA}) and pair_tokens("abc", "") == ({"abc"}, set())


# ----------------------------------------------------------------------------- learning
def _pairs(n, lat, non, start=0):
    return [(f"S1-{start + i}", lat, non) for i in range(n)]


def test_support_counts_distinct_s1_entities_not_pairs():
    # one entity with 10 pool records must not reach support 3
    m, st = learn_token_map([("S1-0", "sharma", DEV_SHARMA)] * 10, min_support=3, min_confidence=0.5)
    assert m == {} and st["rejected"] == {"low_support": 1}
    m, _ = learn_token_map(_pairs(3, "sharma", DEV_SHARMA), 3, 0.5)
    assert m == {DEV_SHARMA: "sharma"}


def test_word_order_and_multi_token_names_resolve_by_cooccurrence():
    data = _pairs(4, "sharma medical", f"{DEV_MEDICAL} {DEV_SHARMA}") + _pairs(4, "gupta medical", f"{DEV_MEDICAL} गुप्ता", 100) + \
        _pairs(4, "sharma store", f"{DEV_SHARMA} {DEV_STORE}", 200)
    m, st = learn_token_map(data, 3, 0.5)
    # tokens seen in several contexts resolve regardless of word order ...
    assert m[DEV_SHARMA] == "sharma" and m[DEV_MEDICAL] == "medical"
    # ... a token only ever seen in ONE context ties between its co-occurring words: rejected, never guessed
    assert "गुप्ता" not in m and DEV_STORE not in m and st["rejected"]["ambiguous_tie"] == 2
    more = data + _pairs(3, "gupta traders", "गुप्ता ट्रेडर्स", 300)
    assert learn_token_map(more, 3, 0.5)[0]["गुप्ता"] == "gupta"  # a second context disambiguates it


def test_confidence_filter_and_ambiguous_ties_rejected_deterministically():
    noisy = _pairs(3, "alpha", "ब") + _pairs(3, "beta", "ब", 10) + _pairs(3, "gamma", "ब", 20)  # conf 1/3
    m, st = learn_token_map(noisy, 3, 0.5)
    assert "ब" not in m and st["rejected"].get("low_confidence") == 1
    tie = _pairs(4, "alpha beta", "क")  # alpha and beta always co-occur: a tie
    m, st = learn_token_map(tie, 3, 0.5)
    assert m == {} and st["rejected"] == {"ambiguous_tie": 1}


def test_learning_is_order_independent():
    data = _pairs(5, "sharma medical", f"{DEV_SHARMA} {DEV_MEDICAL}") + _pairs(4, "gupta medical", f"गुप्ता {DEV_MEDICAL}", 50)
    ref, _ = learn_token_map(data, 3, 0.5)
    for seed in range(5):
        d = list(data)
        random.Random(seed).shuffle(d)
        assert learn_token_map(d, 3, 0.5)[0] == ref


# ----------------------------------------------------------------------------- EXP-N-03 legal-form stage
LEGAL = frozenset({"private", "limited", "pvt", "ltd", "llp"})
DEV_PVT, DEV_LTD = "प्राइवेट", "लिमिटेड"


def _legal_pairs():
    # 30 DIFFERENT businesses "<name> private limited" written in Devanagari with the legal words
    names = [f"name{i}" for i in range(30)]
    return [(f"S1-{i}", f"{n} private limited", f"देव{i} {DEV_PVT} {DEV_LTD}") for i, n in enumerate(names)]


def test_legal_stage_learns_legal_words_from_diverse_businesses():
    got, st = learn_legal_tokens(_legal_pairs(), mapped=set(), legal_forms=LEGAL, min_support=20, min_confidence=0.8)
    assert got == {DEV_PVT, DEV_LTD} and st["legal_tokens"] == 2
    # every देव{i} token is seen once -> low support, never a legal word
    assert st["rejected"]["low_support"] == 30


def test_legal_stage_never_drops_a_real_name_word():
    # गुप्ता always appears with "gupta" (a content word) even though every business is "... pvt ltd"
    data = [(f"S1-{i}", "gupta medical pvt ltd", f"गुप्ता {DEV_PVT}") for i in range(40)]
    got, st = learn_legal_tokens(data, mapped=set(), legal_forms=LEGAL, min_support=20, min_confidence=0.8)
    assert "गुप्ता" not in got and st["rejected"]["explained_by_content_word"] >= 1
    # tokens already mapped by stage 1 are not reconsidered
    got2, _ = learn_legal_tokens(_legal_pairs(), mapped={DEV_PVT}, legal_forms=LEGAL, min_support=20, min_confidence=0.8)
    assert got2 == {DEV_LTD}


def test_legal_stage_confidence_filter():
    # the token appears with a legal form in only half of its businesses
    data = [(f"S1-{i}", f"n{i} private" if i % 2 else f"n{i}", f"x{i} {DEV_PVT}") for i in range(60)]
    got, st = learn_legal_tokens(data, mapped=set(), legal_forms=LEGAL, min_support=20, min_confidence=0.8)
    assert DEV_PVT not in got and st["rejected"]["low_confidence"] >= 1


def test_drop_tokens_in_view_and_empty_fallback():
    m = {DEV_SHARMA: "sharma", DEV_PVT: DROP, DEV_LTD: DROP}
    assert apply_map_one(f"{DEV_SHARMA} {DEV_PVT} {DEV_LTD}", m) == "sharma"
    assert apply_map_one(f"{DEV_PVT} {DEV_LTD}", m) == f"{DEV_PVT} {DEV_LTD}"  # never empty: unchanged
    assert apply_map_one("sharma private", m) == "sharma private"  # Latin names untouched


# ----------------------------------------------------------------------------- applying
def test_apply_unseen_mixed_duplicates_empty_and_ascii_fast_path():
    m = {DEV_SHARMA: "sharma", DEV_MEDICAL: "medical"}
    assert apply_map_one(f"{DEV_SHARMA} {DEV_MEDICAL}", m) == "sharma medical"
    assert apply_map_one(f"sharma {DEV_SHARMA} 24", m) == "sharma 24"  # duplicate after mapping removed
    assert apply_map_one(f"{DEV_SHARMA} अज्ञात", m) == "sharma अज्ञात"  # unseen token kept
    assert apply_map_one("", m) == "" and apply_map_one("plain latin", m) == "plain latin"
    arr = pa.array(["plain", f"{DEV_SHARMA} x", "", None], pa.string())
    out = apply_map(arr, m)
    assert out.to_pylist() == ["plain", "sharma x", "", None]
    assert apply_map(arr, {}) is arr  # no-map fallback: untouched


def test_map_roundtrip(tmp_path):
    x = XlitMap({DEV_SHARMA: "sharma"}, {"learned_from_digest": XlitMap.ids_digest(["b", "a"])})
    x.save(tmp_path / "m.json")
    y = XlitMap.load(tmp_path / "m.json")
    assert y.mapping == x.mapping and y.meta["learned_from_digest"] == XlitMap.ids_digest(["a", "b"])


# ----------------------------------------------------------------------------- blocking / features
def _tbl(names, addrs, m=None):
    t = table(names, addrs)
    return t if m is None else t.append_column("name_xlit", apply_map(t.column("name_core"), m))


def test_blocking_uses_xlit_view_only_when_configured():
    m = {DEV_SHARMA: "sharma", DEV_MEDICAL: "medical"}
    s1 = _tbl(["Sharma Medical"], ["12 station road pune"], m)
    pool = _tbl([f"{DEV_SHARMA} {DEV_MEDICAL}", "Other Shop"], ["somewhere else", "12 station road pune"], m)
    c = bcfg()
    c["passes"]["sorted_core"] = {"enabled": True, "max_block_size": 50, "weight": 1.0}
    base, _ = pairs(s1, pool, c)
    assert (0, 0) not in {(s, p) for s, p, _, _ in base}  # B-04 behaviour: cross-script pair unreachable
    c2 = dict(c, name_view="name_xlit")
    got, _ = pairs(s1, pool, c2)
    assert (0, 0) in {(s, p) for s, p, _, _ in got} and all(f & SC for s, p, f, _ in got if (s, p) == (0, 0))
    # all-Latin records: identical keys with and without the view
    s1b, poolb = _tbl(["Apex Traders"], ["1 mg road"], m), _tbl(["Apex Traders", "Apex Foods"], ["x", "y"], m)
    assert pairs(s1b, poolb, c)[0] == pairs(s1b, poolb, c2)[0]


def test_xlit_features():
    m = {DEV_SHARMA: "sharma", DEV_MEDICAL: "medical"}
    s1 = _tbl(["Sharma Medical"], ["12 station road"], m)
    pool = _tbl([f"{DEV_SHARMA} {DEV_MEDICAL}", "Gupta Store"], ["12 station road", "12 station road"], m)
    f = pair_features(s1, pool, np.array([0, 0]), np.array([0, 1]), 1, {"extra": ["xlit_core_jaccard", "xlit_token_set"]})
    assert f["xlit_core_jaccard"].tolist() == [1.0, 0.0] and f["xlit_token_set"][0] == pytest.approx(1.0)
    assert f["name_core_jaccard"][0] == 0.0  # the original view is unchanged
    base = pair_features(s1, pool, np.array([0]), np.array([0]), 1, {"extra": []})
    assert "xlit_core_jaccard" not in base


# ----------------------------------------------------------------------------- synthetic end-to-end + leakage
@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("n02_repo")
    info = make_synthetic_repo(root, n_train=260, n_test=90)
    c = yaml.safe_load(Path(info["config"]).read_text(encoding="utf-8"))
    c["experiment"]["id"] = "N02-SYN"
    c["matcher"]["type"] = "lightgbm"
    c["matcher"]["features"] = c["matcher"]["features"] + ["num_conflict", "legal_form_conflict", "xlit_core_jaccard", "xlit_token_set"]
    c["features"] = {"strip_leading_zeros": True, "extra": ["num_conflict", "legal_form_conflict", "xlit_core_jaccard", "xlit_token_set"]}
    c["decision"] = {"threshold_from": "model", "threshold": 0.5, "competition": {"enabled": True, "rule": "best_claimant", "margin": 0.0}}
    c["fit"].update({"max_train_s1": None, "inner_tune_fraction": 0.3})
    c["lightgbm"] = {"num_boost_round": 50, "early_stopping_rounds": 10, "params": {"num_leaves": 7, "min_data_in_leaf": 5, "num_threads": 1}}
    c["scoring"]["tag"] = "scored_n02"
    c["xlit"] = {"enabled": True, "min_support": 2, "min_confidence": 0.5, "folds": 5}
    c["blocking"]["name_view"] = "name_xlit"
    c["blocking"]["passes"]["sorted_core"] = {"enabled": True, "max_block_size": 50, "weight": 1.0}
    c["blocking"]["passes"]["name_addr"] = {"enabled": True, "max_name_tokens": 2, "max_words": 3, "word_max_df": 500000, "max_block_size": 50, "weight": 1.0}
    p = root / "n02.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    for st in ("preprocess", "split", "block-eval", "fit", "train", "infer", "submit"):
        assert main([st, "--config", str(p)]) == 0, st
    info["cfg"] = p
    return info


def test_leakage_validation_ids_forbidden(repo):
    c = load_config(repo["cfg"])
    paths = Paths(c)
    val = load_val_ids(paths)
    with pytest.raises(LeakageError):
        learn_map_for(c, paths, fit_s1_subset(c, paths)[:20] + val[:1], "bad")


def test_final_map_learned_from_fit_set_only_and_nonempty(repo):
    c = load_config(repo["cfg"])
    paths = Paths(c)
    m = XlitMap.load(paths.run_dir("N02-SYN") / "xlit_map.json")
    assert m.meta["learned_from_digest"] == XlitMap.ids_digest(fit_s1_subset(c, paths))
    assert m.meta["learned_from_digest"] != XlitMap.ids_digest(fit_s1_subset(c, paths) + load_val_ids(paths))
    assert len(m.mapping) >= 1 and all(token_script(k) == "nonlatin" and token_script(v) == "latin" for k, v in m.mapping.items())


def test_crossfit_plan_excludes_own_fold_and_inner_tune(repo):
    c = load_config(repo["cfg"])
    paths = Paths(c)
    fit = fit_s1_subset(c, paths)
    fold = dict(zip(fit, fold_of(c, fit)))
    from ber.pipeline.run_baseline import _hash_frac

    tune = {s for s in fit if _hash_frac(c["seed"], s) < float(c["fit"]["inner_tune_fraction"])}
    assert {s for s, f in fold.items() if f == 0} == tune  # fold 0 is exactly the inner-tune split
    for f, scored, learn in crossfit_plan(c, fit):
        assert not set(scored) & set(learn) and not set(learn) & tune
    rep = read_json(paths.run_dir("N02-SYN") / "fit_report.json")
    folds = rep["scoring_stats"]["crossfit_folds"]
    assert len(folds) == 5 and sum(x["scored_s1"] for x in folds) == len(fit)
    assert rep["model"]["meta"]["xlit"]["final_map_learned_from_digest"] == XlitMap.ids_digest(fit)


def test_training_rows_are_fit_s1_only(repo):
    c = load_config(repo["cfg"])
    paths = Paths(c)
    t = pa.concat_tables([pq.read_table(f, columns=["s1_row", "xlit_core_jaccard"]) for f in paths.scored_dir("train", "scored_n02_fit").glob("*.parquet")])
    ids = set(_ids(paths, "train", "source1").take(pa.array(t.column("s1_row").to_numpy().astype(np.int64))).to_pylist())
    assert ids <= set(fit_s1_subset(c, paths)) and not ids & set(load_val_ids(paths))
    assert {p.name.split("-")[0] for p in paths.scored_dir("train", "scored_n02_fit").glob("*.parquet")} <= {f"fold{i}" for i in range(5)}


def test_e2e_outputs_and_d01_invariants(repo):
    paths = Paths(load_config(repo["cfg"]))
    s = read_json(paths.run_dir("N02-SYN") / "submission_summary.json")
    assert s["strict_validation_errors"] == [] and s["competition"]["after"]["conflicted_records"] == 0

    def read(p):
        with open(p, encoding="utf-8", newline="") as f:
            r = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
        return {x[0]: [i for i in x[1].split(",") if i] for x in r[1:]}

    m, c = read(paths.output / "matching_results.tsv"), read(paths.output / "candidate_pairs.tsv")
    assert list(m) == list(c)
    seen = set()
    for k, ids in m.items():
        assert set(ids) <= set(c[k]) and not set(ids) & seen
        seen |= set(ids)
    raw = (paths.output / "matching_results.tsv").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw
    model = read_json(paths.run_dir("N02-SYN") / "model.json")
    assert "xlit_core_jaccard" in model["features"] and model["meta"]["blocking"]["name_view"] == "name_xlit"
