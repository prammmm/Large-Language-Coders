"""EXP-B-08: top-K 20 -> 30. Blocker superset / ordering / ties, config isolation from EXP-B-06, and a synthetic
end-to-end run with a small and a larger K (retrained model, own artefacts, no leakage, deterministic output)."""

import csv
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from ber.blocking.blocker import CountryBlocker
from ber.config import Paths, load_config
from ber.data.split import load_val_ids
from ber.models import model_path
from ber.pipeline.run_baseline import fit_s1_subset, main
from ber.utils.runtime import read_json

from synthetic import make_synthetic_repo
from test_blocking_v2 import cfg as bcfg, table


def cands(s1, pool, top_k):
    c = bcfg(name_addr=True, addr_pair=True)
    c["passes"]["exact_core"]["enabled"] = True
    c["passes"]["name_token"] = {"enabled": True, "max_df": 50, "max_block_size": 50, "weight": 1.0}
    c["top_k"] = top_k
    b = CountryBlocker(s1, pool, c)
    out = {k: [] for k in ("s1_local", "pool_local", "cheap_score", "cheap_rank", "pass_flags")}
    for lo, hi in b.chunk_bounds():
        cc = b.candidates(lo, hi)
        for k in out:
            out[k] += cc[k].tolist()
    return {k: np.asarray(v) for k, v in out.items()}


def rows(n_pool=40):
    rng = np.random.default_rng(42)
    words = ["harbor", "lotus", "clinic", "maple", "cedar", "store"]
    s1 = table(["Harbor Lotus Clinic", "Cedar Store", "Zzqx Unmatched", ""], ["Maple Avenue 12", "Main Road 3", "Nowhere 9", ""])
    names = [" ".join(rng.choice(words, 2, replace=False)).title() for _ in range(n_pool)]
    addrs = [f"Maple Avenue {int(rng.integers(1, 20))}" for _ in range(n_pool)]
    return s1, table(names, addrs)


def test_larger_k_is_a_superset_with_identical_top_ranks():
    s1, pool = rows()
    a, b = cands(s1, pool, 5), cands(s1, pool, 12)
    ka = set(zip(a["s1_local"].tolist(), a["pool_local"].tolist()))
    kb = set(zip(b["s1_local"].tolist(), b["pool_local"].tolist()))
    assert ka < kb  # strict superset: the fixture really truncates at K 5
    assert a["cheap_rank"].max() == 4 and b["cheap_rank"].max() >= 5
    top = b["cheap_rank"] < 5  # the first K ranks of the larger list are exactly the smaller list, same order
    for key in a:
        np.testing.assert_array_equal(a[key], b[key][top])


def test_no_duplicates_valid_ids_ties_and_empty_s1():
    s1, pool = rows()
    b = cands(s1, pool, 30)
    codes = b["s1_local"] * 1000 + b["pool_local"]
    assert len(np.unique(codes)) == len(codes)  # no duplicate (S1, record)
    assert b["pool_local"].min() >= 0 and b["pool_local"].max() < pool.num_rows and b["s1_local"].max() < s1.num_rows
    for s in np.unique(b["s1_local"]):
        m = b["s1_local"] == s
        sc, pl, rk = b["cheap_score"][m], b["pool_local"][m], b["cheap_rank"][m]
        assert (rk == np.arange(m.sum())).all()  # ranks 0..n-1 in stored order
        assert (np.diff(sc) <= 0).all()  # score descending
        tie = np.diff(sc) == 0
        assert (np.diff(pl)[tie] > 0).all()  # ties -> lower pool index first
    assert 3 not in set(b["s1_local"].tolist())  # empty name + address: no candidates, no crash
    again = cands(s1, pool, 30)
    for k in b:
        np.testing.assert_array_equal(b[k], again[k])  # deterministic


def test_real_b08_config_differs_from_b06_only_where_intended():
    b06, b08 = load_config("configs/exp_b06_name_pair.yaml"), load_config("configs/exp_b08_topk30.yaml")
    assert b08["blocking"]["top_k"] == 30 and b06["blocking"]["top_k"] == 20
    assert b08["blocking"]["passes"] == b06["blocking"]["passes"] and b08["matcher"]["features"] == b06["matcher"]["features"]
    assert {k: v for k, v in b08["blocking"].items() if k != "top_k"} == {k: v for k, v in b06["blocking"].items() if k != "top_k"}
    for sec in ("features", "decision", "fit", "xlit", "lightgbm", "split", "seed"):
        assert b08.get(sec) == b06.get(sec), sec
    p = model_path(b08, Paths(b08))
    assert p.parts[-2] == "EXP-B-08" and "EXP-B-06" not in str(p)  # never the B-06 model
    assert b08["scoring"]["tag"] == "scored_EXP-B-08" and b08["scoring"]["eval_source_tag"] is None
    assert "EXP-B-08" in b08["paths"]["output_dir"] and b08["paths"]["output_dir"] != "output"


@pytest.fixture(scope="module")
def two_k(tmp_path_factory):
    root = tmp_path_factory.mktemp("topk_repo")
    info = make_synthetic_repo(root, n_train=260, n_test=90)
    base = yaml.safe_load(Path(info["config"]).read_text(encoding="utf-8"))
    out = {}
    for tag, k in (("K4", 4), ("K8", 8)):
        c = yaml.safe_load(yaml.safe_dump(base))
        c["experiment"]["id"] = f"SYN-{tag}"
        c["matcher"].update({"type": "lightgbm", "model_path": None})
        c["decision"] = {"threshold_from": "model", "threshold": 0.5, "competition": {"enabled": True, "rule": "best_claimant", "margin": 0.0}}
        c["fit"].update({"max_train_s1": None, "inner_tune_fraction": 0.3})
        c["lightgbm"] = {"num_boost_round": 40, "early_stopping_rounds": 10, "params": {"num_leaves": 7, "min_data_in_leaf": 5, "num_threads": 1}}
        c["scoring"]["tag"] = f"scored_{tag}"
        c["paths"]["output_dir"] = f"out_{tag}"
        c["blocking"]["top_k"] = k
        p = root / f"{tag}.yaml"
        p.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
        stages = ("preprocess", "split") if not out else ()  # the split is frozen after the first run (guard)
        for st in stages + ("fit", "train", "infer", "submit"):
            assert main([st, "--config", str(p)]) == 0, (tag, st)
        out[tag] = p
    return out


def _tsv(p):
    with open(p, encoding="utf-8", newline="") as f:
        r = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    return {a[0]: [x for x in a[1].split(",") if x] for a in r[1:]}


def test_larger_k_retrains_on_its_own_candidates(two_k):
    cfgs = {t: load_config(p) for t, p in two_k.items()}
    paths = {t: Paths(c) for t, c in cfgs.items()}
    m4, m8 = (read_json(model_path(cfgs[t], paths[t])) for t in ("K4", "K8"))
    assert m8["meta"]["blocking"]["top_k"] == 8 and m4["meta"]["blocking"]["top_k"] == 4  # each model knows its blocker
    assert model_path(cfgs["K8"], paths["K8"]).parent.name == "SYN-K8"
    ranks = pa.concat_tables([pq.read_table(p, columns=["cheap_rank"]) for p in paths["K8"].scored_dir("train", "scored_K8_fit").glob("*.parquet")])
    assert ranks.column("cheap_rank").to_numpy().max() >= 4  # fit candidates include the new ranks
    assert model_path(cfgs["K8"], paths["K8"]).read_bytes() != model_path(cfgs["K4"], paths["K4"]).read_bytes()


def test_no_validation_s1_in_fit(two_k):
    c = load_config(two_k["K8"])
    paths = Paths(c)
    val = set(load_val_ids(paths))
    assert not set(fit_s1_subset(c, paths)) & val
    t = pa.concat_tables([pq.read_table(p, columns=["s1_row"]) for p in paths.scored_dir("train", "scored_K8_fit").glob("*.parquet")])
    ids = pq.read_table(paths.interim_file("train", "source1"), columns=["entity_id"]).column("entity_id").to_pylist()
    assert not {ids[i] for i in np.unique(t.column("s1_row").to_numpy())} & val


def test_outputs_superset_candidates_valid_and_deterministic(two_k):
    o4, o8 = Paths(load_config(two_k["K4"])).output, Paths(load_config(two_k["K8"])).output
    c4, c8 = _tsv(o4 / "candidate_pairs.tsv"), _tsv(o8 / "candidate_pairs.tsv")
    m8 = _tsv(o8 / "matching_results.tsv")
    assert list(c4) == list(c8)
    assert sum(map(len, c8.values())) > sum(map(len, c4.values()))
    for k in c8:
        assert c8[k][: len(c4[k])] == c4[k]  # the K4 list is the prefix of the K8 list (candidate order = rank)
        assert len(c8[k]) == len(set(c8[k])) and set(m8[k]) <= set(c8[k])
    assert read_json(Paths(load_config(two_k["K8"])).run_dir("SYN-K8") / "submission_summary.json")["strict_validation_errors"] == []
    before = {f: (o8 / f).read_bytes() for f in ("candidate_pairs.tsv", "matching_results.tsv")}
    assert main(["submit", "--config", str(two_k["K8"])]) == 0
    assert all((o8 / f).read_bytes() == b for f, b in before.items())
