"""EXP-S-01: synthetic end-to-end of the stage-2 pipeline + leakage, chunking and determinism tests."""

import csv
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from ber.config import Paths, load_config
from ber.models import make_scorer
from ber.pipeline import run_stage2
from ber.pipeline.run_baseline import main as stage1_main, score_split
from ber.utils.runtime import read_json

from synthetic import make_synthetic_repo


def _read_tsv(p):
    with open(p, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    return {r[0]: [x for x in r[1].split(",") if x] for r in rows[1:]}


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("s2_repo")
    info = make_synthetic_repo(root, n_train=320, n_test=90)
    c = yaml.safe_load(Path(info["config"]).read_text(encoding="utf-8"))
    c["experiment"]["id"] = "SYN-S1"
    c["matcher"]["type"] = "lightgbm"
    c["decision"] = {"threshold_from": "model", "threshold": 0.5, "competition": {"enabled": True, "rule": "best_claimant", "margin": 0.0}}
    c["fit"].update({"max_train_s1": 90, "inner_tune_fraction": 0.3})  # leaves non-fit, non-validation S1 for stage 2
    c["lightgbm"] = {"num_boost_round": 40, "early_stopping_rounds": 10, "params": {"num_leaves": 7, "min_data_in_leaf": 5, "num_threads": 1}}
    c["scoring"]["tag"] = "scored_s1"
    p1 = root / "s1.yaml"
    p1.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    for st in ("preprocess", "split", "fit", "train", "infer", "submit"):
        assert stage1_main([st, "--config", str(p1)]) == 0, st
    cfg1 = load_config(p1)
    paths = Paths(cfg1)
    score_split(cfg1, paths, "train", None, paths.scored_dir("train", "alltrain"), make_scorer(cfg1, paths), False)
    s2 = {"inherit": str(p1), "experiment": {"id": "SYN-S2"}, "paths": {"output_dir": "out_s2"},
          "matcher": {"type": "lightgbm", "model_path": str(paths.run_dir("SYN-S1") / "model.json")},
          "stage2": {"stage1_run": "SYN-S1", "train_tag": "alltrain", "xlit_map": None, "min_score": 0.02, "peers": 3,
                     "inner_fraction": 0.3, "chunk_pairs": 40, "workers": 1, "thresholds": {"start": 0.3, "stop": 0.9, "step": 0.05},
                     "lightgbm": {"num_boost_round": 60, "early_stopping_rounds": 10,
                                  "params": {"num_leaves": 7, "min_data_in_leaf": 5, "num_threads": 1}}}}
    p2 = root / "s2.yaml"
    p2.write_text(yaml.safe_dump(s2, sort_keys=False), encoding="utf-8")
    for st in ("features", "fit", "evaluate", "submit"):
        assert run_stage2.main([st, "--config", str(p2)]) == 0, st
    info.update({"p1": p1, "p2": p2})
    return info


def test_training_partition_has_no_leakage(repo):
    cfg = load_config(repo["p2"])
    paths = Paths(cfg)
    P, n = run_stage2.populations(cfg, paths)
    tr = P["stage2_train"] | P["stage2_inner"]
    assert tr.any() and P["E_half_B"].any() and P["validation"].any()
    for bad in ("validation", "E_half_B", "fit"):
        assert not (tr & P[bad]).any()
    meta = read_json(paths.run_dir("SYN-S2") / "stage2_model.json")
    assert meta["train_s1"] == int(P["stage2_train"].sum()) and meta["inner_s1"] == int(P["stage2_inner"].sum())
    assert meta["variant"] in ("threshold", "threshold+D01") and 0.3 <= meta["threshold"] <= 0.9


def test_features_cover_all_eligible_pairs_and_chunking_is_exact(repo):
    cfg = load_config(repo["p2"])
    paths = Paths(cfg)
    stats = read_json(run_stage2.feat_dir(cfg, paths, "train").parent / f"stage2_SYN-S2_stats.json")
    assert stats["parts"] > 1  # chunk_pairs 40 forces several chunks
    t = run_stage2.read_features(cfg, paths, "train")
    d = pq.read_table(next(iter(sorted(paths.scored_dir("train", "alltrain").glob("*.parquet"))))).schema  # noqa: F841
    assert t.num_rows == stats["eligible_pairs"] and t.column("score").to_numpy().min() >= 0.02
    ref = t.to_pydict()
    # the same features with one single chunk: identical (no S1 split across chunks, peers stay inside S1)
    big = run_stage2.feat_dir(cfg, paths, "train")
    cfg2 = dict(cfg, stage2=dict(cfg["stage2"], chunk_pairs=10**9))
    run_stage2.build_features(cfg2, paths, "train")
    t2 = run_stage2.read_features(cfg2, paths, "train").to_pydict()
    for k in ref:
        np.testing.assert_array_equal(np.asarray(ref[k], np.float64), np.asarray(t2[k], np.float64))
    run_stage2.build_features(cfg, paths, "train")  # restore the chunked parts
    assert big.exists()


def test_submission_valid_and_matches_subset_of_unchanged_candidates(repo):
    p1, p2 = Paths(load_config(repo["p1"])), Paths(load_config(repo["p2"]))
    s = read_json(p2.run_dir("SYN-S2") / "submission_summary.json")
    assert s["strict_validation_errors"] == [] and s["competition_after"]["predicted_pairs"] >= 0
    assert (p1.output / "candidate_pairs.tsv").read_bytes() == (p2.output / "candidate_pairs.tsv").read_bytes()  # candidates unchanged
    m, c = _read_tsv(p2.output / "matching_results.tsv"), _read_tsv(p2.output / "candidate_pairs.tsv")
    assert list(m) == list(c)
    for k, ids in m.items():
        assert set(ids) <= set(c[k]) and len(ids) == len(set(ids))
        assert all(x.startswith(("S2-", "S3-")) for x in ids)
    raw = (p2.output / "matching_results.tsv").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw


def test_evaluation_report_structure(repo):
    p2 = Paths(load_config(repo["p2"]))
    ev = read_json(p2.run_dir("SYN-S2") / "stage2_eval.json")
    for pop in ("E_half_B", "validation", "validation_half_A", "validation_half_B"):
        assert "control" in ev["results"][pop] and "stage2" in ev["results"][pop]
    assert set(ev["flips_E_half_B"]) == {"added_pairs", "added_true", "removed_pairs", "removed_true"}


def test_deterministic_refit_and_submit(repo):
    p2 = Paths(load_config(repo["p2"]))
    model1 = (p2.run_dir("SYN-S2") / "stage2_model.lgb.txt").read_bytes()
    out1 = (p2.output / "matching_results.tsv").read_bytes()
    assert run_stage2.main(["fit", "--config", str(repo["p2"])]) == 0
    assert run_stage2.main(["submit", "--config", str(repo["p2"])]) == 0
    assert (p2.run_dir("SYN-S2") / "stage2_model.lgb.txt").read_bytes() == model1
    assert (p2.output / "matching_results.tsv").read_bytes() == out1


def test_exclude_features_and_shared_feature_parts(repo, tmp_path):
    cfg = load_config(repo["p2"])
    c = dict(cfg["stage2"], exclude_features=["s1_n_cand", "s1_rel_pos", "peer_n"])
    names = run_stage2.model_features(c)
    assert "s1_n_cand" not in names and "score" in names and len(names) == len(run_stage2.feature_names(3)) - 3
    with pytest.raises(ValueError):
        run_stage2.model_features(dict(c, exclude_features=["nope"]))
    s = yaml.safe_load(Path(repo["p2"]).read_text(encoding="utf-8"))
    s["experiment"]["id"] = "SYN-S2b"
    s["paths"]["output_dir"] = "out_s2b"
    s["stage2"].update({"features_from": "SYN-S2", "exclude_features": ["s1_n_cand", "s1_rel_pos", "peer_n"]})
    p = tmp_path / "s2b.yaml"
    p.write_text(yaml.safe_dump(s, sort_keys=False), encoding="utf-8")
    for st in ("fit", "evaluate", "submit"):
        assert run_stage2.main([st, "--config", str(p)]) == 0, st
    paths = Paths(load_config(p))
    meta = read_json(paths.run_dir("SYN-S2b") / "stage2_model.json")
    assert "s1_n_cand" not in meta["features"] and read_json(paths.run_dir("SYN-S2b") / "submission_summary.json")["strict_validation_errors"] == []


def test_train_countries_restricts_stage2_training_partition(repo):
    cfg = load_config(repo["p2"])
    paths = Paths(cfg)
    P_all, _ = run_stage2.populations(cfg, paths)
    cfg_us = dict(cfg, stage2=dict(cfg["stage2"], train_countries=["us"]))
    P_us, _ = run_stage2.populations(cfg_us, paths)
    t = pq.read_table(paths.interim_file("train", "source1"), columns=["country"]).column("country").to_pylist()
    from ber.data.normalize_records import country_key
    is_us = np.array([country_key(x) == "us" for x in t])
    tr = P_us["stage2_train"] | P_us["stage2_inner"]
    assert tr.any() and not (tr & ~is_us).any()
    assert (tr <= (P_all["stage2_train"] | P_all["stage2_inner"])).all()  # a subset, still leakage-free
    np.testing.assert_array_equal(P_us["E_half_B"], P_all["E_half_B"])  # evaluation populations unchanged


def test_drop_mask_removes_competitors_and_keeps_s1_level_features(repo, tmp_path):
    cfg = load_config(repo["p2"])
    paths = Paths(cfg)
    full = run_stage2.read_features(cfg, paths, "train").to_pydict()
    drop = np.zeros(int(max(full["s1_row"])) + 2, bool)
    drop[np.unique(full["s1_row"])[::3]] = True
    st = run_stage2.build_features(cfg, paths, "train", drop_s1=drop, out=tmp_path / "dropped")
    t = pa.concat_tables([pq.read_table(p) for p in sorted((tmp_path / "dropped").glob("*.parquet"))]).to_pydict()
    assert st["dropped_s1"] == int(drop.sum()) and not np.isin(t["s1_row"], np.flatnonzero(drop)).any()
    pos = {(a, b): i for i, (a, b) in enumerate(zip(full["s1_row"], full["pool_row"]))}
    idx = [pos[(a, b)] for a, b in zip(t["s1_row"], t["pool_row"])]
    assert len(idx) == len(t["s1_row"])  # kept pairs are a subset of the full competition's pairs
    for f in ("score", "s1_rank", "s1_best", "p1_name_jacc", "peer_max_sim"):  # S1-level: independent of other S1
        np.testing.assert_array_equal(np.asarray(t[f], np.float64), np.asarray(full[f], np.float64)[idx])
    assert (np.asarray(t["rec_n_claim"]) <= np.asarray(full["rec_n_claim"])[idx]).all()  # fewer competitors
    assert read_json(run_stage2.feat_dir(cfg, paths, "train").parent / "stage2_SYN-S2_stats.json")["dropped_s1"] == 0


# ----------------------------------------------------------------------------- EXP-S-02: veto-only stage 2
def _variant(repo, root, exp_id, out_dir, **stage2):
    s = {"inherit": str(repo["p2"]), "experiment": {"id": exp_id}, "paths": {"output_dir": out_dir},
         "stage2": {"features_from": "SYN-S2", "model_from": "SYN-S2", **stage2}}
    p = root / f"{exp_id}.yaml"
    p.write_text(yaml.safe_dump(s, sort_keys=False), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def veto(repo, tmp_path_factory):
    root = tmp_path_factory.mktemp("s2_veto")
    p2 = Paths(load_config(repo["p2"]))
    mdir = p2.run_dir("SYN-S2")
    before = {f: ((mdir / f).read_bytes(), (mdir / f).stat().st_mtime_ns) for f in ("stage2_model.lgb.txt", "stage2_model.json")}
    pv = _variant(repo, root, "SYN-S3", "out_s3", add_floor=float("inf"))
    assert "add_floor: .inf" in pv.read_text(encoding="utf-8")  # the YAML spelling used by the real config
    for st in ("evaluate", "submit"):
        assert run_stage2.main([st, "--config", str(pv)]) == 0, st
    pn = _variant(repo, root, "SYN-S3n", "out_s3n", add_floor=None)
    assert run_stage2.main(["submit", "--config", str(pn)]) == 0
    return {"pv": pv, "pn": pn, "before": before, "mdir": mdir}


def _outputs(p):
    return Paths(load_config(p)).output


def test_veto_output_equals_intersection_of_control_and_stage2(repo, veto):
    ctl = _read_tsv(_outputs(repo["p1"]) / "matching_results.tsv")  # the stage-1 system with D-01 = the control
    s2 = _read_tsv(_outputs(repo["p2"]) / "matching_results.tsv")  # unrestricted stage 2 (the S-01 analogue)
    v = _read_tsv(_outputs(veto["pv"]) / "matching_results.tsv")
    assert list(v) == list(ctl) == list(s2)
    assert sum(map(len, ctl.values())) > 0 and sum(map(len, s2.values())) > 0
    for k in v:
        assert set(v[k]) <= set(ctl[k]) and set(v[k]) <= set(s2[k])
        assert set(v[k]) == set(ctl[k]) & set(s2[k])
    # (on this synthetic test split stage 2 happens to agree with the control; the non-trivial cases follow)


def _clone_model(repo, exp_id, threshold, variant):
    """Test-only: the SYN-S2 model with another decision rule, so stage 2 really adds / removes pairs (the
    synthetic test pool is saturated: with D-01 every record already has its control claimant)."""
    import json
    import shutil
    paths = Paths(load_config(repo["p2"]))
    src, dst = paths.run_dir("SYN-S2"), paths.run_dir(exp_id)
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "stage2_model.lgb.txt", dst / "stage2_model.lgb.txt")
    meta = json.loads((src / "stage2_model.json").read_text(encoding="utf-8"))
    (dst / "stage2_model.json").write_text(json.dumps(meta | {"threshold": threshold, "variant": variant}), encoding="utf-8")


@pytest.mark.parametrize("threshold,variant,direction", [(0.0, "threshold", "additions"), (0.98, "threshold+D01", "removals")])
def test_veto_intersection_when_stage2_really_adds_and_removes(repo, tmp_path, threshold, variant, direction):
    mid = f"SYN-S2t{int(threshold * 100)}"
    _clone_model(repo, mid, threshold, variant)
    runs = {}
    for name, floor in (("free", None), ("veto", float("inf"))):
        p = _variant(repo, tmp_path, f"{mid}-{name}", f"out_{mid}_{name}", add_floor=floor)
        y = yaml.safe_load(p.read_text(encoding="utf-8"))
        y["stage2"]["model_from"] = mid
        p.write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
        for st in ("evaluate", "submit"):
            assert run_stage2.main([st, "--config", str(p)]) == 0, st
        runs[name] = p
    ctl = _read_tsv(_outputs(repo["p1"]) / "matching_results.tsv")
    free = _read_tsv(_outputs(runs["free"]) / "matching_results.tsv")
    v = _read_tsv(_outputs(runs["veto"]) / "matching_results.tsv")
    add = sum(len(set(free[k]) - set(ctl[k])) for k in v)
    rem = sum(len(set(ctl[k]) - set(free[k])) for k in v)
    assert (add if direction == "additions" else rem) > 0  # the unrestricted stage 2 really differs from the control
    for k in v:
        assert set(v[k]) == set(ctl[k]) & set(free[k])
    s = read_json(Paths(load_config(runs["veto"])).run_dir(f"{mid}-veto") / "submission_summary.json")
    assert s["vs_control_test"]["additions"] == 0 and s["vs_control_test"]["removals"] == rem and s["strict_validation_errors"] == []
    ev = read_json(Paths(load_config(runs["veto"])).run_dir(f"{mid}-veto") / "stage2_eval.json")
    evf = read_json(Paths(load_config(runs["free"])).run_dir(f"{mid}-free") / "stage2_eval.json")
    assert ev["flips_E_half_B"]["added_pairs"] == 0
    assert ev["flips_E_half_B"]["removed_pairs"] == evf["flips_E_half_B"]["removed_pairs"]  # removals are exactly stage 2's
    assert ev["results"]["E_half_B"]["stage2_unrestricted"]["f05"] == evf["results"]["E_half_B"]["stage2"]["f05"]


def test_veto_reports_no_additions_and_structure(veto):
    paths = Paths(load_config(veto["pv"]))
    s = read_json(paths.run_dir("SYN-S3") / "submission_summary.json")
    assert s["strict_validation_errors"] == [] and s["vs_control_test"]["additions"] == 0
    assert s["vs_control_test"]["final_pairs"] == s["matching_file"]["ids"] == s["vs_control_test"]["control_pairs"] - s["vs_control_test"]["removals"]
    assert s["decision"]["add_floor"] == "inf" and s["competition_after"]["conflicted_records"] == 0
    ev = read_json(paths.run_dir("SYN-S3") / "stage2_eval.json")
    assert ev["flips_E_half_B"]["added_pairs"] == 0 and ev["decision"]["add_floor"] == "inf"
    for pop, r in ev["results"].items():
        assert r["vs_control"]["additions_per_s1"] == 0, pop
        assert r["stage2"]["macro_precision"] >= 0 and "stage2_unrestricted" in r and "empty_rate" in r["stage2"]


def test_add_floor_null_reproduces_stage2_output_bytes(repo, veto):
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        assert (_outputs(veto["pn"]) / f).read_bytes() == (_outputs(repo["p2"]) / f).read_bytes()


def test_model_from_is_read_only(veto):
    for f, (b, t) in veto["before"].items():
        assert (veto["mdir"] / f).read_bytes() == b and (veto["mdir"] / f).stat().st_mtime_ns == t  # untouched
    paths = Paths(load_config(veto["pv"]))
    assert not list(paths.run_dir("SYN-S3").glob("stage2_model*"))  # no copy / refit in the veto run
    with pytest.raises(RuntimeError, match="read-only"):
        run_stage2.main(["fit", "--config", str(veto["pv"])])
    with pytest.raises(RuntimeError, match="read-only"):
        run_stage2.main(["features", "--config", str(veto["pv"]), "--split", "train"])


def test_veto_candidates_unchanged_ids_inside_and_empty_rows(repo, veto):
    out = _outputs(veto["pv"])
    assert (out / "candidate_pairs.tsv").read_bytes() == (_outputs(repo["p1"]) / "candidate_pairs.tsv").read_bytes()
    m, c = _read_tsv(out / "matching_results.tsv"), _read_tsv(out / "candidate_pairs.tsv")
    ctl = _read_tsv(_outputs(repo["p1"]) / "matching_results.tsv")
    for k, ids in m.items():
        assert set(ids) <= set(c[k]) and len(ids) == len(set(ids)) and all(x.startswith(("S2-", "S3-")) for x in ids)
        if not c[k] or not ctl[k]:
            assert ids == []  # no candidates / singleton under the control -> stays empty
    assert any(not v for v in m.values())
    raw = (out / "matching_results.tsv").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf") and b"\r" not in raw


def test_veto_submit_is_deterministic(veto):
    out = _outputs(veto["pv"])
    first = {f: (out / f).read_bytes() for f in ("matching_results.tsv", "candidate_pairs.tsv")}
    assert run_stage2.main(["submit", "--config", str(veto["pv"])]) == 0
    for f, b in first.items():
        assert (out / f).read_bytes() == b


def test_add_floor_config_validation(repo):
    cfg = load_config(repo["p2"])
    for bad in ("inf", -1, True, [1]):
        with pytest.raises(ValueError):
            run_stage2.add_floor(dict(cfg, stage2=dict(cfg["stage2"], add_floor=bad)))
    assert run_stage2.add_floor(cfg) is None
    assert run_stage2.add_floor(dict(cfg, stage2=dict(cfg["stage2"], add_floor=yaml.safe_load(".inf")))) == float("inf")


def test_streamed_control_equals_all_pairs_reference(repo):
    from ber.decision.competition import best_claimant_keep
    from ber.pipeline.run_baseline import _read_parts
    cfg = load_config(repo["p2"])
    paths = Paths(cfg)
    for split in ("train", "test"):
        code, score, conflicts, n_cand = run_stage2.control_decisions(cfg, paths, split)
        d = _read_parts(run_stage2.stage1_parts(cfg, paths, split), ["s1_row", "pool_row", "score"])
        s1, pool, sc = d["s1_row"].astype(np.int64), d["pool_row"].astype(np.int64), d["score"].astype(np.float32)
        acc = best_claimant_keep(pool, s1, sc, sc >= np.float32(run_stage2.stage1_threshold(cfg)))  # the original, all-pairs computation
        ref = np.sort(s1[acc] * run_stage2.SHIFT + pool[acc])
        assert len(ref) > 0
        np.testing.assert_array_equal(code, ref)
        np.testing.assert_array_equal(n_cand, np.bincount(s1))
        np.testing.assert_array_equal(score, sc[acc][np.argsort(s1[acc] * run_stage2.SHIFT + pool[acc], kind="stable")])
        assert conflicts["conflicted_records"] == 0
