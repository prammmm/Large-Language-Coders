"""EXP-S-03: S-02-style veto REFITTED on EXP-B-09 stage-1 outputs. Written before the S-03 config existed.

A subset of stage-1 decisions  B only removals  C no candidate outside stage 1  D D-01 unchanged  E partitions
F threshold from the inner split only  G fitted on B-09 (not B-06) outputs  H candidates byte-identical to stage 1
I stage-1 artefacts untouched  J empty / singleton rows  K deterministic  L independent paths  M S-02 regression (existing tests)."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from ber.config import Paths, load_config
from ber.pipeline import run_stage2
from ber.utils.runtime import read_json

from test_stage2_pipeline import repo  # noqa: F401  (module fixture: synthetic stage 1 + stage 2)


# ----------------------------------------------------------------------------- real config (G, D, L, stage-2 recipe unchanged)
def test_G_D_L_s03_config_points_at_b09_and_keeps_the_s01_recipe():
    s01, s03, b09 = load_config("configs/exp_s01_stage2.yaml"), load_config("configs/exp_s03_veto_on_b09.yaml"), load_config("configs/exp_b09_name_num.yaml")
    c = s03["stage2"]
    assert c["stage1_run"] == "EXP-B-09" and c["train_tag"] == "forensic_alltrain_B09" and c["xlit_map"] == "experiments/runs/EXP-B-09/xlit_map.json"
    assert s03["scoring"]["tag"] == "scored_EXP-B-09" and s03["matcher"]["model_path"] == "experiments/runs/EXP-B-09/model.json"
    assert "B-06" not in json.dumps(c) and "B06" not in json.dumps(c)  # G: never the B-06 stage-1 outputs
    assert run_stage2.add_floor(s03) == float("inf")  # veto only
    assert not c.get("model_from") and not c.get("features_from")  # refitted, own features
    for k in ("min_score", "peers", "inner_fraction", "thresholds", "lightgbm", "chunk_pairs", "exclude_features", "train_countries"):
        assert c.get(k) == s01["stage2"].get(k), k  # the S-01/S-02 stage-2 recipe is unchanged
    assert s03["blocking"] == b09["blocking"] and s03["decision"] == b09["decision"]  # D: D-01 / decision unchanged
    assert s03["fit"] == b09["fit"] and s03["seed"] == 42
    assert s03["experiment"]["id"] == "EXP-S-03" and "EXP-S-03" in s03["paths"]["output_dir"]  # L
    assert run_stage2.stage1_threshold(s03) == 0.65  # B-09's inner-split stage-1 threshold


# ----------------------------------------------------------------------------- synthetic refit + veto (A, B, C, E, F, H, I, J, K)
def _tsv(p):
    with open(p, encoding="utf-8", newline="") as f:
        r = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    return {a[0]: [x for x in a[1].split(",") if x] for a in r[1:]}


@pytest.fixture(scope="module")
def s03(repo, tmp_path_factory):  # noqa: F811
    root = tmp_path_factory.mktemp("s03")
    p1 = Paths(load_config(repo["p1"]))
    mdir = p1.run_dir("SYN-S1")
    before = {f: (mdir / f).read_bytes() for f in ("model.json",)}
    y = yaml.safe_load(Path(repo["p2"]).read_text(encoding="utf-8"))
    y["experiment"]["id"] = "SYN-S03"
    y["paths"]["output_dir"] = "out_s03"
    y["stage2"]["add_floor"] = float("inf")  # refit (no model_from) + veto only
    p = root / "s03.yaml"
    p.write_text(yaml.safe_dump(y, sort_keys=False), encoding="utf-8")
    for st in ("features", "fit", "evaluate", "submit"):
        assert run_stage2.main([st, "--config", str(p)]) == 0, st
    return {"p": p, "before": before, "mdir": mdir}


def test_A_B_C_H_subset_no_additions_candidates_identical(repo, s03):  # noqa: F811
    ctl = _tsv(Paths(load_config(repo["p1"])).output / "matching_results.tsv")
    out = Paths(load_config(s03["p"])).output
    v, c = _tsv(out / "matching_results.tsv"), _tsv(out / "candidate_pairs.tsv")
    assert list(v) == list(ctl)
    for k in v:
        assert set(v[k]) <= set(ctl[k])  # A/B: only removals
        assert set(v[k]) <= set(c[k])  # C
    assert (out / "candidate_pairs.tsv").read_bytes() == (Paths(load_config(repo["p1"])).output / "candidate_pairs.tsv").read_bytes()  # H
    s = read_json(Paths(load_config(s03["p"])).run_dir("SYN-S03") / "submission_summary.json")
    assert s["vs_control_test"]["additions"] == 0 and s["strict_validation_errors"] == []
    ev = read_json(Paths(load_config(s03["p"])).run_dir("SYN-S03") / "stage2_eval.json")
    assert ev["flips_E_half_B"]["added_pairs"] == 0 and all(r["vs_control"]["additions_per_s1"] == 0 for r in ev["results"].values())


def test_E_F_partitions_and_inner_threshold(s03):
    cfg = load_config(s03["p"])
    paths = Paths(cfg)
    P, _ = run_stage2.populations(cfg, paths)
    tr = P["stage2_train"] | P["stage2_inner"]
    for bad in ("validation", "E_half_B", "fit"):
        assert not (tr & P[bad]).any()
    meta = read_json(paths.run_dir("SYN-S03") / "stage2_model.json")
    # F: the chosen rule is the best of the INNER table (the only place the threshold is selected)
    best = max(meta["inner_table"].items(), key=lambda kv: (round(kv[1]["f05"], 6), kv[0].startswith("threshold+D01"), -abs(float(kv[0].split("@")[1]) - 0.5)))
    assert best[0] == f"{meta['variant']}@{meta['threshold']:.2f}"
    assert meta["train_s1"] == int(P["stage2_train"].sum()) and meta["inner_s1"] == int(P["stage2_inner"].sum())


def test_I_stage1_untouched_J_empty_rows_K_deterministic_L_paths(repo, s03):  # noqa: F811
    for f, b in s03["before"].items():
        assert (s03["mdir"] / f).read_bytes() == b  # I
    out = Paths(load_config(s03["p"])).output
    v, c = _tsv(out / "matching_results.tsv"), _tsv(out / "candidate_pairs.tsv")
    assert all(v[k] == [] for k in c if not c[k])  # J: no candidates -> empty
    first = {f: (out / f).read_bytes() for f in ("matching_results.tsv", "candidate_pairs.tsv")}
    assert run_stage2.main(["submit", "--config", str(s03["p"])]) == 0
    assert all((out / f).read_bytes() == b for f, b in first.items())  # K
    run_dir = Paths(load_config(s03["p"])).run_dir("SYN-S03")
    assert (run_dir / "stage2_model.lgb.txt").exists() and run_dir != Paths(load_config(repo["p2"])).run_dir("SYN-S2")  # L: own model


def test_M_s02_regression_suite_present():
    # the S-02 veto behaviour on its original control is covered by tests/test_stage2_pipeline.py
    # (veto = intersection with forced additions/removals, add_floor=null byte-identical, streamed control == reference)
    src = Path("tests/test_stage2_pipeline.py").read_text(encoding="utf-8")
    for name in ("test_veto_output_equals_intersection_of_control_and_stage2", "test_veto_intersection_when_stage2_really_adds_and_removes",
                 "test_add_floor_null_reproduces_stage2_output_bytes", "test_streamed_control_equals_all_pairs_reference"):
        assert f"def {name}" in src
