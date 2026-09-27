"""EXP-M-03: LightGBM matcher unit tests + synthetic end-to-end (fit -> train -> infer -> submit)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from ber.config import Paths, load_config
from ber.data.split import load_val_ids
from ber.models import make_scorer
from ber.models.gbdt import LightGBMModel
from ber.models.logistic import feature_matrix
from ber.pipeline.run_baseline import fit_s1_subset, main
from ber.utils.runtime import read_json

from synthetic import make_synthetic_repo

REPO = Path(__file__).resolve().parents[1]
OFFICIAL = REPO / "student_resource" / "utils" / "validate_submission.py"


def _xor_data(n=6000, seed=0):
    """A conditional (non-linear) rule: match iff name similar AND numbers agree."""
    rng = np.random.default_rng(seed)
    name = rng.random(n).astype(np.float32)
    conflict = (rng.random(n) < 0.5).astype(np.float32)
    y = (name > 0.6) & (conflict == 0)
    return {"name": name, "conflict": conflict}, y


def test_lightgbm_import_and_learns_conditional_rule(tmp_path):
    import lightgbm

    assert lightgbm.__version__.startswith("4.")
    feats, y = _xor_data()
    X = feature_matrix(feats, ["name", "conflict"])
    m = LightGBMModel(["name", "conflict"], {"num_threads": 1}).fit(X[:5000], y[:5000], X[5000:], y[5000:], num_boost_round=200)
    p = m.score({k: v[5000:] for k, v in feats.items()})
    assert p.shape == (1000,) and p.dtype == np.float32 and (p >= 0).all() and (p <= 1).all()
    acc = ((p >= 0.5) == y[5000:]).mean()
    assert acc > 0.97
    # the rule needs an interaction: high name similarity alone must not be enough
    assert m.score({"name": np.array([0.95], np.float32), "conflict": np.array([1.0], np.float32)})[0] < 0.2
    assert m.score({"name": np.array([0.95], np.float32), "conflict": np.array([0.0], np.float32)})[0] > 0.8
    m.meta["threshold"] = 0.5
    m.save(tmp_path / "model.json")
    m2 = LightGBMModel.load(tmp_path / "model.json")
    assert np.allclose(m.score(feats), m2.score(feats), atol=1e-6)
    assert m2.meta["threshold"] == 0.5 and set(m2.meta["feature_importance"]) == {"name", "conflict"}
    assert (tmp_path / "model.lgb.txt").exists()


def test_lightgbm_deterministic_with_seed():
    feats, y = _xor_data()
    X = feature_matrix(feats, ["name", "conflict"])
    a = LightGBMModel(["name", "conflict"], {"num_threads": 1}).fit(X, y, num_boost_round=50).predict_proba_matrix(X)
    b = LightGBMModel(["name", "conflict"], {"num_threads": 1}).fit(X, y, num_boost_round=50).predict_proba_matrix(X)
    assert np.array_equal(a, b)


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("lgbm_repo")
    info = make_synthetic_repo(root, n_train=260, n_test=90)
    cfg = yaml.safe_load(Path(info["config"]).read_text(encoding="utf-8"))
    cfg["experiment"]["id"] = "EXP-TEST-LGBM"
    cfg["matcher"]["type"] = "lightgbm"
    cfg["matcher"]["features"] = cfg["matcher"]["features"] + ["num_conflict", "legal_form_conflict"]
    cfg["features"] = {"strip_leading_zeros": True, "extra": ["num_conflict", "legal_form_conflict"]}
    cfg["decision"] = {"threshold_from": "model", "threshold": 0.5}
    cfg["fit"].update({"max_train_s1": None, "inner_tune_fraction": 0.3})
    cfg["lightgbm"] = {"num_boost_round": 100, "early_stopping_rounds": 20,
                       "params": {"num_leaves": 7, "min_data_in_leaf": 5, "num_threads": 1, "seed": 42}}
    cfg["scoring"]["tag"] = "scored_lgbm"
    cfg_path = root / "config_lgbm.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    for st in ("preprocess", "split", "fit", "train", "infer", "submit"):
        assert main([st, "--config", str(cfg_path)]) == 0, st
    # second experiment re-using the first one's fit parts read-only (the EXP-M-03 mechanism)
    cfg2 = dict(cfg)
    cfg2["experiment"] = {"id": "EXP-TEST-LGBM-REUSE"}
    cfg2["fit"] = {**cfg["fit"], "reuse_fit_parts": "scored_lgbm_fit"}
    cfg2["scoring"] = {**cfg["scoring"], "tag": "scored_lgbm2", "eval_source_tag": "scored_lgbm"}
    cfg2_path = root / "config_lgbm2.yaml"
    cfg2_path.write_text(yaml.safe_dump(cfg2, sort_keys=False), encoding="utf-8")
    for st in ("fit", "evaluate"):
        assert main([st, "--config", str(cfg2_path)]) == 0, st
    info.update({"cfg_path": cfg_path, "cfg2_path": cfg2_path})
    return info


def test_no_leakage_and_model_metadata(repo):
    cfg = load_config(repo["cfg_path"])
    paths = Paths(cfg)
    assert not set(fit_s1_subset(cfg, paths)) & set(load_val_ids(paths))
    fr = read_json(paths.run_dir("EXP-TEST-LGBM") / "fit_report.json")
    meta = fr["model"]["meta"]
    assert fr["model"]["type"] == "lightgbm" and 0.3 <= meta["threshold"] <= 0.95
    assert meta["n_train_pairs"] > 0 and meta["n_early_stopping_pairs"] > 0
    assert "num_conflict" in meta["feature_importance"]
    scorer = make_scorer(cfg, paths)
    assert isinstance(scorer, LightGBMModel)


def test_reused_parts_give_identical_model(repo):
    p1 = Paths(load_config(repo["cfg_path"])).run_dir("EXP-TEST-LGBM") / "fit_report.json"
    p2 = Paths(load_config(repo["cfg2_path"])).run_dir("EXP-TEST-LGBM-REUSE") / "fit_report.json"
    m1, m2 = read_json(p1)["model"]["meta"], read_json(p2)["model"]["meta"]
    assert m1["n_train_pairs"] == m2["n_train_pairs"] and m1["threshold"] == m2["threshold"]
    assert read_json(p2)["scoring_stats"]["reused_parts"].endswith("scored_lgbm_fit")


def test_metrics_and_outputs(repo):
    cfg = load_config(repo["cfg_path"])
    paths = Paths(cfg)
    r = read_json(paths.run_dir("EXP-TEST-LGBM") / "metrics.json")
    assert r["matcher"] == "LightGBMModel" and "train-fold" in r["threshold_source"]
    u = r["at_threshold_used"]
    assert u["f05"] >= r["all_empty_f05"] and "false_merges" in u and "singleton_false_positive" in u
    s = read_json(paths.run_dir("EXP-TEST-LGBM") / "submission_summary.json")
    assert s["strict_validation_errors"] == []
    res = subprocess.run(
        [sys.executable, str(OFFICIAL), "-m", str(paths.output / "matching_results.tsv"), "-c", str(paths.output / "candidate_pairs.tsv"),
         "-t", str(paths.raw / "test"), "--check-ids"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert res.returncode == 0 and "PASS" in res.stdout, res.stdout + res.stderr
