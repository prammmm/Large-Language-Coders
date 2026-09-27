"""Synthetic end-to-end run of the learned-matcher path (EXP-M-02 + EXP-B-02 + EXP-F-01
options): fit -> train -> evaluate -> infer -> submit, with leakage and validator checks."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from ber.config import Paths, load_config
from ber.data.split import load_val_ids
from ber.models.logistic import LogisticModel
from ber.pipeline.run_baseline import fit_s1_subset, main
from ber.utils.runtime import read_json

from synthetic import make_synthetic_repo

REPO = Path(__file__).resolve().parents[1]
OFFICIAL = REPO / "student_resource" / "utils" / "validate_submission.py"


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("lr_repo")
    info = make_synthetic_repo(root, n_train=260, n_test=90)
    cfg = yaml.safe_load(Path(info["config"]).read_text(encoding="utf-8"))
    cfg["experiment"]["id"] = "EXP-TEST-LR"
    cfg["matcher"]["type"] = "logistic"
    cfg["decision"] = {"threshold_from": "model", "threshold": 0.5}
    cfg["fit"].update({"max_train_s1": None, "inner_tune_fraction": 0.3})
    cfg["scoring"]["tag"] = "scored_lr"
    cfg["features"] = {"strip_leading_zeros": True, "extra": ["num_conflict", "legal_form_conflict"]}
    cfg["matcher"]["features"] = cfg["matcher"]["features"] + ["num_conflict", "legal_form_conflict"]
    cfg["blocking"]["strip_leading_zeros"] = True
    cfg["blocking"]["passes"]["exact_core"]["enabled"] = True
    cfg["blocking"]["passes"]["sorted_core"]["enabled"] = True
    cfg_path = root / "config_lr.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    for st in ("preprocess", "split", "block-eval", "fit", "train", "evaluate", "infer", "submit"):
        assert main([st, "--config", str(cfg_path)]) == 0, st
    info["cfg_path"] = cfg_path
    return info


def test_fit_never_uses_validation_s1(repo):
    cfg = load_config(repo["cfg_path"])
    paths = Paths(cfg)
    fit_ids = set(fit_s1_subset(cfg, paths))
    val = set(load_val_ids(paths))
    assert fit_ids and val and not (fit_ids & val)
    assert len(fit_ids) + len(val) == len(repo["train_gt"])


def test_model_saved_with_train_fold_threshold(repo):
    cfg = load_config(repo["cfg_path"])
    paths = Paths(cfg)
    m = LogisticModel.load(paths.run_dir("EXP-TEST-LR") / "model.json")
    assert 0.3 <= m.meta["threshold"] <= 0.95
    assert m.meta["n_train_pairs"] > 0 and m.meta["positives"] > 0
    assert "num_conflict" in m.features and "legal_form_conflict" in m.features


def test_metrics_precision_first_fields(repo):
    paths = Paths(load_config(repo["cfg_path"]))
    r = read_json(paths.run_dir("EXP-TEST-LR") / "metrics.json")
    u = r["at_threshold_used"]
    for k in ("f05", "macro_precision", "macro_recall", "false_merges", "singleton_false_positive", "predicted_per_s1",
              "missed_true_matches", "blocking_recall", "candidates_per_s1"):
        assert k in u, k
    assert "train-fold" in r["threshold_source"]
    assert r["threshold_table"] and {"threshold", "f05", "precision", "recall", "singleton_fp", "predicted_per_s1"} <= set(r["threshold_table"][0])
    assert r["rescored_from_stored_features"] is True
    assert u["f05"] >= r["all_empty_f05"]
    be = read_json(paths.run_dir("EXP-TEST-LR") / "blocking_eval.json")
    assert 0 < be["at_k"]["20"]["blocking_recall"] <= 1


def test_outputs_valid(repo):
    cfg = load_config(repo["cfg_path"])
    paths = Paths(cfg)
    s = read_json(paths.run_dir("EXP-TEST-LR") / "submission_summary.json")
    assert s["strict_validation_errors"] == [] and "scored_lr" in s["scored_dir"]
    res = subprocess.run(
        [sys.executable, str(OFFICIAL), "-m", str(paths.output / "matching_results.tsv"), "-c", str(paths.output / "candidate_pairs.tsv"),
         "-t", str(paths.raw / "test"), "--check-ids"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert res.returncode == 0 and "PASS" in res.stdout, res.stdout + res.stderr
