"""EXP-S-04: B-10 blocker (weight 1.0, top-K 20) + fresh stage-1 fit + fresh S-03-style veto refit (config only).
A/E/F blocker = B-10 exactly (3-token pair construction, determinism and dedup are proved in test_blocking_b10)
B weight 1.0  C top-K 20  D no new pass  G fit set disjoint from validation  H fresh stage-1 model under EXP-S-04
I fresh stage-2 model (no reuse of S-03)  J/K veto only, zero additions  L deterministic veto  M/N isolated paths, never root output/."""

from pathlib import Path

import numpy as np
import pytest

from ber.config import Paths, load_config
from ber.data.split import load_val_ids
from ber.decision.stage2 import restrict_additions
from ber.models import model_path

S1C, S2C = "configs/exp_s04_stage1_b10.yaml", "configs/exp_s04_veto_on_b10.yaml"
B09, B10, S03 = "configs/exp_b09_name_num.yaml", "configs/exp_b10_name_pair3.yaml", "configs/exp_s03_veto_on_b09.yaml"


def test_A_B_C_D_blocker_is_b10_exactly():
    s4, b10, b09 = (load_config(x) for x in (S1C, B10, B09))
    assert s4["blocking"] == b10["blocking"]  # A: identical blocker (3-token pair key already implemented + tested)
    nt = s4["blocking"]["passes"]["name_token"]
    assert nt["pair"] is True and nt["max_tokens"] == 3
    assert nt["weight"] == 1.0  # B: NOT B-11's 2.0
    assert s4["blocking"]["top_k"] == 20  # C
    assert set(s4["blocking"]["passes"]) == set(b09["blocking"]["passes"])  # D: no new pass
    others = lambda c: {k: v for k, v in c["blocking"]["passes"].items() if k != "name_token"}  # noqa: E731
    assert others(s4) == others(b09)
    assert load_config(S2C)["blocking"] == b10["blocking"]


def test_stage1_everything_else_unchanged():
    s4, b09 = load_config(S1C), load_config(B09)
    for sec in ("features", "decision", "fit", "xlit", "lightgbm", "split", "seed"):
        assert s4.get(sec) == b09.get(sec), sec
    assert {k: v for k, v in s4["matcher"].items() if k != "model_path"} == {k: v for k, v in b09["matcher"].items() if k != "model_path"}


def test_G_fit_set_disjoint_and_same_as_b09():
    from ber.pipeline.run_baseline import fit_s1_subset
    c = load_config(S1C)
    paths = Paths(c)
    if not paths.val_ids.exists():
        pytest.skip("frozen split not present")
    fit = set(fit_s1_subset(c, paths))
    assert len(fit) == 200000 and not fit & set(load_val_ids(paths))
    assert fit == set(fit_s1_subset(load_config(B09), Paths(load_config(B09))))


def test_H_fresh_stage1_model_path():
    c = load_config(S1C)
    p = model_path(c, Paths(c))
    assert p.as_posix().endswith("experiments/runs/EXP-S-04/model.json")
    assert c["scoring"]["tag"] == "scored_EXP-S-04"
    v = load_config(S2C)
    assert v["matcher"]["model_path"] == "experiments/runs/EXP-S-04/model.json"
    if p.exists():  # after the fit: the model was trained on the S-04 (B-10 blocker) fit parts
        import json
        meta = json.loads(p.read_text(encoding="utf-8"))["meta"]
        assert "scored_EXP-S-04_fit" in meta["training_parts"].replace("\\", "/") and meta["config"].endswith("exp_s04_stage1_b10.yaml")


def test_I_fresh_stage2_no_s03_reuse():
    from ber.pipeline.run_stage2 import feat_dir, model_dir, reuses
    v, s03 = load_config(S2C), load_config(S03)
    paths = Paths(v)
    s2 = v["stage2"]
    assert "model_from" not in s2 and "features_from" not in s2 and not reuses(v, "model_from") and not reuses(v, "features_from")
    assert model_dir(v, paths).name == "EXP-S-04" and feat_dir(v, paths, "train").name == "stage2_EXP-S-04"
    assert s2["stage1_run"] == "EXP-S-04" and s2["train_tag"] == "forensic_alltrain_S04"
    same = {k: v2 for k, v2 in s2.items() if k not in ("stage1_run", "train_tag", "xlit_map")}
    assert same == {k: v3 for k, v3 in s03["stage2"].items() if k not in ("stage1_run", "train_tag", "xlit_map")}  # same recipe
    assert "ef_prune" not in s2


def test_J_K_L_veto_only_zero_additions():
    v = load_config(S2C)
    from ber.pipeline.run_stage2 import add_floor
    assert np.isinf(add_floor(v))
    rng = np.random.default_rng(42)
    n = 5000
    control = rng.random(n) < 0.3
    s2 = rng.random(n) < 0.5
    score = rng.random(n).astype(np.float32)
    fin = restrict_additions(control, s2, score, np.inf)
    assert not (fin & ~control).any()  # J/K: subset of the stage-1 + D-01 accepted set, zero additions
    assert np.array_equal(fin, control & s2)
    assert np.array_equal(fin, restrict_additions(control, s2, score, np.inf))  # L


def test_M_N_isolated_paths_never_root_output():
    for cf in (S1C, S2C):
        c = load_config(cf)
        paths = Paths(c)
        out = Path(c["paths"]["output_dir"]).as_posix()
        assert out.startswith("experiments/runs/EXP-S-04/") and paths.output.resolve() != Path("output").resolve()
        assert c["experiment"]["id"] == "EXP-S-04"
        assert paths.run_dir(c["experiment"]["id"]).as_posix().endswith("experiments/runs/EXP-S-04")
    assert load_config(S2C)["stage2"]["xlit_map"] == "experiments/runs/EXP-S-04/xlit_map.json"
