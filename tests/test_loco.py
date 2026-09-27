"""Step C (leave-one-country-out diagnostic): fit.only_countries and a pinned xlit.map_path."""

from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from ber.config import Paths, load_config
from ber.data.normalize_records import country_key
from ber.data.split import load_val_ids
from ber.normalization.crossscript import XlitMap
from ber.pipeline.run_baseline import fit_s1_subset, main
from ber.pipeline.xlit import ensure_final_map

from synthetic import make_synthetic_repo


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    root = tmp_path_factory.mktemp("loco_repo")
    info = make_synthetic_repo(root, n_train=200, n_test=40)
    c = yaml.safe_load(Path(info["config"]).read_text(encoding="utf-8"))
    p = root / "c.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    for st in ("preprocess", "split"):
        assert main([st, "--config", str(p)]) == 0
    info.update({"root": root, "c": c})
    return info


def _cfg(repo, **fit):
    c = dict(repo["c"])
    c["fit"] = {**c["fit"], **fit}
    p = repo["root"] / f"c_{abs(hash(str(fit)))}.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
    return load_config(p)


def test_only_countries_restricts_fit_set_and_excludes_validation(repo):
    cfg = _cfg(repo, max_train_s1=None, only_countries=["us"])
    paths = Paths(cfg)
    ids = set(fit_s1_subset(cfg, paths))
    t = pq.read_table(paths.interim_file("train", "source1"), columns=["entity_id", "country"])
    ctry = dict(zip(t.column("entity_id").to_pylist(), (country_key(x) for x in t.column("country").to_pylist())))
    assert ids and {ctry[i] for i in ids} == {"us"} and not ids & set(load_val_ids(paths))
    full = set(fit_s1_subset(_cfg(repo, max_train_s1=None), paths))
    assert ids < full  # default (no option) keeps every country


def test_pinned_map_is_used_as_is(repo, tmp_path):
    cfg = _cfg(repo)
    XlitMap({"शर्मा": "sharma"}, {"label": "pinned"}).save(tmp_path / "m.json")
    cfg["xlit"] = {"enabled": True, "map_path": str(tmp_path / "m.json")}
    m = ensure_final_map(cfg, Paths(cfg))
    assert m.mapping == {"शर्मा": "sharma"} and m.meta["label"] == "pinned"
