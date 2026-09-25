"""Config loading. Every tunable lives in a YAML file under ``configs/``."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("inherit", None)  # optional: build on another config file
    if parent:
        base = load_config(Path(path).parent / parent)
        base.pop("_config_path", None)
        cfg = deep_update(base, cfg)
    if overrides:
        cfg = deep_update(cfg, overrides)
    cfg["_config_path"] = str(path)
    return cfg


def parse_sets(items: list[str] | None) -> dict:
    """``["blocking.top_k=30", "matcher.type=logistic"]`` -> nested override dict (YAML-typed values)."""
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


class Paths:
    """Resolved filesystem layout, all relative to ``cfg['paths']['root']``."""

    def __init__(self, cfg: dict):
        p = cfg["paths"]
        root = Path(p.get("root", "."))
        self.root = root
        self.raw = root / p["raw_dir"]
        self.interim = root / p["interim_dir"]
        self.processed = root / p["processed_dir"]
        self.splits = root / p["splits_dir"]
        self.output = root / p["output_dir"]
        self.runs = root / p["runs_dir"]

    def raw_file(self, split: str, source: str) -> Path:
        """source in {'source1','source2','source3','ground_truth'}"""
        return self.raw / split / f"{split}_{source}.tsv"

    def interim_file(self, split: str, source: str) -> Path:
        return self.interim / split / f"{source}.parquet"

    def processed_dir(self, split: str) -> Path:
        return self.processed / split

    def scored_dir(self, split: str, tag: str = "scored") -> Path:
        """Scored candidate parts. ``tag`` separates experiments (EXP-B-01 used "scored")."""
        return self.processed / split / tag

    @property
    def val_ids(self) -> Path:
        return self.splits / "val_s1_ids.txt"

    def run_dir(self, exp_id: str) -> Path:
        return self.runs / exp_id
