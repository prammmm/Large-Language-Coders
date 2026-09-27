"""Matchers. ``make_scorer`` returns the scorer selected by ``cfg['matcher']['type']``."""

from __future__ import annotations

from pathlib import Path

from ber.features.pair_features import FEATURE_NAMES


def model_path(cfg: dict, paths) -> Path:
    """Where a fitted matcher of this experiment lives (``matcher.model_path`` overrides)."""
    p = cfg["matcher"].get("model_path")
    return Path(p) if p else paths.run_dir(cfg["experiment"]["id"]) / "model.json"


def make_scorer(cfg: dict, paths=None, kind: str | None = None):
    """``weighted_rule`` (deterministic, no training) or ``logistic`` (fitted by `fit`)."""
    kind = kind or cfg["matcher"].get("type", "weighted_rule")
    if kind == "weighted_rule":
        from ber.models.weighted_rule import WeightedRuleScorer

        return WeightedRuleScorer.from_config(cfg["matcher"])
    if kind in ("logistic", "lightgbm"):
        path = model_path(cfg, paths)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run the `fit` stage first")
        if kind == "logistic":
            from ber.models.logistic import LogisticModel

            return LogisticModel.load(path)
        from ber.models.gbdt import LightGBMModel

        return LightGBMModel.load(path)
    raise ValueError(f"unknown matcher type {kind!r}")


def scorer_features(scorer) -> list[str]:
    """Raw feature columns a scorer needs (to check stored features are sufficient)."""
    if hasattr(scorer, "required_features"):
        return scorer.required_features
    return list(scorer.weights)


__all__ = ["make_scorer", "model_path", "scorer_features", "FEATURE_NAMES"]
