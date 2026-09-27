"""Baseline 1 matcher: deterministic weighted similarity (no training, DEC-006).

score = sum_i w_i * clip(f_i, 0, 1) / sum_i w_i  over the features named in the config.
Count features must be listed with a ``cap`` (value/cap, clipped) to stay in [0, 1].
"""

from __future__ import annotations

import numpy as np

from ber.features.pair_features import FEATURE_NAMES


class WeightedRuleScorer:
    def __init__(self, weights: dict[str, float], caps: dict[str, float] | None = None):
        unknown = set(weights) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown features in weights: {sorted(unknown)}")
        if not weights or sum(weights.values()) <= 0:
            raise ValueError("weights must be non-empty with a positive sum")
        self.weights = dict(weights)
        self.caps = dict(caps or {})
        self.total = float(sum(weights.values()))

    @classmethod
    def from_config(cls, cfg: dict) -> "WeightedRuleScorer":
        return cls(cfg["weights"], cfg.get("caps"))

    def score(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        n = len(next(iter(feats.values()))) if feats else 0
        s = np.zeros(n, np.float32)
        for name, w in self.weights.items():
            x = feats[name].astype(np.float32)
            if name in self.caps:
                x = x / np.float32(self.caps[name])
            s += np.float32(w) * np.clip(x, 0.0, 1.0)
        return s / np.float32(self.total)
