"""EXP-M-03 matcher: gradient-boosted decision trees (LightGBM, MIT license).

Same inputs and interface as ``LogisticModel`` (features by name, "a*b" = product), so it
plugs into the same `fit` / `train` / `evaluate` / `infer` stages. Trees can express
conditional rules the linear model cannot, e.g. "names almost identical AND house numbers
conflict -> not a match".

Saved as two files: ``model.json`` (type, features, threshold, meta, importance) and the
LightGBM text model ``model.lgb.txt`` next to it (plain text, reproducible).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ber.models.logistic import base_features, feature_matrix

DEFAULT_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 8,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "num_threads": 8,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}


class LightGBMModel:
    def __init__(self, features: list[str], params: dict | None = None, booster=None, meta: dict | None = None):
        self.features = list(features)
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.booster = booster
        self.meta = dict(meta or {})

    # ------------------------------------------------------------------ fitting
    def fit(self, X, y, X_valid=None, y_valid=None, num_boost_round: int = 1000, early_stopping_rounds: int = 50) -> "LightGBMModel":
        import lightgbm as lgb

        t0 = time.perf_counter()
        train = lgb.Dataset(np.asarray(X, np.float32), label=np.asarray(y, np.float32), feature_name=self._safe_names(), free_raw_data=True)
        callbacks, valid_sets = [], []
        if X_valid is not None and len(X_valid):
            valid = lgb.Dataset(np.asarray(X_valid, np.float32), label=np.asarray(y_valid, np.float32), reference=train)
            valid_sets = [valid]
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
        params = {**self.params, "metric": "binary_logloss"}
        self.booster = lgb.train(params, train, num_boost_round=num_boost_round, valid_sets=valid_sets, callbacks=callbacks)
        self.meta.update({
            "n_train_pairs": int(len(y)), "positives": int(np.sum(y)),
            "n_early_stopping_pairs": int(len(y_valid)) if y_valid is not None else 0,
            "num_boost_round_max": num_boost_round, "early_stopping_rounds": early_stopping_rounds,
            "best_iteration": int(self.booster.best_iteration or self.booster.current_iteration()),
            "fit_seconds": round(time.perf_counter() - t0, 2),
            "params": params,
        })
        return self

    def _safe_names(self) -> list[str]:
        return [n.replace("*", "_x_") for n in self.features]

    # ------------------------------------------------------------------ inference
    def predict_proba_matrix(self, X) -> np.ndarray:
        it = self.meta.get("best_iteration") or None
        return self.booster.predict(np.asarray(X, np.float32), num_iteration=it).astype(np.float32)

    def score(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        return self.predict_proba_matrix(feature_matrix(feats, self.features))

    @property
    def required_features(self) -> list[str]:
        return base_features(self.features)

    def importance(self) -> dict[str, dict[str, float]]:
        gain = self.booster.feature_importance("gain", iteration=self.meta.get("best_iteration") or None)
        split = self.booster.feature_importance("split", iteration=self.meta.get("best_iteration") or None)
        tot = float(gain.sum()) or 1.0
        return {f: {"gain_share": float(g / tot), "splits": int(s)} for f, g, s in zip(self.features, gain, split)}

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        booster_file = path.with_name(path.stem + ".lgb.txt")
        self.booster.save_model(str(booster_file), num_iteration=self.meta.get("best_iteration") or None)
        d = {"type": "lightgbm", "features": self.features, "booster_file": booster_file.name, "params": self.params,
             "meta": {**self.meta, "feature_importance": self.importance()}}
        path.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LightGBMModel":
        import lightgbm as lgb

        path = Path(path)
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("type") != "lightgbm":
            raise ValueError(f"{path} is not a LightGBM model")
        booster = lgb.Booster(model_file=str(path.with_name(d["booster_file"])))
        meta = dict(d.get("meta", {}))
        meta["best_iteration"] = None  # the saved booster already contains only the best iterations
        return cls(d["features"], d.get("params"), booster, meta)
