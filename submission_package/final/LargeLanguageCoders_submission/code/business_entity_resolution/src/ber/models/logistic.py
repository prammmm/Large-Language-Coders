from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np

def feature_matrix(feats: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    cols = []
    for name in names:
        if '*' in name:
            a, b = name.split('*', 1)
            cols.append(np.asarray(feats[a], np.float32) * np.asarray(feats[b], np.float32))
        else:
            cols.append(np.asarray(feats[name], np.float32))
    return np.stack(cols, axis=1) if cols else np.zeros((0, 0), np.float32)

def base_features(names: list[str]) -> list[str]:
    """Raw feature columns needed to build ``names`` (interaction parts expanded)."""
    out: list[str] = []
    for name in names:
        for part in name.split('*'):
            if part not in out:
                out.append(part)
    return out

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))

class LogisticModel:

    def __init__(self, features: list[str], mean=None, std=None, coef=None, intercept: float=0.0, meta: dict | None=None):
        self.features = list(features)
        d = len(self.features)
        self.mean = np.zeros(d) if mean is None else np.asarray(mean, np.float64)
        self.std = np.ones(d) if std is None else np.asarray(std, np.float64)
        self.coef = np.zeros(d) if coef is None else np.asarray(coef, np.float64)
        self.intercept = float(intercept)
        self.meta = dict(meta or {})

    def fit(self, X: np.ndarray, y: np.ndarray, l2: float=0.0001, max_iter: int=50, tol: float=1e-08, block: int=500000) -> 'LogisticModel':
        t0 = time.perf_counter()
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float64)
        n, d = X.shape
        self.mean = X.mean(axis=0, dtype=np.float64)
        self.std = X.std(axis=0, dtype=np.float64)
        self.std[self.std < 1e-09] = 1.0
        theta = np.zeros(d + 1)
        prev = np.inf
        it = 0
        for it in range(1, max_iter + 1):
            grad = np.zeros(d + 1)
            hess = np.zeros((d + 1, d + 1))
            loss = 0.0
            for a in range(0, n, block):
                Z = (X[a:a + block] - self.mean) / self.std
                Zb = np.hstack([np.ones((len(Z), 1)), Z])
                p = _sigmoid(Zb @ theta)
                yb = y[a:a + block]
                eps = 1e-12
                loss -= float(np.sum(yb * np.log(p + eps) + (1 - yb) * np.log(1 - p + eps)))
                grad += Zb.T @ (p - yb)
                hess += (Zb * (p * (1 - p))[:, None]).T @ Zb
            grad /= n
            hess /= n
            loss /= n
            reg = np.r_[0.0, np.full(d, l2)]
            loss += 0.5 * l2 * float(theta[1:] @ theta[1:])
            grad += reg * theta
            hess += np.diag(reg) + 1e-10 * np.eye(d + 1)
            step = np.linalg.solve(hess, grad)
            theta -= step
            if abs(prev - loss) < tol and np.max(np.abs(step)) < 1e-06:
                break
            prev = loss
        self.intercept, self.coef = (float(theta[0]), theta[1:].copy())
        self.meta.update({'n_train_pairs': int(n), 'positives': int(y.sum()), 'l2': l2, 'iterations': it, 'final_logloss': float(loss), 'fit_seconds': round(time.perf_counter() - t0, 2)})
        return self

    def predict_proba_matrix(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, np.float64) - self.mean) / self.std
        return _sigmoid(Z @ self.coef + self.intercept).astype(np.float32)

    def score(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        return self.predict_proba_matrix(feature_matrix(feats, self.features))

    @property
    def required_features(self) -> list[str]:
        return base_features(self.features)

    def to_dict(self) -> dict:
        return {'type': 'logistic_regression', 'features': self.features, 'mean': self.mean.tolist(), 'std': self.std.tolist(), 'coef': self.coef.tolist(), 'intercept': self.intercept, 'meta': self.meta}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path: Path) -> 'LogisticModel':
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        if d.get('type') != 'logistic_regression':
            raise ValueError(f'{path} is not a logistic regression model')
        return cls(d['features'], d['mean'], d['std'], d['coef'], d['intercept'], d.get('meta'))
