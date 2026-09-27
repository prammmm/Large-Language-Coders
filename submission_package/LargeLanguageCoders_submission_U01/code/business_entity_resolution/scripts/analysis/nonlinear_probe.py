"""EXP-D-02 (diagnostic, NOT adopted): would a NON-LINEAR matcher beat logistic regression on
exactly the same features, training data and threshold protocol?

Protocol = the production `fit` protocol:
  * training pairs = the stored `fit` parts (train-fold S1 only; validation excluded);
  * inner split of those S1 by the same seeded hash; train on the inner-train part, tune the
    threshold on the inner-tune part (entity-level F0.5);
  * evaluate once on the frozen validation parts; compare with the saved LR model.
Model: small MLP (inputs -> 64 -> 32 -> 1), Adam, 4 epochs, seed 42. torch is used only here.

    python scripts/analysis/nonlinear_probe.py --config configs/exp_f01b_conflicts_zeros.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np
import torch

from ber.config import Paths, load_config
from ber.evaluation.metrics import evaluate_arrays, threshold_sweep
from ber.data.split import load_val_ids
from ber.models import make_scorer
from ber.models.logistic import feature_matrix
from ber.pipeline.run_baseline import _read_parts, fit_s1_subset, label_pairs, scored_dir
from ber.utils.runtime import configure_arrow, read_json, write_json

GRID = np.round(np.arange(0.30, 0.951, 0.01), 2)


def _frac(seed, s):
    return int(hashlib.sha1(f"{seed}|{s}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_f01b_conflicts_zeros.yaml")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="nonlinear_probe.json")
    a = ap.parse_args()
    configure_arrow()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    cfg = load_config(a.config)
    paths = Paths(cfg)
    exp = cfg["experiment"]["id"]
    lr = make_scorer(cfg, paths)
    names = lr.features
    raw = list(dict.fromkeys([p for n in names for p in n.split("*")]))

    # ---- training data: stored fit parts (train-fold S1)
    fit_ids = fit_s1_subset(cfg, paths)
    fd = _read_parts(scored_dir(cfg, paths, "train", "fit"), ["s1_row", "pool_row"] + raw)
    y, fpos, n_true_fit, _ = label_pairs(paths, fit_ids, fd["s1_row"], fd["pool_row"])
    ent_tune = np.array([_frac(cfg["seed"], s) < float(cfg["fit"].get("inner_tune_fraction", 0.2)) for s in fit_ids])
    ptune = ent_tune[fpos]
    X = feature_matrix(fd, names)
    mu, sd = X[~ptune].mean(0), X[~ptune].std(0) + 1e-6
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.from_numpy((X[~ptune] - mu) / sd).float().to(dev)
    yt = torch.from_numpy(y[~ptune].astype(np.float32)).to(dev)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    g = torch.Generator(device="cpu").manual_seed(a.seed)
    t0 = time.perf_counter()
    for _ in range(a.epochs):
        perm = torch.randperm(len(yt), generator=g).to(dev)
        for b in range(0, len(yt), 8192):
            idx = perm[b : b + 8192]
            opt.zero_grad()
            loss = lossf(net(Xt[idx]).squeeze(1), yt[idx])
            loss.backward()
            opt.step()
    train_s = time.perf_counter() - t0

    def predict(Xn):
        with torch.no_grad():
            out = []
            for b in range(0, len(Xn), 262144):
                out.append(torch.sigmoid(net(torch.from_numpy((Xn[b : b + 262144] - mu) / sd).float().to(dev))).squeeze(1).cpu().numpy())
            return np.concatenate(out)

    # threshold on the inner train-fold tune split
    tpos = np.flatnonzero(ent_tune)
    remap = -np.ones(len(fit_ids), np.int64)
    remap[tpos] = np.arange(len(tpos))
    sw = threshold_sweep(remap[fpos[ptune]], y[ptune], predict(X[ptune]), n_true_fit[ent_tune], GRID)
    t_mlp = max(sw, key=lambda r: (round(r["f05"], 6), r["threshold"]))["threshold"]

    # ---- validation
    val = load_val_ids(paths)
    vd = _read_parts(scored_dir(cfg, paths, "train", "eval_read"), ["s1_row", "pool_row"] + raw)
    vy, vpos, vn, vc = label_pairs(paths, val, vd["s1_row"], vd["pool_row"])
    m = len(val)

    def ev(score, thr):
        sel = score >= np.float32(thr)
        r = evaluate_arrays(np.bincount(vpos[sel & vy], minlength=m), np.bincount(vpos[sel], minlength=m), vn, countries=vc)
        keep = ("f05", "macro_precision", "macro_recall", "false_merges", "singleton_false_positive", "predicted_per_s1")
        return {**{k: r[k] for k in keep}, "by_country": {c: {k: v[k] for k in keep} for c, v in r["by_country"].items()}}

    t_lr = float(read_json(paths.run_dir(exp) / "metrics.json")["threshold_used"])
    rep = {
        "train_pairs": int((~ptune).sum()), "train_seconds": round(train_s, 1), "device": dev, "epochs": a.epochs,
        "logistic_regression": {"threshold": t_lr, **ev(lr.score(vd), t_lr)},
        "mlp": {"threshold": t_mlp, **ev(predict(feature_matrix(vd, names)), t_mlp)},
    }
    rep["seed"] = a.seed
    rep["mlp_inner_tune_f05"] = max(r["f05"] for r in sw)
    write_json(paths.run_dir(exp) / "analysis" / a.out, rep)
    for k in ("logistic_regression", "mlp"):
        r = rep[k]
        print(k, "t=%.2f F0.5 %.4f P %.4f R %.4f FM %d sFP %d" % (r["threshold"], r["f05"], r["macro_precision"], r["macro_recall"],
                                                                   r["false_merges"], r["singleton_false_positive"]),
              {c: round(v["f05"], 4) for c, v in r["by_country"].items()})


if __name__ == "__main__":
    main()
