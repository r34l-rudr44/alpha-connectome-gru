"""Probe (local CPU): do causal hand features carry signal the GRU ensemble misses?

Fits LightGBM on causal per-step features for (a) the targets directly and (b) the GRU residual y - p_gru,
using one set of valid sequences, and evaluates on held-out valid sequences:
  WP(gru), WP(gbm), WP(gru + gbm_resid), WP(blend)  -- masked (is_scored) and unmasked (need_prediction).
A residual gain > ~0.003 means the feature set is worth using (stacking / distillation / GRU inputs).

Usage: python analysis/probe.py --preds runs/ens_r3_valid.npy --n-fit 400 --n-eval 300 --stride 25
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

META = ("seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1")
LAGS = (1, 5, 20, 100, 500)
EMAS = (0.1, 0.01)
VOL_A = 0.05
WARM = config.WARMUP


def read_seq(pf, rg, feat):
    t = pf.read_row_group(rg)
    X = np.stack([t.column(c).to_numpy(zero_copy_only=False) for c in feat], 1).astype(np.float32, copy=False)
    y = np.stack([t.column("t0").to_numpy(), t.column("t1").to_numpy()], 1).astype(np.float32)
    need = t.column("need_prediction").to_numpy().astype(bool)
    sc = t.column("is_scored").to_numpy().astype(bool)
    return X, y, need, sc


def ema(X, a):
    from scipy.signal import lfilter

    y, _ = lfilter([a], [1.0, -(1.0 - a)], X, axis=0, zi=((1.0 - a) * X[:1]))
    return y.astype(np.float32)


def feature_names(feat):
    names = list(feat)
    for k in LAGS:
        names += [f"{c}_d{k}" for c in feat]
    for a in EMAS:
        names += [f"{c}_dev{a}" for c in feat]
    names += [f"{c}_vol" for c in feat]
    names += [f"x_{feat[i]}_minus_{feat[i + 52]}" for i in range(52)]
    names += ["step"]
    return names


def causal_features(X):
    """(T,112) -> (T, F) float32; every column uses only rows <= t."""
    T = len(X)
    blocks = [X]
    for k in LAGS:
        idx = np.maximum(np.arange(T) - k, 0)
        blocks.append(X - X[idx])
    for a in EMAS:
        blocks.append(X - ema(X, a))
    dX = np.diff(X, axis=0, prepend=X[:1])
    blocks.append(ema(np.abs(dX), VOL_A))
    blocks.append(X[:, :52] - X[:, 52:104])
    blocks.append((np.arange(T, dtype=np.float32) / 20000.0)[:, None])
    return np.concatenate(blocks, 1)


def wpearson(y, p):
    y = np.clip(y, -2, 2); p = np.clip(p, -2, 2); w = np.abs(y)
    out = []
    for j in range(y.shape[1]):
        wj = w[:, j]; yj = y[:, j]; pj = p[:, j]
        my = (wj * yj).sum() / wj.sum(); mp = (wj * pj).sum() / wj.sum()
        cov = (wj * (yj - my) * (pj - mp)).sum()
        out.append(float(cov / np.sqrt((wj * (yj - my) ** 2).sum() * (wj * (pj - mp) ** 2).sum())))
    return out


def main():
    import lightgbm as lgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="(n_valid_rg, 20000, 2) float16 GRU-ensemble predictions on valid")
    ap.add_argument("--n-fit", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--eval-stride", type=int, default=10)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default="runs/probe.json")
    a = ap.parse_args()

    pf = pq.ParquetFile(str(config.VALID))
    feat = [c for c in pf.schema_arrow.names if c not in META]
    P = np.load(a.preds, mmap_mode="r")
    assert P.shape[0] == pf.num_row_groups, (P.shape, pf.num_row_groups)
    rng = np.random.default_rng(0); perm = rng.permutation(pf.num_row_groups)
    fit_idx, eval_idx = perm[: a.n_fit], perm[a.n_fit: a.n_fit + a.n_eval]
    names = feature_names(feat)

    t0 = time.time()
    Fs, Ys, Rs, Ws = [], [], [], []
    for n, rg in enumerate(fit_idx):
        X, y, need, sc = read_seq(pf, int(rg), feat)
        p = np.asarray(P[rg], dtype=np.float32)
        sel = np.arange(WARM + (rg % a.stride), len(X), a.stride)  # staggered stride
        F = causal_features(X)[sel]
        Fs.append(F); Ys.append(y[sel]); Rs.append(y[sel] - p[sel]); Ws.append(sc[sel])
        if n % 100 == 0:
            print(f"fit features {n}/{len(fit_idx)} {time.time() - t0:.0f}s", flush=True)
    Xf = np.concatenate(Fs); Yf = np.concatenate(Ys); Rf = np.concatenate(Rs); Sf = np.concatenate(Ws)
    del Fs, Ys, Rs, Ws
    print(f"fit matrix {Xf.shape} ({Xf.nbytes / 1e9:.2f} GB) scored {Sf.mean():.3f} in {time.time() - t0:.0f}s", flush=True)

    # Row weights mirror the metric: |clip(y)| ; scored rows get extra weight (the mask is state-dependent).
    wrow = np.abs(np.clip(Yf, -2, 2)) * (0.5 + Sf[:, None])
    params = dict(n_estimators=a.trees, learning_rate=0.05, num_leaves=63, min_child_samples=200, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.4, reg_lambda=1.0, n_jobs=a.threads, verbose=-1)
    models = {}
    for tag, T in (("y", Yf), ("r", Rf)):
        for j in range(2):
            t1 = time.time()
            m = lgb.LGBMRegressor(**params).fit(Xf, T[:, j], sample_weight=wrow[:, j] + 1e-3)
            models[(tag, j)] = m
            print(f"fit {tag}{j} {time.time() - t1:.0f}s", flush=True)
    del Xf

    ev = {k: [] for k in ("y", "p", "gbm", "res", "sc")}
    t1 = time.time()
    for n, rg in enumerate(eval_idx):
        X, y, need, sc = read_seq(pf, int(rg), feat)
        p = np.asarray(P[rg], dtype=np.float32)
        sel = np.arange(WARM + (rg % a.eval_stride), len(X), a.eval_stride)
        F = causal_features(X)[sel]
        ev["y"].append(y[sel]); ev["p"].append(p[sel]); ev["sc"].append(sc[sel])
        ev["gbm"].append(np.stack([models[("y", j)].predict(F, num_threads=a.threads) for j in range(2)], 1))
        ev["res"].append(np.stack([models[("r", j)].predict(F, num_threads=a.threads) for j in range(2)], 1))
        if n % 100 == 0:
            print(f"eval {n}/{len(eval_idx)} {time.time() - t1:.0f}s", flush=True)
    E = {k: np.concatenate(v) for k, v in ev.items()}

    res = {"n_fit": a.n_fit, "n_eval": a.n_eval, "stride": a.stride, "trees": a.trees, "n_eval_rows": int(len(E["y"]))}
    for mask_name, m in (("scored", E["sc"]), ("all", np.ones(len(E["y"]), bool))):
        y, p, g, r = E["y"][m], E["p"][m], E["gbm"][m], E["res"][m]
        d = {"gru": wpearson(y, p), "gbm": wpearson(y, g)}
        for s in (0.5, 1.0):
            d[f"gru+{s}res"] = wpearson(y, p + s * r)
        for b in (0.1, 0.2, 0.3):
            d[f"blend{b}"] = wpearson(y, (1 - b) * p + b * g)
        res[mask_name] = {k: [round(v[0], 4), round(v[1], 4), round((v[0] + v[1]) / 2, 4)] for k, v in d.items()}
        print(mask_name, json.dumps(res[mask_name], indent=0), flush=True)
    # residual-model feature importance: what does the GRU miss?
    imp = {}
    for j in range(2):
        g = models[("r", j)].booster_.feature_importance("gain")
        top = np.argsort(-g)[:25]
        imp[f"r{j}"] = [(names[i], round(float(g[i] / g.sum()), 4)) for i in top]
    res["resid_top_features"] = imp
    print(json.dumps(imp, indent=0), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"saved {a.out} total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
