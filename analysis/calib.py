"""Post-hoc calibration of GRU-ensemble predictions (local CPU).

Weighted Pearson is scale-free but not invariant to non-linear transforms, and the metric clips y at +-2 while the
model may be fitting the unclipped conditional mean. Tests, fit on valid seqs A / evaluated on valid seqs B:
  1. power transform  p -> sign(p)|p|^g
  2. 1-D binned conditional mean  p_j -> E_w[clip(y_j) | p_j]
  3. small GBM stacker on (p0, p1, step, P_scored, running stats of p)  -> y_j
Usage: python analysis/calib.py --preds runs/ens_r3_valid.npy --maskw runs/maskw_valid.npy
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

WARM = config.WARMUP


def wpearson(y, p):
    y = np.clip(y, -2, 2); p = np.clip(p, -2, 2); w = np.abs(y)
    out = []
    for j in range(y.shape[1]):
        wj = w[:, j]; yj = y[:, j]; pj = p[:, j]
        my = (wj * yj).sum() / wj.sum(); mp = (wj * pj).sum() / wj.sum()
        cov = (wj * (yj - my) * (pj - mp)).sum()
        out.append(float(cov / np.sqrt((wj * (yj - my) ** 2).sum() * (wj * (pj - mp) ** 2).sum())))
    return [round(out[0], 4), round(out[1], 4), round((out[0] + out[1]) / 2, 4)]


def ema(X, a):
    from scipy.signal import lfilter

    y, _ = lfilter([a], [1.0, -(1.0 - a)], X, axis=0, zi=((1.0 - a) * X[:1]))
    return y.astype(np.float32)


def load(pf, P, W, idx, stride):
    ys, ps, ss, fs = [], [], [], []
    for rg in idx:
        t = pf.read_row_group(int(rg), columns=["t0", "t1", "is_scored"])
        y = np.stack([t.column("t0").to_numpy(), t.column("t1").to_numpy()], 1).astype(np.float32)
        sc = t.column("is_scored").to_numpy().astype(bool)
        p = np.asarray(P[rg], dtype=np.float32)
        w = np.asarray(W[rg], dtype=np.float32)[:, None] if W is not None else np.zeros((len(y), 1), np.float32)
        step = (np.arange(len(y), dtype=np.float32) / 20000.0)[:, None]
        dp = np.diff(p, axis=0, prepend=p[:1])
        feats = np.concatenate([p, step, w, ema(np.abs(dp), 0.05), p - ema(p, 0.02), np.abs(p).max(1, keepdims=True)], 1)
        sel = np.arange(WARM + (int(rg) % stride), len(y), stride)
        ys.append(y[sel]); ps.append(p[sel]); ss.append(sc[sel]); fs.append(feats[sel])
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ss), np.concatenate(fs)


def binned_map(p, y, w, nb=200):
    """Weighted conditional mean of clip(y) in quantile bins of p; returns (edges, means) for np.interp."""
    q = np.quantile(p, np.linspace(0, 1, nb + 1)); q[0] -= 1e-6; q[-1] += 1e-6
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, nb - 1)
    yc = np.clip(y, -2, 2)
    num = np.bincount(b, weights=w * yc, minlength=nb); den = np.bincount(b, weights=w, minlength=nb) + 1e-9
    centers = np.bincount(b, weights=w * p, minlength=nb) / den
    return centers, num / den


def main():
    import lightgbm as lgb

    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True); ap.add_argument("--maskw", default=None)
    ap.add_argument("--n-fit", type=int, default=900); ap.add_argument("--n-eval", type=int, default=900)
    ap.add_argument("--stride", type=int, default=4); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="runs/calib.json")
    a = ap.parse_args()
    pf = pq.ParquetFile(str(config.VALID))
    P = np.load(a.preds, mmap_mode="r"); W = np.load(a.maskw, mmap_mode="r") if a.maskw else None
    rng = np.random.default_rng(0); perm = rng.permutation(pf.num_row_groups)
    t0 = time.time()
    yA, pA, sA, fA = load(pf, P, W, perm[: a.n_fit], a.stride)
    yB, pB, sB, fB = load(pf, P, W, perm[a.n_fit: a.n_fit + a.n_eval], a.stride)
    print(f"loaded A {len(yA):,} B {len(yB):,} rows in {time.time() - t0:.0f}s", flush=True)
    res = {"rows_A": int(len(yA)), "rows_B": int(len(yB))}

    def report(name, pb):
        res[name] = {"scored": wpearson(yB[sB], pb[sB]), "all": wpearson(yB, pb)}
        print(name, res[name], flush=True)

    report("gru", pB)
    for g in (0.8, 0.9, 1.1, 1.25, 1.5):
        report(f"pow{g}", np.sign(pB) * np.abs(pB) ** g)
    # 1-D conditional-mean calibration, metric weights on scored rows of A
    for fit_mask_name, m in (("scored", sA), ("all", np.ones(len(yA), bool))):
        pc = np.empty_like(pB)
        for j in range(2):
            w = np.abs(np.clip(yA[m, j], -2, 2))
            c, mu = binned_map(pA[m, j], yA[m, j], w)
            pc[:, j] = np.interp(pB[:, j], c, mu)
        report(f"bin200_fit_{fit_mask_name}", pc)
    # GBM stacker on prediction-derived features (+ P_scored); weights = |clip(y)| * (0.5 + scored)
    params = dict(n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=500, subsample=0.8,
                  subsample_freq=1, reg_lambda=5.0, n_jobs=a.threads, verbose=-1)
    cols = {"p_only": [0, 1], "p_step": [0, 1, 2], "p_maskw": [0, 1, 3], "all": list(range(fA.shape[1]))}
    wA = np.abs(np.clip(yA, -2, 2)) * (0.5 + sA[:, None])
    for name, ci in cols.items():
        t1 = time.time(); ps = np.empty_like(pB)
        for j in range(2):
            m = lgb.LGBMRegressor(**params).fit(fA[:, ci], np.clip(yA[:, j], -2, 2), sample_weight=wA[:, j] + 1e-3)
            ps[:, j] = m.predict(fB[:, ci], num_threads=a.threads)
        report(f"stack_{name}", ps)
        report(f"stack_{name}_half", 0.5 * (ps + pB))
        print(f"  ({time.time() - t1:.0f}s)", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"saved {a.out} total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
