"""Target reverse-engineering on the local sample (CPU). What is y a function of, in the future feature stream?

Usage: python analysis/target_probe.py [--file sample/datasets/valid_sample.parquet] [--out runs/target_probe.json]
"""
import argparse
import json

import numpy as np
import pyarrow.parquet as pq

from wnn.data import feature_columns, read_sequence

HORIZONS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="sample/datasets/valid_sample.parquet")
    ap.add_argument("--out", default="runs/target_probe.json")
    a = ap.parse_args()
    pf = pq.ParquetFile(a.file); feat = feature_columns(a.file)
    seqs = [read_sequence(pf, i, feat) for i in range(pf.num_row_groups)]
    F = len(feat); out = {}

    # 1. target structure
    ch0 = []; ch1 = []; both = []; runs = []
    for X, y, need, sc in seqs:
        d0 = np.diff(y[:, 0]) != 0; d1 = np.diff(y[:, 1]) != 0
        ch0.append(d0.mean()); ch1.append(d1.mean()); both.append((d0 & d1).sum() / max(1, (d0 | d1).sum()))
        idx = np.flatnonzero(d0); runs.extend(np.diff(idx).tolist())
    out["change_rate_t0"] = float(np.mean(ch0)); out["change_rate_t1"] = float(np.mean(ch1))
    out["frac_changes_simultaneous"] = float(np.mean(both))
    out["run_length_quantiles"] = np.percentile(runs, [10, 25, 50, 75, 90, 99]).tolist()
    Y = np.concatenate([y[need] for _, y, need, _ in seqs])
    out["corr_t0_t1"] = corr(Y[:, 0], Y[:, 1])
    # conditional mean of t1 given t0 bins
    q = np.quantile(Y[:, 0], np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(q, Y[:, 0]) - 1, 0, 9)
    out["E_t1_given_t0_decile"] = [float(Y[b == k, 1].mean()) for k in range(10)]
    out["t0_decile_edges"] = q.tolist()
    # distinct values: is y quantised?
    u = np.unique(np.round(Y[:, 0], 5)); out["n_unique_t0"] = int(len(u)); out["n_rows"] = int(len(Y))

    # 2. single-column future correlations: corr(y_t, X[t+h,c]) and corr(y_t, X[t+h,c]-X[t,c]); pooled over seqs
    lvl = np.zeros((2, len(HORIZONS), F)); dif = np.zeros((2, len(HORIZONS), F))
    for hi, h in enumerate(HORIZONS):
        ys, xs, ds = [], [], []
        for X, y, need, sc in seqs:
            T = len(X); t = np.arange(99, T - h)
            ys.append(y[t]); xs.append(X[t + h]); ds.append(X[t + h] - X[t])
        ys = np.concatenate(ys); xs = np.concatenate(xs); ds = np.concatenate(ds)
        for k in range(2):
            for c in range(F):
                lvl[k, hi, c] = corr(ys[:, k], xs[:, c]); dif[k, hi, c] = corr(ys[:, k], ds[:, c]) if h > 0 else 0.0
    for k in range(2):
        top = sorted(((abs(lvl[k, hi, c]), feat[c], HORIZONS[hi], float(lvl[k, hi, c])) for hi in range(len(HORIZONS)) for c in range(F)), reverse=True)[:15]
        out[f"top_level_corr_t{k}"] = [(n, h, round(v, 4)) for _, n, h, v in top]
        top = sorted(((abs(dif[k, hi, c]), feat[c], HORIZONS[hi], float(dif[k, hi, c])) for hi in range(len(HORIZONS)) for c in range(F)), reverse=True)[:15]
        out[f"top_diff_corr_t{k}"] = [(n, h, round(v, 4)) for _, n, h, v in top]

    # 3. event alignment: when y changes at t, which columns change at t+h (h in 0..3) more than base rate?
    align = {}
    for h in (0, 1, 2, 3):
        num = np.zeros(F); den = 0; base = np.zeros(F); nb = 0
        for X, y, need, sc in seqs:
            dy = np.diff(y[:, 0]) != 0  # change between t and t+1 indexed at t
            dX = np.diff(X, axis=0) != 0
            t = np.flatnonzero(dy[99:-h - 1]) + 99
            num += dX[t + h].sum(0); den += len(t); base += dX.sum(0); nb += len(dX)
        align[h] = {feat[c]: (round(float(num[c] / den), 3), round(float(base[c] / nb), 3)) for c in np.argsort(-num / den)[:8]}
    out["change_alignment_h_{P(col changes at t+h | y changes at t), base}"] = align

    # 4. ridge on future diffs (all cols, horizons) - how linear is y in the future stream?
    hs = [1, 5, 20, 100, 500, 1000]
    tr, te = seqs[: len(seqs) // 2], seqs[len(seqs) // 2:]

    def design(S):
        A, B = [], []
        for X, y, need, sc in S:
            T = len(X); t = np.arange(99, T - max(hs), 3)
            cols = [X[t]] + [X[t + h] - X[t] for h in hs]
            A.append(np.concatenate(cols, 1)); B.append(y[t])
        return np.concatenate(A), np.concatenate(B)
    A, B = design(tr); A2, B2 = design(te)
    mu, sd = A.mean(0), A.std(0) + 1e-6; A = (A - mu) / sd; A2 = (A2 - mu) / sd
    W = np.linalg.solve(A.T @ A + 10.0 * np.eye(A.shape[1]), A.T @ B)
    P = A2 @ W
    out["ridge_future_diffs_test_corr"] = [corr(B2[:, k], P[:, k]) for k in range(2)]
    W_c = np.linalg.solve(A[:, :F].T @ A[:, :F] + 10.0 * np.eye(F), A[:, :F].T @ B)
    out["ridge_current_only_test_corr"] = [corr(B2[:, k], (A2[:, :F] @ W_c)[:, k]) for k in range(2)]

    # 5. a0..a7 behaviour + per-sequence drift of all features
    Xall = np.concatenate([X for X, *_ in seqs]); gm, gs = Xall.mean(0), Xall.std(0)
    seq_means = np.stack([X.mean(0) for X, *_ in seqs]); seq_stds = np.stack([X.std(0) for X, *_ in seqs])
    drift = seq_means.std(0) / gs; sratio = seq_stds.std(0) / gs
    out["per_seq_mean_drift_over_global_std_top"] = {feat[c]: round(float(drift[c]), 3) for c in np.argsort(-drift)[:15]}
    out["per_seq_std_variation_top"] = {feat[c]: round(float(sratio[c]), 3) for c in np.argsort(-sratio)[:15]}
    out["per_seq_mean_drift_median"] = float(np.median(drift))
    out["a_cols"] = {}
    for c in range(104, 112):
        col = Xall[:, c]
        out["a_cols"][feat[c]] = {"n_unique": int(len(np.unique(col))), "change_rate": float((np.diff(col) != 0).mean()),
                                  "seq_mean_std": float(seq_means[:, c].std()), "global_std": float(gs[c]),
                                  "autocorr_lag1": corr(col[:-1], col[1:])}
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
