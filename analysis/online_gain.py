"""Headroom of per-sequence test-time adaptation, from cached valid predictions (free, local CPU).

Labels are (almost) deterministic functions of the next few dozen feature rows, so at step t a submission could
know y for steps <= t - h and adapt to the sequence it is in. This script measures what that is worth for the
simplest adaptation, a per-sequence gain on each prediction:
  oracle  : gain fitted on the whole sequence (upper bound)
  causal  : gain at step t fitted on rows <= t - h, shrunk towards 1 with prior strength lam
Run: python analysis/online_gain.py [preds.npy]
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from wnn.train import weighted_pearson_np  # noqa: E402

P = np.load(sys.argv[1] if len(sys.argv) > 1 else "runs/ens_r3_valid.npy", mmap_mode="r")
t = np.load("runs/valid_targets.npz")
Y, SC = t["y"], t["scored"]
N, T, _ = Y.shape
WARM = 99


def score(preds):  # preds: list of (T,2) per sequence
    m = SC.copy(); m[:, :WARM] = False
    y = Y[m]; p = np.concatenate([preds[i][m[i]] for i in range(N)])
    return 0.5 * (weighted_pearson_np(y[:, 0], p[:, 0]) + weighted_pearson_np(y[:, 1], p[:, 1]))


base = [np.asarray(P[i]) for i in range(N)]
print(f"baseline                      {score(base):.5f}")

# per-sequence statistics with the metric's weights |clip(y)|
out_oracle = []
slopes = np.zeros((N, 2))
for i in range(N):
    y = np.clip(Y[i], -2, 2); w = np.abs(y); p = base[i]
    g = (w * p * y).sum(0) / np.maximum((w * p * p).sum(0), 1e-9)
    slopes[i] = g
    out_oracle.append(p * g)
print(f"oracle per-sequence gain      {score(out_oracle):.5f}   (slope quantiles 5/50/95%: "
      f"{np.round(np.quantile(slopes, [0.05, 0.5, 0.95]), 2)})")

gmean = np.median(slopes, 0)
for h in (50, 200):
    for lam in (0.03, 0.1, 0.3, 1.0):
        out = []
        for i in range(N):
            y = np.clip(Y[i], -2, 2); w = np.abs(y); p = base[i]
            Spy = np.cumsum(w * p * y, 0); Spp = np.cumsum(w * p * p, 0)
            Spy = np.concatenate([np.zeros((h, 2)), Spy[:-h]]); Spp = np.concatenate([np.zeros((h, 2)), Spp[:-h]])
            prior = lam * Spp[-1]  # prior strength as a fraction of a full sequence's information
            g = (prior * gmean + Spy) / (prior + Spp)
            out.append(p * g)
        print(f"causal gain  h={h:<4d} lam={lam:<5} {score(out):.5f}")
