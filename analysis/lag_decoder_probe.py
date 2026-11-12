"""Can past labels be recovered a few dozen steps later, and does that help the causal model? (free, local CPU)

Stage 1: a small GRU reads features up to step t and outputs y[t-h] for several delays h (a decoding task, not a
         forecast). Trained on the 10 train sample sequences.
Stage 2: on the 10 valid sample sequences, a linear stack y[t] ~ p[t] + sum_h b_h * (yhat[t-h | t] - p[t-h]),
         cross-fitted 5/5, where p is the shipped causal model. Compared with p alone and with the same stack using
         the true y[t-h] (ceiling if decoding were perfect).
Run: python analysis/lag_decoder_probe.py runs/s4_c3_tr/weights.npz
"""
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, ".")
from wnn.data import feature_columns, read_sequence  # noqa: E402
from wnn.infer import StepModel  # noqa: E402
from wnn.train import weighted_pearson_np  # noqa: E402

H = [10, 25, 50, 100, 200]
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(4)


def load(path):
    pf = pq.ParquetFile(path); feat = feature_columns(path)
    out = [read_sequence(pf, i, feat) for i in range(pf.num_row_groups)]
    return (np.stack([o[0] for o in out]), np.stack([o[1] for o in out]), np.stack([o[3] for o in out]))


Xtr, Ytr, _ = load("sample/datasets/train_sample.parquet")
Xva, Yva, SCva = load("sample/datasets/valid_sample.parquet")
N, T, F = Xva.shape
mu = np.nanmean(Xtr.reshape(-1, F), 0); sd = np.nanstd(Xtr.reshape(-1, F), 0) + 1e-6
norm = lambda X: np.clip(np.nan_to_num((X - mu) / sd), -8, 8).astype(np.float32)  # noqa: E731


def delayed(Y):  # target tensor (N,T,2*len(H)): column block j holds y[t-H[j]]
    D = np.zeros((Y.shape[0], Y.shape[1], 2 * len(H)), np.float32)
    for j, h in enumerate(H):
        D[:, h:, 2 * j:2 * j + 2] = Y[:, :-h]
    return D


class Dec(torch.nn.Module):
    def __init__(self, hidden=96):
        super().__init__()
        self.gru = torch.nn.GRU(F, hidden, batch_first=True)
        self.out = torch.nn.Linear(hidden, 2 * len(H))

    def forward(self, x):
        return self.out(self.gru(x)[0])


# ---- stage 1: train the decoder on chunks of the train sequences
CH, WARM = 2500, 300
Xn, Dn = torch.from_numpy(norm(Xtr)), torch.from_numpy(delayed(np.clip(Ytr, -2, 2)))
chunks = [(i, s) for i in range(Xn.shape[0]) for s in range(0, T - CH + 1, CH // 2)]
dec = Dec(); opt = torch.optim.AdamW(dec.parameters(), lr=3e-3, weight_decay=1e-4)
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 25
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EPOCHS * ((len(chunks) + 9) // 10))
t0 = time.time()
for ep in range(EPOCHS):
    perm = np.random.permutation(len(chunks)); tot = 0.0
    for b in range(0, len(perm), 10):
        idx = [chunks[k] for k in perm[b:b + 10]]
        x = torch.stack([Xn[i, s:s + CH] for i, s in idx]); d = torch.stack([Dn[i, s:s + CH] for i, s in idx])
        loss = ((dec(x)[:, WARM:] - d[:, WARM:]) ** 2).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0); opt.step(); sched.step()
        tot += loss.item()
    if ep % 5 == 4 or ep == EPOCHS - 1:
        print(f"decoder epoch {ep + 1}: loss {tot / ((len(perm) + 9) // 10):.4f}  ({time.time() - t0:.0f}s)", flush=True)

dec.eval()
with torch.no_grad():
    Dhat = dec(torch.from_numpy(norm(Xva))).numpy()  # (N,T,2*len(H)): estimate of y[t-h] made at step t
Dtrue = delayed(np.clip(Yva, -2, 2))
print("\ndecoder quality on the valid sample (Pearson of yhat[t-h|t] with y[t-h], rows >= 500):")
for j, h in enumerate(H):
    c = [np.corrcoef(Dhat[:, 500:, 2 * j + k].ravel(), Dtrue[:, 500:, 2 * j + k].ravel())[0, 1] for k in (0, 1)]
    print(f"  h={h:<4d} t0 {c[0]:.3f}  t1 {c[1]:.3f}")

# ---- stage 2: does the decoded past residual improve the shipped causal model?
sm = StepModel(sys.argv[1]); P = np.zeros((N, T, 2), np.float32)
for i in range(N):
    sm.reset()
    for t in range(T):
        P[i, t] = sm.step(Xva[i, t])
Yc = np.clip(Yva, -2, 2)
m = SCva.copy(); m[:, :500] = False


def wp(pred, seqs):
    mm = m[seqs]; y = Yva[seqs][mm]; p = pred[seqs][mm]
    return 0.5 * (weighted_pearson_np(y[:, 0], p[:, 0]) + weighted_pearson_np(y[:, 1], p[:, 1]))


def lagged(A, h):
    B = np.zeros_like(A); B[:, h:] = A[:, :-h]; return B


def stack(D, hs, label):
    """Features: p[t] (both targets) and D[t-h] - p[t-h] for each h; ridge fit on 5 sequences, test on the other 5."""
    Z = [P] + [D[..., 2 * H.index(h):2 * H.index(h) + 2] - lagged(P, h) for h in hs]
    Z = np.concatenate(Z, -1)
    out = np.zeros_like(P)
    for fit, test in ((np.arange(0, 5), np.arange(5, 10)), (np.arange(5, 10), np.arange(0, 5))):
        z = Z[fit][:, 500:].reshape(-1, Z.shape[-1]); y = Yc[fit][:, 500:].reshape(-1, 2)
        w = np.abs(y).sum(1, keepdims=True) + 1e-3
        A = (z * w).T @ z + 1e-2 * np.eye(z.shape[1]) * w.sum(); B = (z * w).T @ y
        out[test] = Z[test] @ np.linalg.solve(A, B)
    s = wp(out, np.arange(N))
    print(f"  {label:<44s} {s:.5f}  ({s - base:+.5f})")


base = wp(P, np.arange(N))
print(f"\nshipped causal model on the 10 valid sample sequences: {base:.5f}")
stack(Dhat, [], "rescaled only (control)")
for hs in ([200], [100], [50], [25], [10], [50, 100, 200], H):
    stack(Dhat, hs, f"+ decoded residual, h={hs}")
for hs in ([200], [100], [50], H):
    stack(Dtrue, hs, f"+ TRUE residual (ceiling), h={hs}")
