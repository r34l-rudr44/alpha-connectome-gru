"""Can a sequence model predict is_scored much better than LightGBM-on-current-state? Local CPU probe on the sample.

Trains a causal GRU (and a bidirectional one, to see whether the mask depends on the future) on n_fit valid-sample
sequences, reports holdout AUC against the LightGBM features of modal/mask_model.py.
Usage: PYTHONPATH=. python analysis/mask_gru_probe.py
"""
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from wnn.data import feature_columns, read_sequence

sys.path.insert(0, "modal")
from mask_model import _features  # noqa: E402

torch.manual_seed(0); np.random.seed(0)
f = "sample/datasets/valid_sample.parquet"; pf = pq.ParquetFile(f); feat = feature_columns(f)
seqs = [read_sequence(pf, i, feat) for i in range(pf.num_row_groups)]
fit, hold = seqs[:8], seqs[8:]
Xall = np.concatenate([x for x, *_ in fit]); mu, sd = Xall.mean(0), Xall.std(0) + 1e-6


def prep(S):
    X = np.stack([(x - mu) / sd for x, *_ in S]).clip(-10, 10).astype(np.float32)
    y = np.stack([sc for *_, sc in S]).astype(np.float32); need = np.stack([n for _, _, n, _ in S])
    return torch.tensor(X), torch.tensor(y), torch.tensor(need)


Xf, yf, nf = prep(fit); Xh, yh, nh = prep(hold)

# LightGBM reference (same split)
import lightgbm as lgb  # noqa: E402
F = np.concatenate([_features(x)[n][::2] for x, _, n, _ in fit]); L = np.concatenate([sc[n][::2] for _, _, n, sc in fit])
Fh = np.concatenate([_features(x)[n] for x, _, n, _ in hold]); Lh = np.concatenate([sc[n] for _, _, n, sc in hold])
m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, min_child_samples=200, verbose=-1)
m.fit(F, L); print(f"LightGBM holdout AUC {roc_auc_score(Lh, m.predict_proba(Fh)[:, 1]):.4f}", flush=True)


class M(nn.Module):
    def __init__(self, h=64, bi=False):
        super().__init__()
        self.gru = nn.GRU(112, h, batch_first=True, bidirectional=bi); self.head = nn.Linear(h * (2 if bi else 1), 1)

    def forward(self, x):
        return self.head(self.gru(x)[0]).squeeze(-1)


def run(bi, epochs=6, chunk=2000):
    net = M(bi=bi); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4); t0 = time.time()
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xf))
        for s in range(0, 20000, chunk):  # independent chunks (no state carry) - fine for a probe
            xb, yb, nb = Xf[perm, s:s + chunk], yf[perm, s:s + chunk], nf[perm, s:s + chunk]
            loss = nn.functional.binary_cross_entropy_with_logits(net(xb)[nb], yb[nb])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad():
            p = net(Xh)[nh].numpy(); auc = roc_auc_score(yh[nh].numpy(), p)
        print(f"{'bi' if bi else 'causal'} GRU ep{ep + 1}: loss {loss.item():.4f} holdout AUC {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)


run(False); run(True)
