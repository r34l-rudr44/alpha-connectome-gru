"""Do models trained on train + 80% of valid score better on the valid sequences they saw? (2026-11-11: no.)

Per-sequence and pooled scored WP of train-only vs train+valid models on the valid sample; free, local.
Usage: python analysis/paired_seen_unseen.py
"""
import sys, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, '.')
from wnn.data import feature_columns, read_sequence
from wnn.infer import StepModel
from metric import weighted_pearson

rng = np.random.default_rng(0); perm = rng.permutation(1873); hold = set(perm[:int(round(0.2 * 1873))].tolist())
p = 'sample/datasets/valid_sample.parquet'; pf = pq.ParquetFile(p); feat = feature_columns(p)
names = ['s4_c3_tr', 's4_c3_tr_s1', 's3_c3', 's4_c3_s1']
ms = {n: StepModel(f'runs/{n}/weights.npz') for n in names}
rows = []; pool = {n: [] for n in names}; Y = []; H = []
for rg in range(pf.num_row_groups):
    X, y, need, sc = read_sequence(pf, rg, feat)
    m = need & sc
    out = {}
    for n, sm in ms.items():
        sm.reset(); P = np.array([sm.step(x).copy() for x in X])
        out[n] = 0.5 * (weighted_pearson(y[m, 0], P[m, 0]) + weighted_pearson(y[m, 1], P[m, 1]))
        pool[n].append(P[m])
    Y.append(y[m]); H.append(np.full(m.sum(), rg in hold))
    print(rg, 'HOLDOUT' if rg in hold else 'seen-by-hv', int(m.sum()), ' '.join(f'{n}={out[n]:.4f}' for n in names), flush=True)
Y = np.concatenate(Y); H = np.concatenate(H)
for tag, sel in (('seen-by-hv rows', ~H), ('holdout rows', H), ('all', np.ones(len(H), bool))):
    if sel.sum() == 0: continue
    print(tag, int(sel.sum()), ' '.join(f"{n}={0.5*(weighted_pearson(Y[sel,0], np.concatenate(pool[n])[sel,0]) + weighted_pearson(Y[sel,1], np.concatenate(pool[n])[sel,1])):.4f}" for n in names))
