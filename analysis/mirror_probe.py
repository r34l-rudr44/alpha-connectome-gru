"""Bid/ask and i0/i1 column structure on the samples: per-index means, stds and target correlations.

Result: bid j and ask j are different components (no mirror symmetry by index); i0 and i1 share one transform.
Usage: python analysis/mirror_probe.py
"""
import numpy as np, pyarrow.parquet as pq, sys
sys.path.insert(0, '.')
from wnn.data import feature_columns, read_sequence
np.set_printoptions(precision=3, suppress=True, linewidth=220)
for name in ['valid_sample', 'train_sample']:
    p = f'sample/datasets/{name}.parquet'
    pf = pq.ParquetFile(p); feat = feature_columns(p)
    Xs, ys = [], []
    for rg in range(pf.num_row_groups):
        X, y, need, sc = read_sequence(pf, rg, feat)
        Xs.append(X[99:]); ys.append(y[99:])
    X = np.concatenate(Xs); y = np.concatenate(ys)
    print(name, pf.num_row_groups, X.shape)
    q = [1, 10, 25, 50, 75, 90, 99]
    print(' t0 q', np.percentile(y[:, 0], q), 'mean', y[:, 0].mean(), 'std', y[:, 0].std())
    print(' t1 q', np.percentile(y[:, 1], q), 'mean', y[:, 1].mean(), 'std', y[:, 1].std())
    def c(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))
    for inst, off in (('i0', 0), ('i1', 52)):
        for grp, o, n in (('p', 0, 11), ('v', 22, 11)):
            print(f' {inst}_{grp}: per index j: mean_bid, mean_ask, std_bid, std_ask, corr(bid,ask), c(bid,t0), c(ask,t1), c(bid,t1), c(ask,t0)')
            for j in range(n):
                b = X[:, off + o + j]; a = X[:, off + o + 11 + j]
                print(f'   j={j:2d} {b.mean():6.3f} {a.mean():6.3f} {b.std():6.3f} {a.std():6.3f} {c(b,a):6.3f} | {c(b,y[:,0]):6.3f} {c(a,y[:,1]):6.3f} | {c(b,y[:,1]):6.3f} {c(a,y[:,0]):6.3f}')
        for grp, o in (('dp', 44), ('dv', 48)):
            print(f' {inst}_{grp}: mean std c(t0) c(t1)')
            for j in range(4):
                f = X[:, off + o + j]
                print(f'   j={j} {f.mean():6.3f} {f.std():6.3f} {c(f,y[:,0]):6.3f} {c(f,y[:,1]):6.3f}')
    print(' a: mean std c(t0) c(t1)')
    for j in range(8):
        f = X[:, 104 + j]
        print(f'   a{j} {f.mean():6.3f} {f.std():6.3f} {c(f,y[:,0]):6.3f} {c(f,y[:,1]):6.3f}')
