"""Covariate-shift weights for training sequences: w = P(valid | seq descriptors) / P(train | ...) (local CPU).

Train and valid are consecutive time periods (organisers), and the regime mix drifts. A classifier on the
per-sequence descriptors (modal/seq_stats.py) gives importance weights that re-weight train towards the valid period.
Writes runs/seqw/train.npy (n_train,) and runs/seqw/valid.npy (ones), normalised to mean 1, clipped.

Usage: PYTHONPATH=. python analysis/shift_weights.py [--clip 5] [--power 1]
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

from analysis.regime import feature_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=float, default=5.0); ap.add_argument("--power", type=float, default=1.0)
    ap.add_argument("--out", default="runs/seqw")
    a = ap.parse_args()
    tr, va = np.load("runs/seq_stats/train.npz"), np.load("runs/seq_stats/valid.npz")
    Ftr, names = feature_table(tr); Fva, _ = feature_table(va)
    X = np.concatenate([Ftr, Fva]); y = np.r_[np.zeros(len(Ftr)), np.ones(len(Fva))]
    Z = StandardScaler().fit_transform(X)
    clf = LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced")
    p = cross_val_predict(clf, Z, y, cv=5, method="predict_proba")[:, 1]  # out-of-fold P(valid | x), balanced prior
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, p))
    w = (p / (1 - p)).clip(1e-3, None) ** a.power  # density ratio valid/train (balanced prior cancels the class sizes)
    w_tr = w[: len(Ftr)].clip(0, a.clip); w_tr = w_tr / w_tr.mean()
    os.makedirs(a.out, exist_ok=True)
    np.save(f"{a.out}/train.npy", w_tr.astype(np.float32)); np.save(f"{a.out}/valid.npy", np.ones(len(Fva), np.float32))
    clf.fit(Z, y); coef = dict(sorted(zip(names, clf.coef_[0].round(3)), key=lambda kv: -abs(kv[1]))[:12])
    ess = float(w_tr.sum() ** 2 / (w_tr ** 2).sum())
    info = {"auc_train_vs_valid": auc, "ess_train": ess, "n_train": len(Ftr), "w_quantiles": np.percentile(w_tr, [1, 10, 50, 90, 99]).round(3).tolist(),
            "top_coef": coef}
    json.dump(info, open(f"{a.out}/info.json", "w"), indent=1); print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
