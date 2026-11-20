"""Regime study (local CPU): do sequences fall into distinct markets/regimes, and does the model's accuracy or the
scoring mask depend on them? Uses runs/seq_stats/{train,valid}.npz (modal/seq_stats.py), runs/valid_targets.npz and
runs/ens_r3_valid.npy.

Usage: PYTHONPATH=. python analysis/regime.py [--pred runs/ens_r3_valid.npy] [--out runs/regime.json]
"""
import argparse
import json

import numpy as np
from scipy.stats import ks_2samp, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from analysis.blend_eval import wp

WARM = 99


def feature_table(d):
    """Per-sequence regime descriptors: a-column stats, log activity per column group, a7 cardinality."""
    cols, names = [], []
    for k in ("a_mean", "a_std", "a_change"):
        cols.append(d[k]); names += [f"{k}{i}" for i in range(8)]
    cols.append(np.log1p(d["grp_absdx"])); names += [f"logdx_{g}" for g in d["group_keys"]]
    cols.append(d["grp_change"]); names += [f"chg_{g}" for g in d["group_keys"]]
    cols.append(np.log(d["a_nuniq"][:, 7:8])); names.append("log_a7_nuniq")
    return np.concatenate(cols, 1).astype(np.float64), names


def per_seq_wp(y, sc, p):
    out = np.zeros(len(y))
    for i in range(len(y)):
        m = sc[i]
        out[i] = wp(y[i][m], p[i][m])[2] if m.sum() > 50 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="runs/ens_r3_valid.npy"); ap.add_argument("--out", default="runs/regime.json")
    ap.add_argument("--k", type=int, nargs="+", default=[2, 3, 4, 6])
    a = ap.parse_args()
    tr, va = np.load("runs/seq_stats/train.npz"), np.load("runs/seq_stats/valid.npz")
    t = np.load("runs/valid_targets.npz"); y, sc = t["y"], t["scored"].copy(); sc[:, :WARM] = False
    p = np.asarray(np.load(a.pred, mmap_mode="r"), dtype=np.float32)
    Ftr, names = feature_table(tr); Fva, _ = feature_table(va)
    res = {"n_train": len(Ftr), "n_valid": len(Fva)}

    # 1. train vs valid shift per descriptor (KS statistic)
    ks = {n: float(ks_2samp(Ftr[:, j], Fva[:, j]).statistic) for j, n in enumerate(names)}
    res["ks_train_vs_valid_top"] = dict(sorted(ks.items(), key=lambda kv: -kv[1])[:12])
    res["ks_median"] = float(np.median(list(ks.values())))

    # 2. per-sequence difficulty and mask rate vs descriptors
    wps = per_seq_wp(y, sc, p); ok = ~np.isnan(wps)
    res["per_seq_wp"] = {"mean": float(np.nanmean(wps)), "std": float(np.nanstd(wps)),
                         "q10_q50_q90": np.nanpercentile(wps, [10, 50, 90]).tolist()}
    srate = va["scored_rate"][:, 0]; yabs = va["y_absmean"].mean(1)
    corr_wp = {n: float(spearmanr(Fva[ok, j], wps[ok]).statistic) for j, n in enumerate(names)}
    corr_sr = {n: float(spearmanr(Fva[:, j], srate).statistic) for j, n in enumerate(names)}
    corr_ya = {n: float(spearmanr(Fva[:, j], yabs).statistic) for j, n in enumerate(names)}
    top = lambda d: dict(sorted(d.items(), key=lambda kv: -abs(kv[1]))[:10])
    res["spearman_vs_seq_wp_top"] = top(corr_wp); res["spearman_vs_scored_rate_top"] = top(corr_sr)
    res["spearman_vs_abs_y_top"] = top(corr_ya)
    res["scored_rate"] = {"mean": float(srate.mean()), "q10_q50_q90": np.percentile(srate, [10, 50, 90]).tolist()}

    # 3. clustering on descriptors (fit on train+valid), per-cluster composition, pooled WP, mask rate
    sclr = StandardScaler().fit(np.concatenate([Ftr, Fva]))
    Ztr, Zva = sclr.transform(Ftr), sclr.transform(Fva)
    res["clusters"] = {}
    for k in a.k:
        km = KMeans(k, n_init=10, random_state=0).fit(np.concatenate([Ztr, Zva]))
        ctr, cva = km.predict(Ztr), km.predict(Zva)
        rows = []
        for c in range(k):
            m = cva == c
            if m.sum() == 0:
                rows.append({"c": c, "n_train": int((ctr == c).sum()), "n_valid": 0}); continue
            msk = sc[m]; ym, pm = y[m][msk], p[m][msk]
            rows.append({"c": c, "n_train": int((ctr == c).sum()), "n_valid": int(m.sum()),
                         "frac_train": float((ctr == c).mean()), "frac_valid": float(m.mean()),
                         "wp_pooled": wp(ym, pm)[2], "scored_rate": float(srate[m].mean()), "abs_y": float(yabs[m].mean()),
                         "seq_wp_mean": float(np.nanmean(wps[m])), "weight_share": float(np.abs(np.clip(ym, -2, 2)).sum())})
        tot = sum(r.get("weight_share", 0) for r in rows)
        for r in rows:
            if "weight_share" in r:
                r["weight_share"] = round(r["weight_share"] / tot, 3)
        # per-cluster affine recalibration (scale per cluster, fit on even seqs, test on odd) vs global
        ev = np.arange(len(Fva)) % 2 == 0
        gl = {}
        for tag, sel in (("fit_even", ev), ("test_odd", ~ev)):
            gl[tag] = {}
        def pooled(sel, pred):
            msk = sc[sel]; return wp(y[sel][msk], pred[sel][msk])[2]
        scales = np.ones((k, 2))
        for c in range(k):
            m = (cva == c) & ev
            if m.sum() < 5:
                continue
            msk = sc[m]; ym, pm = np.clip(y[m][msk], -2, 2), p[m][msk]; w = np.abs(ym)
            for j in range(2):
                scales[c, j] = (w[:, j] * ym[:, j] * pm[:, j]).sum() / max((w[:, j] * pm[:, j] ** 2).sum(), 1e-6)
        p_adj = p * scales[cva][:, None, :]
        rows_summary = {"k": k, "rows": rows, "recal_odd_global": pooled(~ev, p), "recal_odd_per_cluster_scale": pooled(~ev, p_adj),
                        "scales": scales.round(3).tolist()}
        res["clusters"][str(k)] = rows_summary

    # 4. a7 as a categorical: how many sequences share a7 values; a7 mean per seq distribution
    a7 = va["a_mean"][:, 7]
    res["a7_seq_mean_q"] = np.percentile(a7, [0, 10, 50, 90, 100]).tolist()
    res["a7_first_nuniq_valid"] = int(len(np.unique(va["a_first"][:, 7])))
    res["a7_first_nuniq_train"] = int(len(np.unique(tr["a_first"][:, 7])))
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
