"""Exploratory statistics on the Modal volume: target stats, mask stats, feature scales.

Streams row groups (one sequence each), keeps only running sums, prints a summary.
Usage: modal run modal/eda.py [--n-train 300]
"""
import modal

app = modal.App("wnn-connectome-eda")
vol = modal.Volume.from_name("wnn-connectome-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install("pyarrow", "numpy")
PACK = "/data/wnn_connectome_starterpack"


@app.function(image=image, volumes={"/data": vol}, timeout=2 * 3600, cpu=4, memory=16384)
def eda(n_train: int = 300):
    import json
    import numpy as np
    import pyarrow.parquet as pq

    def summarize(path, max_rg, with_mask):
        pf = pq.ParquetFile(path)
        cols = pf.schema_arrow.names
        feat = [c for c in cols if c not in ("seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1")]
        F = len(feat)
        s = np.zeros(F); s2 = np.zeros(F); mn = np.full(F, np.inf); mx = np.full(F, -np.inf)
        nan = np.zeros(F); const_in_seq = np.zeros(F); n = 0
        ys = []; masks = []; steps = []
        ac1 = []; ac100 = []
        for rg in range(min(max_rg, pf.num_row_groups)):
            t = pf.read_row_group(rg)
            X = np.stack([t.column(c).to_numpy() for c in feat], 1).astype(np.float64)
            nan += np.isnan(X).sum(0)
            Xf = np.nan_to_num(X)
            s += Xf.sum(0); s2 += (Xf ** 2).sum(0); n += len(X)
            mn = np.minimum(mn, np.nanmin(X, 0)); mx = np.maximum(mx, np.nanmax(X, 0))
            const_in_seq += (np.nanstd(X, 0) == 0)
            y = np.stack([t.column("t0").to_numpy(), t.column("t1").to_numpy()], 1)
            need = t.column("need_prediction").to_numpy()
            m = need & (t.column("is_scored").to_numpy() if with_mask else True)
            ys.append(y[need]); masks.append(m[need]); steps.append(t.column("step_in_seq").to_numpy()[need])
            yy = y[need]
            for lag, acc in ((1, ac1), (100, ac100)):
                a, b = yy[:-lag], yy[lag:]
                acc.append([np.corrcoef(a[:, k], b[:, k])[0, 1] for k in range(2)])
        y = np.concatenate(ys); m = np.concatenate(masks); st = np.concatenate(steps)
        mean = s / n; std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0))
        q = np.quantile(y, [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999], axis=0)
        out = {
            "file": path, "sequences": min(max_rg, pf.num_row_groups), "rows_need_pred": int(len(y)),
            "target_mean": y.mean(0).tolist(), "target_std": y.std(0).tolist(),
            "target_quantiles_0.1%..99.9%": q.T.round(4).tolist(),
            "frac_abs_y_gt_2": (np.abs(y) > 2).mean(0).tolist(),
            "frac_abs_y_gt_1": (np.abs(y) > 1).mean(0).tolist(),
            "corr_t0_t1": float(np.corrcoef(y[:, 0], y[:, 1])[0, 1]),
            "autocorr_lag1_mean": np.mean(ac1, 0).tolist(), "autocorr_lag100_mean": np.mean(ac100, 0).tolist(),
            "mask_rate": float(m.mean()),
            "mask_rate_by_step_decile": [float(m[(st >= lo) & (st < lo + 2000)].mean()) for lo in range(0, 20000, 2000)],
            "mean_abs_y_scored": np.abs(y[m]).mean(0).tolist(), "mean_abs_y_unscored": np.abs(y[~m]).mean(0).tolist(),
            "std_y_scored": y[m].std(0).tolist(), "std_y_unscored": y[~m].std(0).tolist(),
            "feat_nan_frac_max": float((nan / n).max()),
            "feat_nan_cols": [feat[i] for i in np.where(nan > 0)[0]][:20],
            "feat_const_in_seq_frac": {feat[i]: round(const_in_seq[i] / min(max_rg, pf.num_row_groups), 3) for i in np.where(const_in_seq > 0)[0]},
            "feat_mean": dict(zip(feat, mean.round(4).tolist())),
            "feat_std": dict(zip(feat, std.round(4).tolist())),
            "feat_min": dict(zip(feat, mn.round(3).tolist())),
            "feat_max": dict(zip(feat, mx.round(3).tolist())),
        }
        return out

    res = {"valid": summarize(f"{PACK}/datasets/valid.parquet", 10**6, True),
           "train_sample": summarize(f"{PACK}/datasets/train.parquet", n_train, False)}
    txt = json.dumps(res, indent=1)
    with open("/data/eda.json", "w") as f:
        f.write(txt)
    vol.commit()
    return txt


@app.local_entrypoint()
def main(n_train: int = 300, detach: bool = False):
    if detach:  # modal run --detach modal/eda.py --detach ; fetch later: modal volume get wnn-connectome-data eda.json
        print("spawned", eda.spawn(n_train).object_id)
        return
    txt = eda.remote(n_train)
    with open("runs_eda.json", "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt[:6000])
