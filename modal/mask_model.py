"""Learn P(is_scored | current features) on valid with LightGBM and dump per-step weights for train/valid.

The scoring mask is not random (scored rows have larger |y| and 2x larger order-book changes; causal AUC ~0.84 on the
sample), so training the GRU with these weights focuses it on the scored distribution.
Outputs /data/maskw/{train,valid}.npy as float16 (n_row_groups, T) plus /data/maskw/info.json.

Usage: modal run --detach modal/mask_model.py --detach            (CPU only)
Local smoke: python modal/mask_model.py --local sample/datasets --out runs/maskw_smoke
"""
import modal

app = modal.App("wnn-connectome-mask")
vol = modal.Volume.from_name("wnn-connectome-data")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("numpy", "pyarrow", "lightgbm", "scikit-learn"))
PACK = "/data/wnn_connectome_starterpack"
META = ("seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1")


def feature_columns(path):  # torch-free copies of wnn.data helpers
    import pyarrow.parquet as pq

    return [c for c in pq.ParquetFile(path).schema_arrow.names if c not in META]


def read_sequence(pf, rg, feat):
    import numpy as np

    t = pf.read_row_group(rg)
    X = np.stack([t.column(c).to_numpy(zero_copy_only=False) for c in feat], 1).astype(np.float32, copy=False)
    need = t.column("need_prediction").to_numpy().astype(bool)
    sc = t.column("is_scored").to_numpy().astype(bool) if "is_scored" in t.column_names else need.copy()
    return X, None, need, sc


def _features(X):
    """Causal per-step features: X, dX, previous dX, and abs-change summaries (all computable at test time)."""
    import numpy as np

    dX = np.diff(X, axis=0, prepend=X[:1])
    dXp = np.roll(dX, 1, axis=0); dXp[0] = 0
    s = [np.abs(dX).sum(1, keepdims=True), np.abs(dX[:, :52]).sum(1, keepdims=True), np.abs(dX[:, 52:104]).sum(1, keepdims=True)]
    return np.concatenate([X, dX, dXp] + s, 1)


def build(train_path, valid_path, out_dir, n_fit=300, n_hold=100, stride=4, trees=200, threads=8):
    import json
    import os
    import time
    import numpy as np
    import pyarrow.parquet as pq
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    os.makedirs(out_dir, exist_ok=True)
    feat = feature_columns(train_path)
    pv = pq.ParquetFile(valid_path); nv = pv.num_row_groups
    rng = np.random.default_rng(0); perm = rng.permutation(nv)
    fit_idx, hold_idx = perm[:n_fit], perm[n_fit:n_fit + n_hold]

    def collect(idx, stride):
        Fs, ys = [], []
        for rg in idx:
            X, y, need, sc = read_sequence(pv, int(rg), feat)
            F = _features(X)[need][::stride]
            Fs.append(F); ys.append(sc[need][::stride])
        return np.concatenate(Fs), np.concatenate(ys)

    t0 = time.time()
    Xf, yf = collect(fit_idx, stride); Xh, yh = collect(hold_idx, stride)
    print(f"fit rows {len(Xf):,} pos {yf.mean():.4f}; hold rows {len(Xh):,}; {time.time() - t0:.0f}s", flush=True)
    m = lgb.LGBMClassifier(n_estimators=trees, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.5, min_child_samples=200, n_jobs=threads, verbose=-1)
    m.fit(Xf, yf, eval_set=[(Xh, yh)], eval_metric="auc", callbacks=[lgb.log_evaluation(50)])
    auc = roc_auc_score(yh, m.predict_proba(Xh)[:, 1])
    print(f"holdout AUC {auc:.4f}; fit {time.time() - t0:.0f}s", flush=True)
    del Xf, Xh

    def dump(path, name):
        pf = pq.ParquetFile(path); n = pf.num_row_groups
        arr = np.lib.format.open_memmap(f"{out_dir}/{name}.npy", mode="w+", dtype=np.float16, shape=(n, 20000))
        t1 = time.time()
        for rg in range(n):
            X, y, need, sc = read_sequence(pf, rg, feat)
            arr[rg] = m.predict_proba(_features(X), num_threads=threads)[:, 1].astype(np.float16)
            if rg % 500 == 0:
                print(f"{name} {rg}/{n} {time.time() - t1:.0f}s", flush=True)
        arr.flush(); mean_p = float(np.asarray(arr[:, 99:], dtype=np.float32).mean()); del arr
        return mean_p

    info = {"auc_holdout": auc, "n_fit": int(n_fit), "stride": stride, "trees": trees}
    info["mean_p_valid"] = dump(valid_path, "valid")
    info["mean_p_train"] = dump(train_path, "train")
    json.dump(info, open(f"{out_dir}/info.json", "w"), indent=1)
    print(json.dumps(info), flush=True)
    return info


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=32768, timeout=4 * 3600)
def run(n_fit: int = 300, trees: int = 200):
    info = build(f"{PACK}/datasets/train.parquet", f"{PACK}/datasets/valid.parquet", "/data/maskw",
                 n_fit=n_fit, trees=trees, threads=8)
    vol.commit()
    return info


@app.local_entrypoint()
def main(n_fit: int = 300, trees: int = 200, detach: bool = False):
    if detach:
        print("spawned", run.spawn(n_fit, trees).object_id); return
    print(run.remote(n_fit, trees))


if __name__ == "__main__":  # local smoke on the sample parquets
    import argparse

    ap = argparse.ArgumentParser(); ap.add_argument("--local", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(f"{a.local}/train_sample.parquet", f"{a.local}/valid_sample.parquet", a.out, n_fit=7, n_hold=3, stride=2,
          trees=50, threads=4)
