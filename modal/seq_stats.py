"""Per-sequence statistics for the regime study (Modal CPU, cheap). Output: /data/seq_stats/{train,valid}.npz,
also returned to the caller and saved to runs/seq_stats/.

Per sequence: a0..a7 mean/std/first/last/change-rate, a7 n_unique, column means/stds (112 each), mean |dX| per
column group (i0/i1 x bid p, ask p, bid v, ask v, dp, dv), price change rates, target stats (where present).

Usage: modal run modal/seq_stats.py
Local smoke: PYTHONPATH=. python modal/seq_stats.py --local sample/datasets --out runs/seq_stats_smoke
"""
import modal

app = modal.App("wnn-connectome-seqstats")
vol = modal.Volume.from_name("wnn-connectome-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "pyarrow")
PACK = "/data/wnn_connectome_starterpack"
META = ("seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1")
WARM = 99


def _groups(feat):
    """Column index groups: (instrument, kind) -> indices. Names look like i0_bp3 / i0_ap3 / i0_bv3 / i0_av3 / i0_dp1 / i0_dv1."""
    import re

    g = {}
    for j, c in enumerate(feat):
        m = re.match(r"(i\d)_([a-z]+)\d*$", c)
        if m:
            g.setdefault(f"{m.group(1)}_{m.group(2)}", []).append(j)
    return g


def seq_stats(path, rg, feat, groups):
    import numpy as np
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    t = pf.read_row_group(rg)
    X = np.stack([t.column(c).to_numpy(zero_copy_only=False) for c in feat], 1).astype(np.float32)
    names = set(t.column_names)
    out = {}
    a = [j for j, c in enumerate(feat) if c.startswith("a")]
    A = X[:, a]
    out["a_mean"] = A.mean(0); out["a_std"] = A.std(0); out["a_first"] = A[0]; out["a_last"] = A[-1]
    out["a_change"] = (np.diff(A, axis=0) != 0).mean(0)
    out["a_nuniq"] = np.array([len(np.unique(A[:, k])) for k in range(A.shape[1])], np.float32)
    out["col_mean"] = X.mean(0); out["col_std"] = X.std(0)
    dX = np.abs(np.diff(X, axis=0))
    keys = sorted(groups)
    out["grp_absdx"] = np.array([dX[:, groups[k]].mean() for k in keys], np.float32)
    out["grp_change"] = np.array([(dX[:, groups[k]].sum(1) > 0).mean() for k in keys], np.float32)
    out["grp_mean"] = np.array([X[:, groups[k]].mean() for k in keys], np.float32)
    out["grp_std"] = np.array([X[:, groups[k]].std() for k in keys], np.float32)
    need = t.column("need_prediction").to_numpy().astype(bool)
    out["need_rate"] = np.array([need.mean()], np.float32)
    if "t0" in names:
        y = np.stack([t.column("t0").to_numpy(), t.column("t1").to_numpy()], 1).astype(np.float32)[WARM:]
        out["y_mean"] = y.mean(0); out["y_absmean"] = np.abs(y).mean(0); out["y_std"] = y.std(0)
        out["y_zero"] = (y == 0).mean(0); out["y_change"] = (np.diff(y, axis=0) != 0).mean(0)
        out["y_autocorr20"] = np.array([np.corrcoef(y[:-20, k], y[20:, k])[0, 1] for k in range(2)], np.float32)
    if "is_scored" in names:
        sc = t.column("is_scored").to_numpy().astype(bool)
        out["scored_rate"] = np.array([sc[WARM:].mean()], np.float32)
        runs = np.diff(np.flatnonzero(np.diff(sc.astype(np.int8)) != 0))
        out["scored_runlen"] = np.array([runs.mean() if len(runs) else 0], np.float32)
    return rg, out


def build(path, out_file, workers=8):
    import time
    from multiprocessing import Pool
    import numpy as np
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path); n = pf.num_row_groups
    feat = [c for c in pf.schema_arrow.names if c not in META]
    groups = _groups(feat)
    t0 = time.time(); res = [None] * n
    with Pool(workers) as pool:
        for i, (rg, out) in enumerate(pool.starmap(seq_stats, [(path, rg, feat, groups) for rg in range(n)], chunksize=8)):
            res[rg] = out
            if i % 1000 == 0:
                print(f"{path} {i}/{n} {time.time() - t0:.0f}s", flush=True)
    keys = res[0].keys()
    arrays = {k: np.stack([r[k] for r in res]) for k in keys}
    arrays["feat"] = np.array(feat); arrays["group_keys"] = np.array(sorted(groups))
    np.savez(out_file, **arrays)
    print(f"{path}: {n} seqs, {time.time() - t0:.0f}s -> {out_file}", flush=True)
    return arrays


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=32768, timeout=2 * 3600)
def run():
    import os

    os.makedirs("/data/seq_stats", exist_ok=True)
    files = {}
    for split in ("valid", "train"):
        build(f"{PACK}/datasets/{split}.parquet", f"/data/seq_stats/{split}.npz", workers=8)
        files[split] = open(f"/data/seq_stats/{split}.npz", "rb").read()
    vol.commit()
    return files


@app.local_entrypoint()
def main(detach: bool = False):
    import pathlib

    if detach:  # fetch later: modal volume get wnn-connectome-data seq_stats/<split>.npz runs/seq_stats/
        print("spawned", run.spawn().object_id); return
    files = run.remote()
    d = pathlib.Path("runs/seq_stats"); d.mkdir(parents=True, exist_ok=True)
    for split, b in files.items():
        (d / f"{split}.npz").write_bytes(b); print("saved", d / f"{split}.npz", len(b))


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser(); ap.add_argument("--local", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for split in ("valid", "train"):
        arr = build(f"{a.local}/{split}_sample.parquet", f"{a.out}/{split}.npz", workers=2)
        print(split, {k: v.shape for k, v in arr.items()})
