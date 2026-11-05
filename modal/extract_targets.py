"""Extract valid targets + scoring mask from the volume as a compact npz (for local, CPU-only prediction analysis).

Writes /data/valid_targets.npz: y (n_rg, 20000, 2) float32, scored (n_rg, 20000) bool.
Usage: modal run modal/extract_targets.py ; then modal volume get wnn-connectome-data valid_targets.npz runs/
"""
import modal

app = modal.App("wnn-connectome-targets")
vol = modal.Volume.from_name("wnn-connectome-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "pyarrow")
PACK = "/data/wnn_connectome_starterpack"


@app.function(image=image, volumes={"/data": vol}, cpu=2, memory=8192, timeout=3600)
def run():
    import numpy as np
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(f"{PACK}/datasets/valid.parquet"); n = pf.num_row_groups
    y = np.empty((n, 20000, 2), np.float32); sc = np.empty((n, 20000), bool)
    for rg in range(n):
        t = pf.read_row_group(rg, columns=["t0", "t1", "is_scored"])
        y[rg, :, 0] = t.column("t0").to_numpy(); y[rg, :, 1] = t.column("t1").to_numpy()
        sc[rg] = t.column("is_scored").to_numpy()
    np.savez("/data/valid_targets.npz", y=y, scored=sc)
    vol.commit()
    return n


@app.local_entrypoint()
def main():
    print("row groups:", run.remote())
