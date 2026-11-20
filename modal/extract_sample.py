"""Pull a small slice of the pack from the Modal volume to the local machine: baseline/, utils.py, docs,
requirements.txt, METRIC.md, README.md and the first N validation sequences as a parquet.

Usage: modal run modal/extract_sample.py --n-valid 20
Writes to ./sample/ (gitignored).
"""
import modal

app = modal.App("wnn-connectome-sample")
vol = modal.Volume.from_name("wnn-connectome-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install("pyarrow")
PACK = "/data/wnn_connectome_starterpack"


@app.function(image=image, volumes={"/data": vol}, timeout=1800, cpu=2, memory=8192)
def extract(n_valid: int, n_train: int):
    import io
    import os
    import pyarrow.parquet as pq

    files = {}
    for root, _, fns in os.walk(PACK):
        for fn in fns:
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, PACK)
            if rel.startswith("datasets/"):
                continue
            files[rel] = open(p, "rb").read()
    for name, n in (("valid", n_valid), ("train", n_train)):
        pf = pq.ParquetFile(f"{PACK}/datasets/{name}.parquet")
        buf = io.BytesIO()
        w = pq.ParquetWriter(buf, pf.schema_arrow, compression="zstd")
        for rg in range(min(n, pf.num_row_groups)):
            w.write_table(pf.read_row_group(rg))
        w.close()
        files[f"datasets/{name}_sample.parquet"] = buf.getvalue()
        files[f"datasets/{name}_meta.txt"] = f"row_groups={pf.num_row_groups} rows={pf.metadata.num_rows}\n".encode()
    return files


@app.local_entrypoint()
def main(n_valid: int = 20, n_train: int = 20):
    import pathlib

    files = extract.remote(n_valid, n_train)
    out = pathlib.Path("sample")
    for rel, b in files.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
        print(f"{rel}: {len(b) / 1e6:.2f} MB")
