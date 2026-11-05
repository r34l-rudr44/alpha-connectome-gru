"""Dump the prediction-average of causal GRU runs on train and valid as soft labels for ensemble distillation.

Writes /data/teacher/<tag>/{train,valid}.npy as float16 (n_row_groups, T, 2) in row-group order (same layout as
wnn/teacher.py), usable with `wnn.train --soft-dir /data/teacher/<tag>`.

Usage: modal run --detach modal/dump_preds.py --names gru128x2,g128_e6,g128_ema,g192_e4 --tag ens4 --detach
"""
import modal

app = modal.App("wnn-connectome-dump")
vol = modal.Volume.from_name("wnn-connectome-data")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy", "pyarrow")
         .add_local_python_source("wnn"))
PACK = "/data/wnn_connectome_starterpack"


@app.function(image=image, volumes={"/data": vol}, gpu="A10G", cpu=4, memory=16384, timeout=3 * 3600)
def dump(names: list[str], tag: str):
    import os
    import time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from wnn.data import SeqDataset
    from wnn.model import GRUModel

    dev = torch.device("cuda")
    models = []
    for name in names:
        ck = torch.load(f"/data/runs/{name}/best.pt", weights_only=False, map_location="cpu")
        a = ck["args"]
        m = GRUModel(len(ck["mean"]), a["hidden"], a["layers"], a["in_proj"], a["dropout"], a["x_clip"]).to(dev)
        m.load_state_dict(ck["model"]); m.eval(); models.append(m)
    out = f"/data/teacher/{tag}"; os.makedirs(out, exist_ok=True)
    for split in ("valid", "train"):
        ds = SeqDataset(f"{PACK}/datasets/{split}.parquet")
        dl = DataLoader(ds, batch_size=8, num_workers=4)
        arr = np.lib.format.open_memmap(f"{out}/{split}.npy", mode="w+", dtype=np.float16, shape=(len(ds), 20000, 2))
        i, t0 = 0, time.time()
        with torch.no_grad():
            for X, *_ in dl:
                X = X.to(dev)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    p = sum(m(X)[0].float() for m in models) / len(models)
                p = p.cpu().numpy().astype(np.float16)
                arr[i:i + len(p)] = p; i += len(p)
                if (i // 8) % 200 == 0:
                    print(f"{split} {i}/{len(ds)} {time.time() - t0:.0f}s", flush=True)
        arr.flush(); del arr
        print(split, "done", flush=True)
    vol.commit()
    return out


@app.local_entrypoint()
def main(names: str, tag: str = "ens", detach: bool = False):
    lst = names.split(",")
    if detach:
        print("spawned", dump.spawn(lst, tag).object_id); return
    print(dump.remote(lst, tag))
