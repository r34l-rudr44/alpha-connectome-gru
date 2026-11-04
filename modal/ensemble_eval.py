"""Evaluate single models and prediction-averaging ensembles on the full validation set (GPU).

Usage: modal run --detach modal/ensemble_eval.py --names gru128,g128_e6,... --detach
Writes /data/ensemble_eval.json (per-model preds are cached in /data/preds/<name>.npy).
"""
import modal

app = modal.App("wnn-connectome-ens")
vol = modal.Volume.from_name("wnn-connectome-data")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy", "pyarrow")
         .add_local_python_source("wnn"))
PACK = "/data/wnn_connectome_starterpack"


@app.function(image=image, volumes={"/data": vol}, gpu="A10G", cpu=4, memory=16384, timeout=3 * 3600)
def evaluate(names: list[str], holdout_frac: float = 0.0, holdout_seed: int = 0):
    import itertools
    import json
    import os
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from wnn.data import SeqDataset, holdout_split
    from wnn.model import GRUModel
    from wnn.train import weighted_pearson_np

    dev = torch.device("cuda")
    vpath = f"{PACK}/datasets/valid.parquet"
    idx = holdout_split(vpath, holdout_frac, holdout_seed)[1] if holdout_frac > 0 else None
    tag = f"_h{holdout_frac}_{holdout_seed}" if holdout_frac > 0 else ""
    ds = SeqDataset(vpath, indices=idx)
    dl = DataLoader(ds, batch_size=8, num_workers=4)  # batch 64 OOMs: cuDNN GRU workspace over 20k steps
    os.makedirs("/data/preds", exist_ok=True)
    Y = M = None
    preds = {}
    for name in names:
        cache = f"/data/preds/{name}{tag}.npy"
        if os.path.exists(cache) and Y is not None:
            preds[name] = np.load(cache); continue
        ck = torch.load(f"/data/runs/{name}/best.pt", weights_only=False, map_location="cpu")
        a = ck["args"]
        m = GRUModel(len(ck["mean"]), a["hidden"], a["layers"], a["in_proj"], a["dropout"], a["x_clip"]).to(dev)
        m.load_state_dict(ck["model"]); m.eval()
        P, Ys, Ms = [], [], []
        with torch.no_grad():
            for X, y, need, scored in dl:
                p, _ = m(X.to(dev)); mask = need & scored
                P.append(p.float().cpu()[mask].numpy()); Ys.append(y[mask].numpy()); Ms.append(mask.numpy())
        preds[name] = np.concatenate(P); np.save(cache, preds[name])
        if Y is None:
            Y = np.concatenate(Ys); np.save(f"/data/preds/_Y{tag}.npy", Y)
        print(name, "done", flush=True)
    if Y is None:
        Y = np.load(f"/data/preds/_Y{tag}.npy")

    def wp(p):
        return 0.5 * (weighted_pearson_np(Y[:, 0], p[:, 0]) + weighted_pearson_np(Y[:, 1], p[:, 1]))

    res = {"single": {n: wp(p) for n, p in preds.items()}}
    res["all"] = wp(np.mean([preds[n] for n in names], 0))
    res["pairs"] = {f"{a}+{b}": wp((preds[a] + preds[b]) / 2) for a, b in itertools.combinations(names, 2)}
    # greedy forward selection
    chosen, best = [], -1
    while True:
        cand = [(wp(np.mean([preds[n] for n in chosen + [c]], 0)), c) for c in names if c not in chosen]
        if not cand or max(cand)[0] <= best + 1e-5:
            break
        best, c = max(cand); chosen.append(c)
    res["greedy"] = {"chosen": chosen, "score": best}
    txt = json.dumps(res, indent=1)
    open(f"/data/ensemble_eval{tag}.json", "w").write(txt); vol.commit()
    print(txt)
    return txt


@app.local_entrypoint()
def main(names: str, holdout_frac: float = 0.0, holdout_seed: int = 0, detach: bool = False):
    lst = names.split(",")
    if detach:
        print("spawned", evaluate.spawn(lst, holdout_frac, holdout_seed).object_id); return
    print(evaluate.remote(lst, holdout_frac, holdout_seed))
