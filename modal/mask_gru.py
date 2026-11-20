"""Sequence mask model: a GRU for P(is_scored | sequence), trained on 80% of valid, scored on the 20% holdout.

The mask is only needed as a *training* row weight (never at test time), so the model may be bidirectional.
Inputs per step: normalised X and dX (224 dims). GRU -> logit; BCE on need rows (step >= 99).
Outputs /data/maskw2_<tag>/{train,valid}.npy float16 (n_row_groups, T) like modal/mask_model.py, plus info.json
with holdout AUC, the old LightGBM AUC on the same holdout rows, P quantiles and precision/recall at top-k fractions.

Usage: modal run --detach modal/mask_gru.py --tag bi --bi --detach
Local smoke: PYTHONPATH=. python modal/mask_gru.py --local sample/datasets --out runs/maskw2_smoke --epochs 2 --hidden 16
"""
import modal

app = modal.App("wnn-connectome-maskgru")
vol = modal.Volume.from_name("wnn-connectome-data")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy", "pyarrow", "scikit-learn")
         .add_local_python_source("wnn"))
PACK = "/data/wnn_connectome_starterpack"
WARM = 99


def build(train_path, valid_path, out_dir, old_dir=None, hidden=128, layers=1, bi=True, epochs=8, batch=16,
          tbptt=2000, lr=2e-3, holdout_frac=0.2, holdout_seed=0, workers=4, max_fit=0, dump_train=True):
    import json
    import os
    import time
    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader
    from wnn.data import SeqDataset, holdout_split, compute_feature_stats

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)
    rest, hold = holdout_split(valid_path, holdout_frac, holdout_seed)
    if max_fit:
        rest = rest[:max_fit]
    fit_ds = SeqDataset(valid_path, indices=rest); hold_ds = SeqDataset(valid_path, indices=hold, feat=fit_ds.feat)
    print(f"device={dev} fit={len(fit_ds)} hold={len(hold_ds)} bi={bi} hidden={hidden}", flush=True)
    mean, std = compute_feature_stats(fit_ds, n=min(200, len(fit_ds)))
    mean_t = torch.tensor(mean, device=dev); std_t = torch.tensor(std, device=dev)
    # dX scale: std of first differences on the stats subset
    dstd = []
    for i in range(min(50, len(fit_ds))):
        X = fit_ds[i][0].numpy(); dstd.append(np.diff(X, axis=0).std(0))
    dstd_t = torch.tensor(np.maximum(np.mean(dstd, 0), 1e-6), device=dev)

    def feats(X):  # X (B,T,F) raw -> (B,T,2F) normalised X and dX, clipped
        Xn = ((X - mean_t) / std_t).clamp(-10, 10)
        dX = torch.diff(X, dim=1, prepend=X[:, :1])
        dXn = (dX / dstd_t).clamp(-10, 10)
        return torch.cat([Xn, dXn], -1)

    class MaskGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(2 * len(mean), hidden, num_layers=layers, batch_first=True, bidirectional=bi)
            self.head = nn.Linear(hidden * (2 if bi else 1), 1)

        def forward(self, x, h=None):
            o, h = self.gru(x, h)
            return self.head(o).squeeze(-1), h

    model = MaskGRU().to(dev)
    print(f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    fit_dl = DataLoader(fit_ds, batch_size=batch, shuffle=True, num_workers=workers, drop_last=True, persistent_workers=workers > 0)
    hold_dl = DataLoader(hold_ds, batch_size=batch, num_workers=workers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    T = fit_ds[0][0].shape[0]
    chunks = -(-T // tbptt) if not bi else 1  # bidirectional: whole sequence at once
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=len(fit_dl) * chunks * epochs, pct_start=0.1)

    @torch.no_grad()
    def predict(dl):
        model.eval(); P, Y, N = [], [], []
        for X, y, need, scored, *_ in dl:
            X = X.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logit, _ = model(feats(X))
            P.append(torch.sigmoid(logit.float()).cpu().numpy()); Y.append(scored.numpy()); N.append(need.numpy())
        model.train()
        return np.concatenate(P), np.concatenate(Y), np.concatenate(N)

    best_auc, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        t0 = time.time(); tot = 0.0; nb = 0
        for X, y, need, scored, *_ in fit_dl:
            X, need, scored = X.to(dev), need.to(dev), scored.to(dev).float()
            Fx = feats(X)
            if bi:
                segs = [(0, T)]
            else:
                segs = [(s, min(s + tbptt, T)) for s in range(0, T, tbptt)]
            h = None
            for s, e in segs:
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    logit, h = model(Fx[:, s:e], h)
                m = need[:, s:e]
                loss = nn.functional.binary_cross_entropy_with_logits(logit.float()[m], scored[:, s:e][m])
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
                if h is not None:
                    h = h.detach()
                tot += loss.item(); nb += 1
        P, Y, N = predict(hold_dl)
        m = N.copy(); m[:, :WARM] = False
        auc = float(roc_auc_score(Y[m], P[m]))
        hist.append({"epoch": ep, "loss": tot / max(nb, 1), "auc": auc, "secs": time.time() - t0})
        print(json.dumps(hist[-1]), flush=True)
        if auc > best_auc:
            best_auc = auc; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    torch.save({"model": best_state, "mean": mean, "std": std, "dstd": dstd_t.cpu().numpy(), "hidden": hidden, "layers": layers, "bi": bi, "hist": hist}, f"{out_dir}/mask_gru.pt")

    info = {"auc_holdout": best_auc, "hist": hist, "bi": bi, "hidden": hidden, "n_fit": len(fit_ds), "n_hold": len(hold_ds)}
    # holdout diagnostics: old LightGBM weights on the same rows, P quantiles, precision/recall at top-k fractions
    P, Y, N = predict(hold_dl); m = N.copy(); m[:, :WARM] = False
    p, yv = P[m], Y[m]
    if old_dir and os.path.exists(f"{old_dir}/valid.npy"):
        old = np.load(f"{old_dir}/valid.npy", mmap_mode="r")
        po = np.asarray(old[hold], dtype=np.float32)[m]
        info["auc_old_lightgbm_same_rows"] = float(roc_auc_score(yv, po))
    qs = [0.5, 0.75, 0.9, 0.95]
    info["p_quantiles_need_rows"] = {str(q): float(np.quantile(p, q)) for q in qs}
    info["scored_rate"] = float(yv.mean())
    order = np.argsort(-p); cum = np.cumsum(yv[order])
    info["topk"] = {}
    for frac in (0.1, 0.15, 0.2, 0.25, 0.3, 0.5):
        k = int(frac * len(p)); info["topk"][str(frac)] = {"precision": float(cum[k - 1] / k), "recall": float(cum[k - 1] / yv.sum()), "p_thresh": float(p[order[k - 1]])}
    # calibration: mean P vs scored rate in P deciles
    bins = np.quantile(p, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(bins, p) - 1, 0, 9)
    info["calibration_deciles"] = [[float(p[b == i].mean()), float(yv[b == i].mean())] for i in range(10)]
    print(json.dumps({k: v for k, v in info.items() if k != "hist"}, indent=1), flush=True)

    def dump(path, name):
        ds = SeqDataset(path, feat=fit_ds.feat); dl = DataLoader(ds, batch_size=batch, num_workers=workers)
        arr = np.lib.format.open_memmap(f"{out_dir}/{name}.npy", mode="w+", dtype=np.float16, shape=(len(ds), T))
        i, t1 = 0, time.time()
        with torch.no_grad():
            for X, *_ in dl:
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    logit, _ = model(feats(X.to(dev)))
                pb = torch.sigmoid(logit.float()).cpu().numpy().astype(np.float16)
                arr[i:i + len(pb)] = pb; i += len(pb)
                if (i // batch) % 100 == 0:
                    print(f"{name} {i}/{len(ds)} {time.time() - t1:.0f}s", flush=True)
        arr.flush(); mp = float(np.asarray(arr[:, WARM:], dtype=np.float32).mean()); del arr
        return mp

    info["mean_p_valid"] = dump(valid_path, "valid")
    if dump_train:
        info["mean_p_train"] = dump(train_path, "train")
    json.dump(info, open(f"{out_dir}/info.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in info.items() if k != "hist"}), flush=True)
    return info


@app.function(image=image, volumes={"/data": vol}, gpu="A10G", cpu=4, memory=24576, timeout=4 * 3600)
def run(tag: str, bi: bool, hidden: int, layers: int, epochs: int):
    info = build(f"{PACK}/datasets/train.parquet", f"{PACK}/datasets/valid.parquet", f"/data/maskw2_{tag}",
                 old_dir="/data/maskw", hidden=hidden, layers=layers, bi=bi, epochs=epochs, workers=4)
    vol.commit()
    return {k: v for k, v in info.items() if k != "hist"}


@app.local_entrypoint()
def main(tag: str = "bi", bi: bool = False, hidden: int = 128, layers: int = 1, epochs: int = 8, detach: bool = False):
    if detach:
        print("spawned", run.spawn(tag, bi, hidden, layers, epochs).object_id); return
    print(run.remote(tag, bi, hidden, layers, epochs))


if __name__ == "__main__":  # local smoke on the sample parquets
    import argparse

    ap = argparse.ArgumentParser(); ap.add_argument("--local", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--hidden", type=int, default=16); ap.add_argument("--bi", action="store_true")
    a = ap.parse_args()
    build(f"{a.local}/train_sample.parquet", f"{a.local}/valid_sample.parquet", a.out, old_dir=None, hidden=a.hidden,
          bi=a.bi, epochs=a.epochs, batch=2, tbptt=2000, workers=0, dump_train=False)
