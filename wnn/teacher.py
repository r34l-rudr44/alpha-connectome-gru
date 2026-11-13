"""Future-aware teacher: a bidirectional GRU trained on full sequences (sees future features).

Its predictions on train/valid are dumped as soft labels for distillation into the causal student.
Usage (on Modal, see modal/teacher_modal.py):
  python -m wnn.teacher --train ... --valid ... --out /data/teacher/bi128 --hidden 128 --epochs 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wnn.data import SeqDataset, compute_feature_stats
from wnn.train import loss_fn, seed_all, weighted_pearson_np


class BiGRUTeacher(nn.Module):
    def __init__(self, n_features=112, hidden=128, layers=1, dropout=0.0, x_clip=10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))
        self.x_clip = x_clip
        self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(2 * hidden), nn.Linear(2 * hidden, 2))

    def set_stats(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean)); self.std.copy_(torch.as_tensor(std))

    def forward(self, x):
        x = (x.float() - self.mean) / self.std
        x = torch.nan_to_num(x, nan=0.0, posinf=self.x_clip, neginf=-self.x_clip).clamp(-self.x_clip, self.x_clip)
        out, _ = self.gru(x)
        return self.head(out)


def lookahead_dilations(K):
    """Kernel-2 dilations whose sum is K - 1, so the stack reads exactly K consecutive rows."""
    d, left = [], K - 1
    step = 1
    while left > 0:
        use = min(step, left); d.append(use); left -= use; step *= 2
    return d


class LookaheadTeacher(nn.Module):
    """Causal GRU plus a branch that reads the next K rows (t+1 .. t+K) and nothing beyond.

    The full-future teacher above reproduces the label (0.988), so it removes no noise. A K-row teacher outputs
    E[y | past, next K rows]: the same conditional mean given the past as y itself, but without the part of the label
    that depends on rows after t+K. Used only to label training data; never submitted.
    """

    def __init__(self, n_features=112, hidden=128, K=25, ch=64, x_clip=10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))
        self.x_clip, self.K = x_clip, K
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.emb = nn.Linear(2 * n_features, ch)
        self.dil = lookahead_dilations(K)
        self.convs = nn.ModuleList([nn.Conv1d(ch, 2 * ch, 2, dilation=d) for d in self.dil])
        self.mix = nn.Linear(ch, hidden)
        nn.init.zeros_(self.mix.weight); nn.init.zeros_(self.mix.bias)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))

    def set_stats(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean)); self.std.copy_(torch.as_tensor(std))

    def forward(self, x):
        x = (x.float() - self.mean) / self.std
        x = torch.nan_to_num(x, nan=0.0, posinf=self.x_clip, neginf=-self.x_clip).clamp(-self.x_clip, self.x_clip)
        g, _ = self.gru(x)
        dx = torch.cat([torch.zeros_like(x[:, :1]), x[:, 1:] - x[:, :-1]], 1)
        z = self.emb(torch.cat([x, dx], -1)).flip(1).transpose(1, 2)  # reversed time: a causal conv here looks ahead
        for conv, d in zip(self.convs, self.dil):
            a, b = conv(nn.functional.pad(z, (d, 0))).chunk(2, 1)
            z = z + torch.tanh(a) * torch.sigmoid(b)
        f = z.transpose(1, 2).flip(1)                                  # f[t] reads rows t .. t+K-1
        f = torch.cat([f[:, 1:], torch.zeros_like(f[:, :1])], 1)       # shift: rows t+1 .. t+K
        return self.head(g + self.mix(f))


@torch.no_grad()
def predict_all(model, ds, device, batch, workers, out_path):
    """Writes (n_seq, T, 2) float16 predictions in dataset order to out_path (npy memmap)."""
    dl = DataLoader(ds, batch_size=batch, num_workers=workers)
    T = ds[0][0].shape[0]
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(len(ds), T, 2))
    i = 0
    model.eval()
    for X, *_ in dl:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            p = model(X.to(device))
        p = p.float().cpu().numpy().astype(np.float16)
        arr[i:i + len(p)] = p; i += len(p)
    arr.flush(); del arr
    model.train()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True); p.add_argument("--valid", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hidden", type=int, default=128); p.add_argument("--layers", type=int, default=1)
    p.add_argument("--epochs", type=int, default=2); p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train", type=int, default=0); p.add_argument("--max-valid", type=int, default=0)
    p.add_argument("--lookahead", type=int, default=0, help="K > 0: causal GRU + a branch reading the next K rows (0 = bidirectional GRU)")
    p.add_argument("--ch", type=int, default=64, help="channels of the look-ahead branch")
    args = p.parse_args()
    seed_all(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tr = SeqDataset(args.train); va = SeqDataset(args.valid, feat=tr.feat)
    if args.max_train: tr.indices = tr.indices[: args.max_train]
    if args.max_valid: va.indices = va.indices[: args.max_valid]
    mean, std = compute_feature_stats(tr, n=200, seed=args.seed)
    if args.lookahead > 0:
        m = LookaheadTeacher(len(tr.feat), args.hidden, args.lookahead, args.ch).to(dev)
    else:
        m = BiGRUTeacher(len(tr.feat), args.hidden, args.layers).to(dev)
    m.set_stats(mean, std)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.wd)
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=dev.type == "cuda")
    total = len(dl) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1)
    hist = []
    for ep in range(args.epochs):
        t0 = time.time(); run = 0.0; nb = 0
        for X, y, need, *_ in dl:
            X, y, need = X.to(dev, non_blocking=True), y.to(dev), need.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                pred = m(X)
            loss = loss_fn(pred.float(), y, need, "wmse")
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
            run += loss.item(); nb += 1
        # validate (teacher sees the future, so this is an upper bound, not a submission score)
        m.eval(); ys, ps = [], []
        with torch.no_grad():
            for X, y, need, scored, *_ in DataLoader(va, batch_size=args.batch, num_workers=args.workers):
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    pr = m(X.to(dev))
                mk = need & scored; ys.append(y[mk].numpy()); ps.append(pr.float().cpu()[mk].numpy())
        m.train()
        y_, p_ = np.concatenate(ys), np.concatenate(ps)
        r = {"epoch": ep, "train_loss": run / max(nb, 1), "secs": time.time() - t0,
             "score": 0.5 * (weighted_pearson_np(y_[:, 0], p_[:, 0]) + weighted_pearson_np(y_[:, 1], p_[:, 1]))}
        hist.append(r); print(json.dumps(r), flush=True)
    torch.save({"model": m.state_dict(), "args": vars(args), "mean": mean, "std": std, "hist": hist}, out / "teacher.pt")
    print("dumping soft labels...", flush=True)
    predict_all(m, tr, dev, args.batch, args.workers, str(out / "train.npy"))
    predict_all(m, va, dev, args.batch, args.workers, str(out / "valid.npy"))
    (out / "hist.json").write_text(json.dumps(hist, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
