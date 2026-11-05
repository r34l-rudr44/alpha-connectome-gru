"""Train the sliding-window Transformer teacher (wnn.local_model) with the same data/loss/eval as wnn.train.

Each 20k-step sequence is processed in `--chunk` segments prefixed by `--window` context steps; one optimizer step per
segment (like TBPTT, but attention re-reads the context so nothing is lost at the boundary).

Local smoke: python -m wnn.train_local --train sample/datasets/train_sample.parquet
             --valid sample/datasets/valid_sample.parquet --max-train 4 --batch 2 --epochs 1 --out runs/smoke_local
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wnn.data import SeqDataset, SyntheticDataset, compute_feature_stats
from wnn.local_model import LocalTransformer
from wnn.train import evaluate, get_parser, loss_fn, seed_all, soft_loss


def train(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        tr_ds, va_ds = SyntheticDataset(64, 2000), SyntheticDataset(16, 2000, seed=1000)
    else:
        soft_tr = f"{args.soft_dir}/train.npy" if args.soft_dir else None
        w_tr = f"{args.weight_dir}/train.npy" if args.weight_dir else None
        tr_ds = SeqDataset(args.train, feat=None, soft_path=soft_tr, weight_path=w_tr)
        if args.max_train:
            tr_ds.indices = tr_ds.indices[: args.max_train]
        va_ds = SeqDataset(args.valid, feat=tr_ds.feat)
        if args.max_valid:
            va_ds.indices = va_ds.indices[: args.max_valid]
    n_feat = tr_ds[0][0].shape[1]
    print(f"device={device} train_seqs={len(tr_ds)} valid_seqs={len(va_ds)} features={n_feat}", flush=True)

    mean, std = compute_feature_stats(tr_ds, n=args.stats_seqs, seed=args.seed)
    np.savez(out / "norm.npz", mean=mean, std=std)
    model = LocalTransformer(n_feat, args.hidden, args.layers, args.heads, args.window, args.ff, args.conv,
                             args.dropout, args.x_clip, args.chunk).to(device)
    model.set_stats(mean, std)
    print(f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    tr = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=device.type == "cuda", drop_last=True, persistent_workers=args.workers > 0)
    va = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    T = tr_ds[0][0].shape[0]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
    total = len(tr) * math.ceil(T / args.chunk) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1)
    use_amp = device.type == "cuda"

    best, hist = -1.0, []
    saved_args = dict(vars(args), arch="local")
    for ep in range(args.epochs):
        t0 = time.time(); run_loss = 0.0; nb = 0
        model.train()
        for X, y, need, scored, soft, rw in tr:
            X, y, need, soft = X.to(device), y.to(device), need.to(device), soft.to(device)
            mw = need.float()
            if args.weight_dir:
                mw = mw * (args.weight_floor + rw.to(device).clamp_min(0).pow(args.weight_pow))
            for lo, s, e in model.iter_chunks(T):
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    pred = model.segment(X[:, lo:e])[:, s - lo:].float()
                loss = args.hard_lambda * loss_fn(pred, y[:, s:e], mw[:, s:e], args.loss)
                if args.soft_dir and args.soft_lambda > 0:
                    loss = loss + args.soft_lambda * soft_loss(pred, soft[:, s:e], y[:, s:e], mw[:, s:e])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step(); sched.step()
                run_loss += loss.item(); nb += 1
            if args.log_every and (nb // math.ceil(T / args.chunk)) % args.log_every == 0:
                print(f"ep {ep} segs {nb} loss {run_loss / nb:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        r = evaluate(model, va, device, args.eval_batches)
        r.update(epoch=ep, train_loss=run_loss / max(nb, 1), secs=time.time() - t0)
        hist.append(r); print(json.dumps(r), flush=True)
        torch.save({"model": model.state_dict(), "args": saved_args, "mean": mean, "std": std, "hist": hist}, out / "last.pt")
        if r["score"] > best:
            best = r["score"]
            torch.save({"model": model.state_dict(), "args": saved_args, "mean": mean, "std": std, "hist": hist}, out / "best.pt")
    (out / "hist.json").write_text(json.dumps(hist, indent=1))
    print(f"best valid score {best:.5f}", flush=True)
    return best


def get_local_parser():
    p = get_parser()
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--ff", type=int, default=256)
    p.add_argument("--conv", type=int, default=5, help="causal depthwise conv kernel per block (0/1 = off)")
    p.add_argument("--chunk", type=int, default=1024)
    p.set_defaults(lr=1e-3, hidden=128, layers=2)
    return p


if __name__ == "__main__":
    train(get_local_parser().parse_args())
