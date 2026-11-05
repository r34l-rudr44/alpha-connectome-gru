"""Train the GRU with truncated BPTT over 20k-step sequences, |y|-weighted loss, WP validation.

Runs locally (CPU, --synthetic for a smoke test) or on Modal (see modal/train_modal.py).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from wnn.data import SeqDataset, SyntheticDataset, compute_feature_stats
from wnn.infer import export_npz
from wnn.model import GRUModel, export_onnx

CLIP = 2.0


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def weighted_pearson_np(y, p):
    y = np.clip(y, -CLIP, CLIP); p = np.clip(p, -CLIP, CLIP); w = np.abs(y)
    sw = w.sum()
    if sw <= 0:
        return 0.0
    my, mp = (w * y).sum() / sw, (w * p).sum() / sw
    dy, dp = y - my, p - mp
    den = math.sqrt((w * dy * dy).sum() * (w * dp * dp).sum())
    return float((w * dy * dp).sum() / den) if den > 0 else 0.0


def loss_fn(pred, y, mask, kind="wmse"):
    """pred,y: (B,T,2); mask: (B,T) bool or float row weights. Weighted by |clipped y| like the metric."""
    yc = y.clamp(-CLIP, CLIP)
    w = yc.abs() * mask.unsqueeze(-1)
    if kind == "wmse":
        return (w * (pred - yc) ** 2).sum() / w.sum().clamp_min(1e-6)
    if kind == "whuber":
        return (w * F.huber_loss(pred, yc, reduction="none", delta=1.0)).sum() / w.sum().clamp_min(1e-6)
    if kind == "corr":  # 1 - weighted Pearson per target, batch-pooled
        losses = []
        for k in range(2):
            wk, yk, pk = w[..., k], yc[..., k], pred[..., k]
            sw = wk.sum().clamp_min(1e-6)
            my, mp = (wk * yk).sum() / sw, (wk * pk).sum() / sw
            dy, dp = yk - my, pk - mp
            den = torch.sqrt((wk * dy * dy).sum() * (wk * dp * dp).sum()).clamp_min(1e-6)
            losses.append(1 - (wk * dy * dp).sum() / den)
        return sum(losses) / 2
    raise ValueError(kind)


def soft_loss(pred, soft, y, mask):
    """Weighted MSE to the teacher's (clipped) prediction, weighted by |clipped true y| like the metric."""
    sc = soft.clamp(-CLIP, CLIP)
    w = y.clamp(-CLIP, CLIP).abs() * mask.unsqueeze(-1)
    return (w * (pred - sc) ** 2).sum() / w.sum().clamp_min(1e-6)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    ys, ps, ms, mn = [], [], [], []
    for b, (X, y, need, scored, *_) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        X = X.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred, _ = model(X)
        pred = pred.float().cpu()
        m = need & scored
        ys.append(y[m].numpy()); ps.append(pred[m].numpy())
        mn.append(y[need].numpy()); ms.append(pred[need].numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    yn, pn = np.concatenate(mn), np.concatenate(ms)
    r = {"t0": weighted_pearson_np(y[:, 0], p[:, 0]), "t1": weighted_pearson_np(y[:, 1], p[:, 1])}
    r["score"] = (r["t0"] + r["t1"]) / 2
    r["score_unmasked"] = (weighted_pearson_np(yn[:, 0], pn[:, 0]) + weighted_pearson_np(yn[:, 1], pn[:, 1])) / 2
    r["n"] = int(len(y))
    model.train()
    return r


def train(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True

    if args.synthetic:
        tr_ds, va_ds = SyntheticDataset(64, 2000, aug=args.aug), SyntheticDataset(16, 2000, seed=1000, aug=args.aug)
    else:
        soft_tr = f"{args.soft_dir}/train.npy" if args.soft_dir else None
        soft_va = f"{args.soft_dir}/valid.npy" if args.soft_dir else None
        w_tr = f"{args.weight_dir}/train.npy" if args.weight_dir else None
        w_va = f"{args.weight_dir}/valid.npy" if args.weight_dir else None
        tr_ds = SeqDataset(args.train, feat=None, aug=args.aug, soft_path=soft_tr, weight_path=w_tr)
        if args.max_train:
            tr_ds.indices = tr_ds.indices[: args.max_train]
        if args.holdout_frac > 0:  # train on train + (1-frac) of valid; select on the frac holdout
            from torch.utils.data import ConcatDataset
            from wnn.data import holdout_split

            rest, hold = holdout_split(args.valid, args.holdout_frac, args.holdout_seed)
            va_ds = SeqDataset(args.valid, indices=hold, feat=tr_ds.feat, aug=args.aug)
            tr_ds = ConcatDataset([tr_ds, SeqDataset(args.valid, indices=rest, feat=va_ds.feat, aug=args.aug,
                                                     soft_path=soft_va, weight_path=w_va)])
            print(f"holdout: {len(hold)} valid seqs held out, {len(rest)} added to training", flush=True)
        else:
            va_ds = SeqDataset(args.valid, feat=tr_ds.feat, aug=args.aug)
        if args.max_valid:
            va_ds.indices = va_ds.indices[: args.max_valid]
    n_feat = tr_ds[0][0].shape[1]
    print(f"device={device} train_seqs={len(tr_ds)} valid_seqs={len(va_ds)} features={n_feat}", flush=True)

    mean, std = compute_feature_stats(tr_ds, n=args.stats_seqs, seed=args.seed)
    np.savez(out / "norm.npz", mean=mean, std=std)

    model = GRUModel(n_feat, args.hidden, args.layers, args.in_proj, args.dropout, args.x_clip).to(device)
    model.set_stats(mean, std)
    print(f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    tr = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=device.type == "cuda", drop_last=True, persistent_workers=args.workers > 0)
    va = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = len(tr) * math.ceil(tr_ds[0][0].shape[0] / args.tbptt)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.1)
    scaler_dtype = torch.bfloat16

    ema = None
    if args.ema_decay > 0:  # exponential moving average of weights, evaluated/saved alongside the raw weights
        import copy
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)

    def ema_update():
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(args.ema_decay).add_(pm.detach(), alpha=1 - args.ema_decay)

    best, hist, step = -1.0, [], 0
    for ep in range(args.epochs):
        t0 = time.time(); run_loss = 0.0; nb = 0
        for X, y, need, scored, soft, rw in tr:
            X, y, need = X.to(device, non_blocking=True), y.to(device, non_blocking=True), need.to(device)
            soft = soft.to(device, non_blocking=True)
            # per-step loss weights: need mask times (optional) row weights^pow, e.g. P(is_scored | x) from modal/mask_model.py
            mw = need.float()
            if args.weight_dir:
                rw = rw.to(device, non_blocking=True).clamp_min(0)
                mw = mw * (args.weight_floor + rw.pow(args.weight_pow))
            if args.target_smooth > 0:  # centered moving average of the labels (training only; eval uses raw y)
                k = args.target_smooth
                y = F.avg_pool1d(y.transpose(1, 2), kernel_size=2 * k + 1, stride=1, padding=k,
                                 count_include_pad=False).transpose(1, 2)
            B, T, _ = X.shape
            h = model.init_state(B, device)
            for s in range(0, T, args.tbptt):
                e = min(s + args.tbptt, T)
                with torch.autocast(device_type=device.type, dtype=scaler_dtype, enabled=device.type == "cuda"):
                    pred, h = model(X[:, s:e], h)
                loss = args.hard_lambda * loss_fn(pred.float(), y[:, s:e], mw[:, s:e], args.loss)
                if args.soft_dir and args.soft_lambda > 0:  # distillation: match the teacher's soft labels
                    loss = loss + args.soft_lambda * soft_loss(pred.float(), soft[:, s:e], y[:, s:e], mw[:, s:e])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step(); sched.step(); step += 1
                if ema is not None:
                    ema_update()
                h = h.detach()
                run_loss += loss.item(); nb += 1
            if args.log_every and nb % args.log_every == 0:
                print(f"ep {ep} batch {nb} loss {run_loss / nb:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        r = evaluate(model, va, device, args.eval_batches)
        r.update(epoch=ep, train_loss=run_loss / max(nb, 1), secs=time.time() - t0)
        cands = [("raw", model, r)]
        if ema is not None:
            re_ = evaluate(ema, va, device, args.eval_batches)
            r["ema_score"] = re_["score"]; r["ema_score_unmasked"] = re_["score_unmasked"]
            cands.append(("ema", ema, re_))
        hist.append(r); print(json.dumps(r), flush=True)
        torch.save({"model": model.state_dict(), "args": vars(args), "mean": mean, "std": std, "hist": hist}, out / "last.pt")
        for tag, mdl, rr in cands:
            if rr["score"] > best:
                best = rr["score"]
                torch.save({"model": mdl.state_dict(), "args": vars(args), "mean": mean, "std": std, "hist": hist, "which": tag}, out / "best.pt")
                export_npz(mdl, str(out / "weights.npz"), aug=args.aug)
                if args.onnx and not args.aug:
                    try:
                        export_onnx(mdl, str(out / "model.onnx")); mdl.to(device)
                    except Exception as e:  # onnx is optional; numba path is primary
                        print("onnx export failed:", e, flush=True)
    (out / "hist.json").write_text(json.dumps(hist, indent=1))
    print(f"best valid score {best:.5f}", flush=True)
    return best


def get_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="datasets/train.parquet")
    p.add_argument("--valid", default="datasets/valid.parquet")
    p.add_argument("--out", default="runs/gru")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-valid", type=int, default=0)
    p.add_argument("--stats-seqs", type=int, default=200)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--in-proj", type=int, default=0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--x-clip", type=float, default=10.0)
    p.add_argument("--loss", default="wmse", choices=["wmse", "whuber", "corr"])
    p.add_argument("--target-smooth", type=int, default=0, help="centered moving-average half-window on training labels")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--tbptt", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--eval-batches", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--onnx", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.0, help="weight EMA decay per step, e.g. 0.999; 0 = off")
    p.add_argument("--aug", action="store_true", help="multi-scale feature augmentation (wnn.features)")
    p.add_argument("--holdout-frac", type=float, default=0.0, help="train on train + valid, keeping this fraction of valid seqs for selection")
    p.add_argument("--holdout-seed", type=int, default=0)
    p.add_argument("--soft-dir", default="", help="dir with teacher train.npy/valid.npy soft labels (distillation)")
    p.add_argument("--soft-lambda", type=float, default=1.0)
    p.add_argument("--hard-lambda", type=float, default=1.0)
    p.add_argument("--weight-dir", default="", help="dir with train.npy/valid.npy per-step row weights, e.g. P(is_scored|x)")
    p.add_argument("--weight-pow", type=float, default=1.0, help="row weight exponent")
    p.add_argument("--weight-floor", type=float, default=0.0, help="added to row weight^pow (keeps unscored-looking rows in play)")
    return p


if __name__ == "__main__":
    train(get_parser().parse_args())
