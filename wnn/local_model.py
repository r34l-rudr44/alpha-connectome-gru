"""Sliding-window causal Transformer: a *teacher* that only looks at the last `window` steps.

Motivation (analysis/memory_probe.py): the trained GRU-128 gets 98.5% of its score from the last 100 steps and all of
it from the last ~500, so the task is local. A windowed attention model trains fully in parallel (no truncated BPTT),
can be much richer per step than a 128-d recurrent state, and is distilled into the deployable GRU afterwards.

forward(x (B,T,F) raw, h=None) -> (y (B,T,2), None): the sequence is processed in chunks of `chunk` steps, each
chunk prefixed with the previous `window` steps as context, so every output sees exactly `window` steps of history
(deeper layers: window per layer). Same procedure at train and inference time.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads, window, ff, conv, dropout):
        super().__init__()
        self.h, self.dh, self.window = heads, d // heads, window
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, conv, groups=d, padding=0) if conv > 1 else None
        self.k = conv
        self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.rel = nn.Parameter(torch.zeros(heads, window))  # additive bias per (head, distance)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)
        self._mask_cache = {}

    def mask(self, n, device, dtype):
        key = (n, device, dtype)
        if key not in self._mask_cache:
            i = torch.arange(n, device=device)
            dist = i[:, None] - i[None, :]  # (n,n) query - key
            self._mask_cache = {key: dist}  # keep only the latest geometry
        dist = self._mask_cache[key]
        allowed = (dist >= 0) & (dist < self.window)
        bias = self.rel[:, dist.clamp(0, self.window - 1)]  # (H,n,n)
        return torch.where(allowed[None], bias, torch.full_like(bias, float("-inf"))).to(dtype)

    def forward(self, x):
        B, n, d = x.shape
        z = self.ln1(x)
        if self.conv is not None:  # causal depthwise conv as a local mixer
            z = z + F.gelu(self.conv(F.pad(z.transpose(1, 2), (self.k - 1, 0))).transpose(1, 2))
        q, k, v = self.qkv(z).view(B, n, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)  # 3 x (B,H,n,dh)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask(n, x.device, q.dtype)[None])
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, n, d)))
        return x + self.drop(self.ff(self.ln2(x)))


class LocalTransformer(nn.Module):
    arch = "local"

    def __init__(self, n_features=112, d=128, layers=2, heads=4, window=256, ff=256, conv=5, dropout=0.0,
                 x_clip=10.0, chunk=1024):
        super().__init__()
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("std", torch.ones(n_features))
        self.x_clip, self.window, self.chunk = x_clip, window, chunk
        self.proj = nn.Linear(n_features, d)
        self.blocks = nn.ModuleList([Block(d, heads, window, ff, conv, dropout) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2))

    def set_stats(self, mean, std):
        self.mean.copy_(torch.as_tensor(mean)); self.std.copy_(torch.as_tensor(std))

    def normalize(self, x):
        x = (x.float() - self.mean) / self.std
        x = torch.nan_to_num(x, nan=0.0, posinf=self.x_clip, neginf=-self.x_clip)
        return x.clamp(-self.x_clip, self.x_clip)

    def segment(self, x_raw):
        """Run one segment (B,n,F) raw -> (B,n,2). Positions < window into the segment see a shorter history."""
        z = self.proj(self.normalize(x_raw))
        for b in self.blocks:
            z = b(z)
        return self.head(z)

    def iter_chunks(self, T):
        for s in range(0, T, self.chunk):
            e = min(s + self.chunk, T); lo = max(0, s - self.window)
            yield lo, s, e

    def forward(self, x, h=None):
        B, T, _ = x.shape
        out = torch.empty(B, T, 2, device=x.device, dtype=torch.float32)
        for lo, s, e in self.iter_chunks(T):
            out[:, s:e] = self.segment(x[:, lo:e])[:, s - lo:].float()
        return out, None

    def init_state(self, batch, device):
        return None


def build_from_ckpt(ck):
    """Instantiate GRUModel or LocalTransformer from a checkpoint dict (as saved by wnn.train / wnn.train_local)."""
    a = ck["args"]; n = len(ck["mean"])
    if a.get("arch") == "local":
        m = LocalTransformer(n, a["hidden"], a["layers"], a["heads"], a["window"], a["ff"], a["conv"], a["dropout"],
                             a["x_clip"], a["chunk"])
    else:
        from wnn.model import GRUModel

        m = GRUModel(n, a["hidden"], a["layers"], a["in_proj"], a["dropout"], a["x_clip"])
    m.load_state_dict(ck["model"]); m.eval()
    return m
