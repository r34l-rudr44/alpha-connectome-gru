"""Step-wise inference for GRUModel in numba (fallback: numpy). Mirrors wnn.model.GRUModel exactly.

Weights are stored as an .npz (see export_npz) so the submission needs no torch.
"""
from __future__ import annotations

import math

import numpy as np

import os

try:
    if os.environ.get("WNN_NO_NUMBA"):
        raise ImportError("disabled")
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f

        return deco if not (a and callable(a[0])) else a[0]


@njit(fastmath=True, cache=True)
def _gru_layer(x, h, WiT, WhT, bi, bh, gi, gh):
    H = h.shape[0]
    F = x.shape[0]
    for j in range(3 * H):
        s = bi[j]
        for k in range(F):
            s += x[k] * WiT[j, k]
        gi[j] = s
        s = bh[j]
        for k in range(H):
            s += h[k] * WhT[j, k]
        gh[j] = s
    for j in range(H):
        r = 1.0 / (1.0 + math.exp(-(gi[j] + gh[j])))
        z = 1.0 / (1.0 + math.exp(-(gi[H + j] + gh[H + j])))
        n = math.tanh(gi[2 * H + j] + r * gh[2 * H + j])
        h[j] = (1.0 - z) * n + z * h[j]


@njit(fastmath=True, cache=True)
def _normalize(x, mean, std, clip, out):
    for k in range(x.shape[0]):
        v = (x[k] - mean[k]) / std[k]
        if v != v:
            v = 0.0
        if v > clip:
            v = clip
        elif v < -clip:
            v = -clip
        out[k] = v


@njit(fastmath=True, cache=True)
def _head(h, ln_g, ln_b, WlT, bl, out, use_ln, tmp):
    H = h.shape[0]
    if use_ln:
        m = 0.0
        for k in range(H):
            m += h[k]
        m /= H
        v = 0.0
        for k in range(H):
            d = h[k] - m
            v += d * d
        inv = 1.0 / math.sqrt(v / H + 1e-5)
        for k in range(H):
            tmp[k] = (h[k] - m) * inv * ln_g[k] + ln_b[k]
    else:
        for k in range(H):
            tmp[k] = h[k]
    for o in range(out.shape[0]):
        s = bl[o]
        for k in range(H):
            s += tmp[k] * WlT[o, k]
        out[o] = s


def export_npz(model, path: str, aug: bool = False, bank: str = ""):
    """Dump GRUModel weights in the layout the kernels expect. bank: EMA-bank spec (wnn.features.parse_bank)."""
    from .features import N_BLOCKS, parse_bank

    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()}
    n_in = len(sd["mean"])
    d = {"mean": sd["mean"], "std": sd["std"], "x_clip": np.float32(model.x_clip),
         "layers": np.int32(model.layers), "hidden": np.int32(model.hidden),
         "aug": np.int32(1 if aug else 0), "n_raw": np.int32(n_in // N_BLOCKS if aug else n_in)}
    if bank:
        alphas, scaled = parse_bank(bank)
        d["bank_alphas"] = alphas; d["bank_scaled"] = np.int32(1 if scaled else 0)
        d["n_raw"] = np.int32(n_in // (1 + len(alphas)))
    assert model.proj is None, "input projection not supported by the kernel yet"
    for l in range(model.layers):
        d[f"WiT{l}"] = np.ascontiguousarray(sd[f"gru.weight_ih_l{l}"])  # (3H, in)
        d[f"WhT{l}"] = np.ascontiguousarray(sd[f"gru.weight_hh_l{l}"])  # (3H, H)
        d[f"bi{l}"] = sd[f"gru.bias_ih_l{l}"]
        d[f"bh{l}"] = sd[f"gru.bias_hh_l{l}"]
    d["use_ln"] = np.int32(1)
    d["ln_g"], d["ln_b"] = sd["head.0.weight"], sd["head.0.bias"]
    d["WlT"], d["bl"] = np.ascontiguousarray(sd["head.1.weight"]), sd["head.1.bias"]
    np.savez(path, **d)
    return path


class StepModel:
    """Online predictor: call step(x) per row; reset() at sequence start."""

    def __init__(self, npz_path: str):
        z = np.load(npz_path)
        self.mean, self.std, self.clip = z["mean"], z["std"], float(z["x_clip"])
        self.L, self.H = int(z["layers"]), int(z["hidden"])
        self.W = [(z[f"WiT{l}"], z[f"WhT{l}"], z[f"bi{l}"], z[f"bh{l}"]) for l in range(self.L)]
        self.use_ln = bool(int(z["use_ln"]))
        self.ln_g, self.ln_b, self.WlT, self.bl = z["ln_g"], z["ln_b"], z["WlT"], z["bl"]
        f = np.float32
        self.aug = bool(int(z["aug"])) if "aug" in z else False
        self.n_raw = int(z["n_raw"]) if "n_raw" in z else len(self.mean)
        self.fstate = np.zeros((4, self.n_raw), f)
        self.bank = "bank_alphas" in z
        if self.bank:
            self.alphas = np.ascontiguousarray(z["bank_alphas"], dtype=f); self.scaled = bool(int(z["bank_scaled"]))
            self.ema = np.zeros((len(self.alphas), self.n_raw), f); self.sc = np.zeros((len(self.alphas), self.n_raw), f)
        self.xa = np.zeros(len(self.mean), f)
        self.first = True
        self.xn = np.zeros(len(self.mean), f)
        self.h = [np.zeros(self.H, f) for _ in range(self.L)]
        self.gi = np.zeros(3 * self.H, f); self.gh = np.zeros(3 * self.H, f)
        self.tmp = np.zeros(self.H, f); self.out = np.zeros(2, f)
        self.reset()
        self.step(np.zeros(self.n_raw, f)); self.reset()  # trigger JIT

    def reset(self):
        for h in self.h:
            h[:] = 0
        self.first = True

    def _augment(self, x):
        if self.bank:
            if HAVE_NUMBA:
                from .features import bank_step as _bank
            else:
                from .features import bank_step_numpy as _bank
            _bank(x, self.alphas, self.scaled, self.ema, self.sc, self.first, self.xa)
            self.first = False
            return self.xa
        if not self.aug:
            return x
        if HAVE_NUMBA:
            from .features import augment_step as _aug
        else:
            from .features import augment_step_numpy as _aug
        _aug(x, self.fstate, self.first, self.xa)
        self.first = False
        return self.xa

    def _step_numpy(self, x: np.ndarray) -> np.ndarray:
        H = self.H
        x = self._augment(x)
        inp = np.clip(np.nan_to_num((x - self.mean) / self.std, nan=0.0), -self.clip, self.clip)
        for l, (WiT, WhT, bi, bh) in enumerate(self.W):
            h = self.h[l]
            gi = WiT @ inp + bi
            gh = WhT @ h + bh
            r = 1.0 / (1.0 + np.exp(-(gi[:H] + gh[:H])))
            z = 1.0 / (1.0 + np.exp(-(gi[H:2 * H] + gh[H:2 * H])))
            n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
            h[:] = (1.0 - z) * n + z * h
            inp = h
        if self.use_ln:
            m = inp.mean(); v = ((inp - m) ** 2).mean()
            inp = (inp - m) / np.sqrt(v + 1e-5) * self.ln_g + self.ln_b
        self.out[:] = self.WlT @ inp + self.bl
        return self.out

    def step(self, x: np.ndarray) -> np.ndarray:
        if not HAVE_NUMBA:
            return self._step_numpy(x)
        x = self._augment(x)
        _normalize(x, self.mean, self.std, self.clip, self.xn)
        inp = self.xn
        for l, (WiT, WhT, bi, bh) in enumerate(self.W):
            _gru_layer(inp, self.h[l], WiT, WhT, bi, bh, self.gi, self.gh)
            inp = self.h[l]
        _head(inp, self.ln_g, self.ln_b, self.WlT, self.bl, self.out, self.use_ln, self.tmp)
        return self.out
