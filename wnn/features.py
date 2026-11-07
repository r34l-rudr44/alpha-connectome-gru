"""Causal multi-scale feature augmentation, identical in batch (training) and online (inference) form.

Per raw feature x (F=112):  x, dx = x - x_prev, EMA_a(x) for a in ALPHAS, EMA_v(|dx|)  ->  F * N_BLOCKS columns.
All state starts at the first row of the sequence (ema = x_0, dx = 0, vol = 0).
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (a and callable(a[0])) else a[0]

ALPHAS = (0.1, 0.01)
VOL_ALPHA = 0.05
N_BLOCKS = 1 + 1 + len(ALPHAS) + 1


def n_aug(F: int) -> int:
    return F * N_BLOCKS


@njit(fastmath=True, cache=True)
def augment_sequence(X, out):
    """X (T,F) float32 -> out (T, F*N_BLOCKS) float32."""
    T, F = X.shape
    e1 = np.empty(F, np.float32); e2 = np.empty(F, np.float32); vol = np.zeros(F, np.float32); prev = np.empty(F, np.float32)
    for k in range(F):
        e1[k] = X[0, k]; e2[k] = X[0, k]; prev[k] = X[0, k]
    a1, a2, av = ALPHAS[0], ALPHAS[1], VOL_ALPHA
    for t in range(T):
        for k in range(F):
            x = X[t, k]
            dx = x - prev[k]
            e1[k] = a1 * x + (1.0 - a1) * e1[k]
            e2[k] = a2 * x + (1.0 - a2) * e2[k]
            vol[k] = av * abs(dx) + (1.0 - av) * vol[k]
            out[t, k] = x
            out[t, F + k] = dx
            out[t, 2 * F + k] = e1[k]
            out[t, 3 * F + k] = e2[k]
            out[t, 4 * F + k] = vol[k]
            prev[k] = x


@njit(fastmath=True, cache=True)
def augment_step(x, state, first, out):
    """Online version. state (4,F): prev, e1, e2, vol. first: 1 on the first row of a sequence."""
    F = x.shape[0]
    a1, a2, av = ALPHAS[0], ALPHAS[1], VOL_ALPHA
    for k in range(F):
        v = x[k]
        if first:
            state[0, k] = v; state[1, k] = v; state[2, k] = v; state[3, k] = 0.0
        dx = v - state[0, k]
        state[1, k] = a1 * v + (1.0 - a1) * state[1, k]
        state[2, k] = a2 * v + (1.0 - a2) * state[2, k]
        state[3, k] = av * abs(dx) + (1.0 - av) * state[3, k]
        out[k] = v; out[F + k] = dx; out[2 * F + k] = state[1, k]; out[3 * F + k] = state[2, k]; out[4 * F + k] = state[3, k]
        state[0, k] = v


def augment_step_numpy(x, state, first, out):
    F = x.shape[0]
    a1, a2, av = ALPHAS[0], ALPHAS[1], VOL_ALPHA
    if first:
        state[0] = x; state[1] = x; state[2] = x; state[3] = 0.0
    dx = x - state[0]
    state[1] = a1 * x + (1.0 - a1) * state[1]
    state[2] = a2 * x + (1.0 - a2) * state[2]
    state[3] = av * np.abs(dx) + (1.0 - av) * state[3]
    out[:F] = x; out[F:2 * F] = dx; out[2 * F:3 * F] = state[1]; out[3 * F:4 * F] = state[2]; out[4 * F:] = state[3]
    state[0] = x


# ---------------------------------------------------------------------------------------------------------------
# Long-horizon causal EMA bank (multiscale EMA bank with half-lives up to 3000).
# Per raw feature x and half-life h (alpha = 1 - 0.5**(1/h)): dev_h = x - EMA_h(x); optionally scaled by
# EMA_h(|dev_h|) (per-sequence adaptive scale). Output columns: [x | dev_h1 | dev_h2 | ...].  State starts at the
# first row of the sequence (ema = x_0, scale ema = 0 -> scaled dev clipped at +-SCALE_CLIP).
# ---------------------------------------------------------------------------------------------------------------
SCALE_EPS = 1e-6
SCALE_CLIP = 10.0


def bank_alphas(half_lives) -> np.ndarray:
    return np.array([1.0 - 0.5 ** (1.0 / float(h)) for h in half_lives], np.float32)


def parse_bank(spec: str):
    """'100,300,1000,3000' or '100,300:s' (scaled) -> (alphas float32 array, scaled bool)."""
    if not spec:
        return None, False
    scaled = spec.endswith(":s")
    spec = spec[:-2] if scaled else spec
    return bank_alphas([float(s) for s in spec.split(",") if s]), scaled


def n_bank(F: int, n_alphas: int) -> int:
    return F * (1 + n_alphas)


@njit(fastmath=True, cache=True)
def bank_sequence(X, alphas, scaled, out):
    """X (T,F) float32, alphas (A,) -> out (T, F*(1+A))."""
    T, F = X.shape
    A = alphas.shape[0]
    ema = np.empty((A, F), np.float32); sc = np.zeros((A, F), np.float32)
    for a in range(A):
        for k in range(F):
            ema[a, k] = X[0, k]
    for t in range(T):
        for k in range(F):
            x = X[t, k]
            out[t, k] = x
            for a in range(A):
                al = alphas[a]
                ema[a, k] = al * x + (1.0 - al) * ema[a, k]
                d = x - ema[a, k]
                if scaled:
                    sc[a, k] = al * abs(d) + (1.0 - al) * sc[a, k]
                    v = d / (sc[a, k] + SCALE_EPS)
                    if v > SCALE_CLIP:
                        v = SCALE_CLIP
                    elif v < -SCALE_CLIP:
                        v = -SCALE_CLIP
                    out[t, (1 + a) * F + k] = v
                else:
                    out[t, (1 + a) * F + k] = d


@njit(fastmath=True, cache=True)
def bank_step(x, alphas, scaled, ema, sc, first, out):
    """Online version. ema, sc: (A,F) state arrays. first: 1 on the first row of a sequence."""
    F = x.shape[0]
    A = alphas.shape[0]
    for k in range(F):
        v = x[k]
        if first:
            for a in range(A):
                ema[a, k] = v; sc[a, k] = 0.0
        out[k] = v
        for a in range(A):
            al = alphas[a]
            ema[a, k] = al * v + (1.0 - al) * ema[a, k]
            d = v - ema[a, k]
            if scaled:
                sc[a, k] = al * abs(d) + (1.0 - al) * sc[a, k]
                z = d / (sc[a, k] + SCALE_EPS)
                if z > SCALE_CLIP:
                    z = SCALE_CLIP
                elif z < -SCALE_CLIP:
                    z = -SCALE_CLIP
                out[(1 + a) * F + k] = z
            else:
                out[(1 + a) * F + k] = d


def bank_step_numpy(x, alphas, scaled, ema, sc, first, out):
    F = x.shape[0]
    if first:
        ema[:] = x[None, :]; sc[:] = 0.0
    out[:F] = x
    for a, al in enumerate(alphas):
        ema[a] = al * x + (1.0 - al) * ema[a]
        d = x - ema[a]
        if scaled:
            sc[a] = al * np.abs(d) + (1.0 - al) * sc[a]
            out[(1 + a) * F:(2 + a) * F] = np.clip(d / (sc[a] + SCALE_EPS), -SCALE_CLIP, SCALE_CLIP)
        else:
            out[(1 + a) * F:(2 + a) * F] = d
