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
