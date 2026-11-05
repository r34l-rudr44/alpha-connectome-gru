"""Tier-0 probe: how much history does the trained GRU actually use? (local CPU, torch)

Runs a checkpoint over valid sequences with the hidden state reset every K steps (K = 20000 is the normal model)
and reports scored WP per K. Also reports WP by position in the sequence for the full-memory model.
If WP(K=1000) ~ WP(K=20000) the model only exploits ~1 TBPTT window of history -> long-memory is an untapped lever.

Usage: python analysis/memory_probe.py --ckpt runs/r2_s0/best.pt --n-seq 300
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from wnn.model import GRUModel  # noqa: E402

META = ("seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1")
WARM = config.WARMUP


def wpearson(y, p):
    y = np.clip(y, -2, 2); p = np.clip(p, -2, 2); w = np.abs(y)
    out = []
    for j in range(y.shape[1]):
        wj = w[:, j]; yj = y[:, j]; pj = p[:, j]
        my = (wj * yj).sum() / wj.sum(); mp = (wj * pj).sum() / wj.sum()
        cov = (wj * (yj - my) * (pj - mp)).sum()
        out.append(float(cov / np.sqrt((wj * (yj - my) ** 2).sum() * (wj * (pj - mp) ** 2).sum())))
    return [round(out[0], 4), round(out[1], 4), round((out[0] + out[1]) / 2, 4)]


def load_model(path):
    ck = torch.load(path, weights_only=False, map_location="cpu")
    a = ck["args"]
    m = GRUModel(len(ck["mean"]), a["hidden"], a["layers"], a["in_proj"], a["dropout"], a["x_clip"])
    m.load_state_dict(ck["model"]); m.eval()
    return m


@torch.no_grad()
def run_reset(m, X, K):
    """X (B,T,F) -> preds (B,T,2) with the state zeroed every K steps."""
    B, T, _ = X.shape
    out = torch.empty(B, T, 2)
    for s in range(0, T, K):
        e = min(s + K, T)
        y, _ = m(X[:, s:e], None)
        out[:, s:e] = y
    return out.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-seq", type=int, default=300)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--ks", default="100,250,500,1000,2000,5000,20000")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="runs/memory_probe.json")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    Ks = [int(k) for k in a.ks.split(",")]
    m = load_model(a.ckpt)
    pf = pq.ParquetFile(str(config.VALID))
    feat = [c for c in pf.schema_arrow.names if c not in META]
    rng = np.random.default_rng(0); idx = rng.permutation(pf.num_row_groups)[: a.n_seq]

    Y, S, P = [], [], {k: [] for k in Ks}
    t0 = time.time()
    for b in range(0, len(idx), a.batch):
        Xs, ys, ss = [], [], []
        for rg in idx[b: b + a.batch]:
            t = pf.read_row_group(int(rg))
            Xs.append(np.stack([t.column(c).to_numpy(zero_copy_only=False) for c in feat], 1).astype(np.float32))
            ys.append(np.stack([t.column("t0").to_numpy(), t.column("t1").to_numpy()], 1).astype(np.float32))
            ss.append(t.column("is_scored").to_numpy().astype(bool))
        X = torch.from_numpy(np.stack(Xs))
        Y.append(np.stack(ys)); S.append(np.stack(ss))
        for k in Ks:
            P[k].append(run_reset(m, X, k))
        print(f"batch {b // a.batch + 1}/{(len(idx) + a.batch - 1) // a.batch} {time.time() - t0:.0f}s", flush=True)
    Y = np.concatenate(Y); S = np.concatenate(S)
    need = np.zeros(Y.shape[1], bool); need[WARM:] = True
    res = {"ckpt": a.ckpt, "n_seq": int(len(idx)), "by_K": {}, "by_position_fullmem": {}}
    for k in Ks:
        p = np.concatenate(P[k])
        msk = S & need[None, :]
        res["by_K"][k] = {"scored": wpearson(Y[msk], p[msk]), "all": wpearson(Y[:, WARM:].reshape(-1, 2), p[:, WARM:].reshape(-1, 2))}
        print(f"K={k:>6}: {res['by_K'][k]}", flush=True)
    p = np.concatenate(P[max(Ks)])
    for lo, hi in ((99, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 10000), (10000, 20000)):
        pos = np.zeros(Y.shape[1], bool); pos[lo:hi] = True
        msk = S & pos[None, :]
        res["by_position_fullmem"][f"{lo}-{hi}"] = wpearson(Y[msk], p[msk])
        print(f"steps {lo:>5}-{hi:<5}: {res['by_position_fullmem'][f'{lo}-{hi}']}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"saved {a.out} total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
