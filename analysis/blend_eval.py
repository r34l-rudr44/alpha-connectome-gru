"""Evaluate blends of prediction dumps on valid (local CPU). Needs runs/valid_targets.npz (modal/extract_targets.py).

Usage: python analysis/blend_eval.py runs/ens_r3_valid.npy runs/lt128_valid.npy
"""
import sys

import numpy as np

WARM = 99


def wp(y, p):
    y = np.clip(y, -2, 2); p = np.clip(p, -2, 2); w = np.abs(y)
    out = []
    for j in range(2):
        wj, yj, pj = w[:, j], y[:, j], p[:, j]
        my = (wj * yj).sum() / wj.sum(); mp = (wj * pj).sum() / wj.sum()
        out.append(float((wj * (yj - my) * (pj - mp)).sum() / np.sqrt((wj * (yj - my) ** 2).sum() * (wj * (pj - mp) ** 2).sum())))
    return round(out[0], 4), round(out[1], 4), round(sum(out) / 2, 4)


def main():
    t = np.load("runs/valid_targets.npz")
    y, sc = t["y"], t["scored"]; sc[:, :WARM] = False
    preds = [np.asarray(np.load(p, mmap_mode="r"), dtype=np.float32) for p in sys.argv[1:]]
    ym = y[sc]
    pm = [p[sc] for p in preds]
    for name, p in zip(sys.argv[1:], pm):
        print(f"{name}: {wp(ym, p)}")
    if len(pm) == 2:
        a, b = pm
        print("corr of preds (t0,t1):", [round(float(np.corrcoef(a[:, j], b[:, j])[0, 1]), 4) for j in range(2)])
        for w in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            print(f"blend {1 - w:.1f}*A + {w:.1f}*B: {wp(ym, (1 - w) * a + w * b)}")


if __name__ == "__main__":
    main()
