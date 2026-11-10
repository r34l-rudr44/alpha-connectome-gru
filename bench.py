"""Latency + score harness for a submission directory.

Runs <subdir>/solution.py the way the scorer does: single thread, one DataPoint
per row, in order. Reports microseconds per step and the WP score on the
sequences used.

Usage: python bench.py <submission_dir> [--parquet PATH] [--n-seq 5] [--start 0]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import config
from metric import score


def load_solution(subdir: Path):
    sys.path.insert(0, str(subdir))
    if not (subdir / "utils.py").exists() and (config.PACK_DIR / "utils.py").exists():
        sys.path.insert(0, str(config.PACK_DIR))
    spec = importlib.util.spec_from_file_location("solution", subdir / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from utils import DataPoint  # noqa: E402  (the pack's utils.py)
    return mod.PredictionModel, DataPoint


def reference_us_per_step(X, n_steps=20000):
    """us/step of one random GRU-128 through the same numba kernel, measured in this process (same machine load).

    Absolute laptop timings are useless as a verdict (the scorer core is several times faster and the laptop is
    usually loaded); the ratio "submission us/step : bare GRU-128 kernel us/step" is what transfers. The ratio
    includes the Python call overhead and the cache cost of several weight sets, so 3 models are about 5 units.
    Calibration against scored submissions (2026-11-11, 2 sequences each):
        gru128.zip  (1 x GRU-128)        1.6 units -> 11.5 min on the scorer
        ens3_r1.zip (3 x GRU-128)        4.9 units -> 32-34 min
        ens4.zip    (4 models, 2.2 MB)   7.4 units -> 56.4 min (too close to the 60-min limit)
    i.e. roughly 7-8 scorer minutes per unit; the scorer itself varies by +-5%, sometimes +30%.
    """
    import tempfile

    from wnn.infer import StepModel

    rng = np.random.default_rng(0); H, F = 128, X.shape[1]; f = np.float32
    d = {"mean": np.zeros(F, f), "std": np.ones(F, f), "x_clip": f(10), "layers": np.int32(1), "hidden": np.int32(H),
         "aug": np.int32(0), "n_raw": np.int32(F), "WiT0": (rng.normal(size=(3 * H, F)) * 0.05).astype(f),
         "WhT0": (rng.normal(size=(3 * H, H)) * 0.05).astype(f), "bi0": np.zeros(3 * H, f), "bh0": np.zeros(3 * H, f),
         "use_ln": np.int32(1), "ln_g": np.ones(H, f), "ln_b": np.zeros(H, f), "n_hw": np.int32(0),
         "WlT": (rng.normal(size=(2, H)) * 0.05).astype(f), "bl": np.zeros(2, f)}
    with tempfile.TemporaryDirectory() as tmp:
        np.savez(f"{tmp}/ref.npz", **d)
        sm = StepModel(f"{tmp}/ref.npz")
    n = min(n_steps, len(X))
    t0 = time.perf_counter()
    for i in range(n):
        sm.step(X[i])
    return (time.perf_counter() - t0) / n * 1e6


SAFE_UNITS = 5.8  # about 40-45 scorer minutes (see reference_us_per_step); 7.4 units took 56 of the 60 minutes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subdir", type=Path)
    ap.add_argument("--parquet", type=Path, default=config.VALID)
    ap.add_argument("--n-seq", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    Model, DataPoint = load_solution(args.subdir)
    pf = pq.ParquetFile(args.parquet)
    cols = pf.schema_arrow.names
    feat_cols = [c for c in cols if c not in ("seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1")]
    assert len(feat_cols) == config.N_FEATURES, len(feat_cols)
    has_targets = "t0" in cols
    has_mask = "is_scored" in cols

    model = Model()
    preds, ys, masks = [], [], []
    steps = 0
    t_model = 0.0
    for rg in range(args.start, min(args.start + args.n_seq, pf.num_row_groups)):
        tbl = pf.read_row_group(rg)
        seq_ix = tbl.column("seq_ix").to_numpy()
        step = tbl.column("step_in_seq").to_numpy()
        need = tbl.column("need_prediction").to_numpy()
        X = np.stack([tbl.column(c).to_numpy() for c in feat_cols], axis=1).astype(np.float32)
        n = len(seq_ix)
        out = np.zeros((n, 2), dtype=np.float32)
        t0 = time.perf_counter()
        for i in range(n):
            dp = DataPoint(seq_ix=int(seq_ix[i]), step_in_seq=int(step[i]), need_prediction=bool(need[i]), state=X[i])
            p = model.predict(dp)
            if need[i]:
                assert p is not None and np.shape(p) == (2,) and np.all(np.isfinite(p)), (rg, i, p)
                out[i] = p
        t_model += time.perf_counter() - t0
        steps += n
        preds.append(out)
        if has_targets:
            ys.append(np.stack([tbl.column("t0").to_numpy(), tbl.column("t1").to_numpy()], axis=1))
        m = need.copy()
        if has_mask:
            m &= tbl.column("is_scored").to_numpy()
        masks.append(m)

    us = t_model / steps * 1e6
    ref = reference_us_per_step(X)
    eq = us / ref
    verdict = "OK" if eq <= SAFE_UNITS else "RISKY: may exceed the 60-minute limit"
    print(f"{steps:,} steps in {t_model:.1f}s -> {us:.1f} us/step on this machine; bare GRU-128 kernel here: {ref:.1f} us/step")
    print(f"= {eq:.2f} units [{verdict}; safe <= {SAFE_UNITS}; scored references: 1.6 units = 11.5 min, 4.9 = 33 min, 7.4 = 56 min]")
    if has_targets:
        print(score(np.concatenate(ys), np.concatenate(preds), np.concatenate(masks)))


if __name__ == "__main__":
    main()
