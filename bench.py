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
    verdict = "OK" if us <= config.LOCAL_TARGET_US_PER_STEP else "TOO SLOW"
    print(f"{steps:,} steps in {t_model:.1f}s -> {us:.1f} us/step  [{verdict}; local target {config.LOCAL_TARGET_US_PER_STEP:.0f}, scorer budget {config.BUDGET_US_PER_STEP:.0f}]")
    print(f"projected scorer time at this speed: {us * config.TEST_STEPS / 60e6:.1f} min")
    if has_targets:
        print(score(np.concatenate(ys), np.concatenate(preds), np.concatenate(masks)))


if __name__ == "__main__":
    main()
