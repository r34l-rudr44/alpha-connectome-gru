"""Submission entry point: stateful GRU, one numba-compiled step per predict() call."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("NUMBA_CACHE_DIR", str(HERE / ".numba_cache"))
sys.path.insert(0, str(HERE))

import numpy as np

from infer import StepModel  # bundled copy of wnn/infer.py


class PredictionModel:
    def __init__(self):
        self.m = StepModel(str(HERE / "weights.npz"))
        self.seq_ix = None

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.m.reset()
        y = self.m.step(np.asarray(data_point.state, dtype=np.float32))
        if not data_point.need_prediction:
            return None
        return y.copy()
