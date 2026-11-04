"""Submission entry point: one or more stateful GRUs (numba kernels), predictions averaged.

Model files: weights.npz (single) or weights_0.npz, weights_1.npz, ... (ensemble; optional
ensemble.json = {"weights": [w0, w1, ...]} for a weighted average).
Backend preference: numba -> onnxruntime (single model only, model.onnx) -> pure numpy.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", str(HERE / ".numba_cache"))
sys.path.insert(0, str(HERE))

import numpy as np


class _OrtBackend:
    def __init__(self, path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        hs = [d if isinstance(d, int) else 1 for d in self.sess.get_inputs()[1].shape]
        self.h0 = np.zeros(hs, dtype=np.float32)
        self.h = self.h0.copy()
        self.x = np.zeros((1, self.sess.get_inputs()[0].shape[1]), dtype=np.float32)

    def reset(self):
        self.h = self.h0.copy()

    def step(self, x):
        self.x[0] = x
        y, self.h = self.sess.run(None, {"x": self.x, "h": self.h})
        return y


class _Ensemble:
    def __init__(self, models, weights):
        self.models = models
        self.w = np.asarray(weights, dtype=np.float32) / float(np.sum(weights))
        self.out = np.zeros(2, dtype=np.float32)

    def reset(self):
        for m in self.models:
            m.reset()

    def step(self, x):
        self.out[:] = 0
        for w, m in zip(self.w, self.models):
            self.out += w * m.step(x)
        return self.out


def _weight_files():
    multi = sorted(HERE.glob("weights_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    return multi if multi else [HERE / "weights.npz"]


def _make_backend():
    files = _weight_files()
    weights = [1.0] * len(files)
    if (HERE / "ensemble.json").exists():
        weights = json.loads((HERE / "ensemble.json").read_text()).get("weights", weights)
    try:
        import numba  # noqa: F401
        from infer import StepModel, HAVE_NUMBA

        if HAVE_NUMBA:
            ms = [StepModel(str(f)) for f in files]
            return "numba", (ms[0] if len(ms) == 1 else _Ensemble(ms, weights))
    except Exception:  # pragma: no cover
        pass
    if len(files) == 1 and (HERE / "model.onnx").exists():
        try:
            return "onnx", _OrtBackend(HERE / "model.onnx")
        except Exception:  # pragma: no cover
            pass
    os.environ["WNN_NO_NUMBA"] = "1"
    from infer import StepModel

    ms = [StepModel(str(f)) for f in files]
    return "numpy", (ms[0] if len(ms) == 1 else _Ensemble(ms, weights))


class PredictionModel:
    def __init__(self):
        self.backend_name, self.m = _make_backend()
        self.seq_ix = None

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.m.reset()
        y = self.m.step(np.asarray(data_point.state, dtype=np.float32))
        if not data_point.need_prediction:
            return None
        return np.asarray(y, dtype=np.float32).reshape(2).copy()
