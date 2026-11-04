"""Submission entry point: stateful GRU, one step per predict() call.

Backend preference: numba kernel (fastest) -> onnxruntime (same speed class as the
official baseline) -> pure numpy. All three produce identical outputs (<1e-6).
"""
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


def _make_backend():
    errors = []
    try:
        import numba  # noqa: F401
        from infer import StepModel, HAVE_NUMBA

        if HAVE_NUMBA:
            return "numba", StepModel(str(HERE / "weights.npz"))
    except Exception as e:  # pragma: no cover
        errors.append(f"numba: {e!r}")
    if (HERE / "model.onnx").exists():
        try:
            return "onnx", _OrtBackend(HERE / "model.onnx")
        except Exception as e:  # pragma: no cover
            errors.append(f"onnx: {e!r}")
    os.environ["WNN_NO_NUMBA"] = "1"
    from infer import StepModel

    return "numpy", StepModel(str(HERE / "weights.npz"))


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
