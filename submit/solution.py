"""Submission entry point: stateful GRU exported to ONNX, one step per predict() call."""
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

try:
    from utils import DataPoint  # provided by the scorer
except Exception:  # pragma: no cover
    DataPoint = object


class PredictionModel:
    def __init__(self):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(HERE / "model.onnx"), so, providers=["CPUExecutionProvider"])
        h_shape = [d if isinstance(d, int) else 1 for d in self.sess.get_inputs()[1].shape]
        self.h0 = np.zeros(h_shape, dtype=np.float32)
        self.h = self.h0.copy()
        self.seq_ix = None
        self.x = np.zeros((1, self.sess.get_inputs()[0].shape[1]), dtype=np.float32)

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.h = self.h0.copy()
        self.x[0] = data_point.state
        y, self.h = self.sess.run(None, {"x": self.x, "h": self.h})
        if not data_point.need_prediction:
            return None
        return y.astype(np.float32)
