"""Fetch a run from the Modal volume and export weights.npz + model.onnx locally.

Usage: python export_run.py gru128 [gru128x2 ...]
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from wnn.infer import StepModel, export_npz
from wnn.model import GRUModel, export_onnx

ROOT = Path(__file__).resolve().parent
os.environ["PYTHONUTF8"] = "1"


def fetch(name: str):
    d = ROOT / "runs" / name
    d.mkdir(parents=True, exist_ok=True)
    for fn in ("best.pt", "hist.json", "norm.npz"):
        subprocess.run(["modal", "volume", "get", "--force", "wnn-connectome-data", f"runs/{name}/{fn}", str(d / fn)],
                       check=True, capture_output=True)
    return d


def export(d: Path):
    ck = torch.load(d / "best.pt", weights_only=False, map_location="cpu")
    a = ck["args"]
    m = GRUModel(len(ck["mean"]), a["hidden"], a["layers"], a["in_proj"], a["dropout"], a["x_clip"])
    m.load_state_dict(ck["model"]); m.eval()
    export_npz(m, str(d / "weights.npz"))
    export_onnx(m, str(d / "model.onnx"))
    # consistency check: torch vs numba step vs onnx
    X = (np.random.default_rng(0).normal(size=(300, len(ck["mean"]))) * 2).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.from_numpy(X)[None])[0][0].numpy()
    sm = StepModel(str(d / "weights.npz")); nb = np.array([sm.step(x).copy() for x in X])
    import onnxruntime as ort
    s = ort.InferenceSession(str(d / "model.onnx"), providers=["CPUExecutionProvider"])
    h = np.zeros((a["layers"], 1, a["hidden"]), np.float32); oo = []
    for x in X:
        y, h = s.run(None, {"x": x[None], "h": h}); oo.append(y)
    print(f"{d.name}: best={ck['hist'][-1]['score']:.5f} numba-vs-torch {np.abs(nb - ref).max():.1e} onnx-vs-torch {np.abs(np.array(oo) - ref).max():.1e}")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        export(fetch(name))
