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
from wnn.local_model import build_from_ckpt
from wnn.model import export_onnx

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
    m = build_from_ckpt(ck)  # GRUModel incl. Highway head / input projection
    assert not a.get("aug") and not a.get("bank"), "aug/bank runs: use the weights.npz exported by the trainer"
    export_npz(m, str(d / "weights.npz"))
    # consistency check: torch vs numba step (the deployed path) on random inputs, plus onnx when it exports
    X = (np.random.default_rng(0).normal(size=(300, len(ck["mean"]))) * 2).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.from_numpy(X)[None])[0][0].numpy()
    sm = StepModel(str(d / "weights.npz")); nb = np.array([sm.step(x).copy() for x in X])
    msg = f"{d.name}: best={max(h['score'] for h in ck['hist']):.5f} numba-vs-torch {np.abs(nb - ref).max():.1e}"
    try:
        import onnxruntime as ort

        export_onnx(m, str(d / "model.onnx"))
        s = ort.InferenceSession(str(d / "model.onnx"), providers=["CPUExecutionProvider"])
        h = np.zeros((a["layers"], 1, a["hidden"]), np.float32); oo = []
        for x in X:
            y, h = s.run(None, {"x": x[None], "h": h}); oo.append(y)
        msg += f" onnx-vs-torch {np.abs(np.array(oo) - ref).max():.1e}"
    except Exception as e:  # onnx is optional (the submission uses the numba kernel)
        msg += f" (onnx skipped: {type(e).__name__})"
    assert np.abs(nb - ref).max() < 1e-4, msg
    print(msg)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        export(fetch(name))
