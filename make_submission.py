"""Build submissions/<name>.zip from one or more run directories.

Usage: python make_submission.py runs/gru128 [runs/g128_s1 ...] --name NAME [--weights 1,1,...] [--bench N]
Single run: weights.npz (+ model.onnx if present). Several runs: weights_<i>.npz + ensemble.json.
"""
import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", type=Path, nargs="+")
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", default="", help="comma-separated ensemble weights")
    ap.add_argument("--bench", type=int, default=0, help="run bench.py on N validation sequences first")
    a = ap.parse_args()

    stage = ROOT / "submissions" / a.name
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for fn in ("solution.py",):
        shutil.copy(ROOT / "submit" / fn, stage / fn)
    for fn in ("infer.py", "features.py"):
        shutil.copy(ROOT / "wnn" / fn, stage / fn)
    if len(a.run_dirs) == 1:
        d = a.run_dirs[0]
        shutil.copy(d / "weights.npz", stage / "weights.npz")
        if (d / "model.onnx").exists():
            shutil.copy(d / "model.onnx", stage / "model.onnx")
    else:
        for i, d in enumerate(a.run_dirs):
            shutil.copy(d / "weights.npz", stage / f"weights_{i}.npz")
        w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * len(a.run_dirs)
        (stage / "ensemble.json").write_text(json.dumps({"weights": w, "runs": [str(d) for d in a.run_dirs]}))
    (stage / "MODEL.md").write_text("runs: " + ", ".join(str(d) for d in a.run_dirs) + "\n")

    if a.bench:
        r = subprocess.run([sys.executable, str(ROOT / "bench.py"), str(stage), "--n-seq", str(a.bench)])
        if r.returncode:
            sys.exit("bench failed")

    zpath = ROOT / "submissions" / f"{a.name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in stage.iterdir():
            if p.is_file():
                z.write(p, p.name)
    mb = zpath.stat().st_size / 1e6
    print(f"{zpath} ({mb:.2f} MB; limit 20)")
    assert mb < 20


if __name__ == "__main__":
    main()
