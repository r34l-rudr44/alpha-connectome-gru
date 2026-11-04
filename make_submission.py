"""Build submissions/<name>.zip from a run directory (weights.npz) + submit/solution.py + submit/infer.py.

Usage: python make_submission.py runs/gru128 --name gru128 [--bench N]
"""
import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--bench", type=int, default=0, help="run bench.py on N validation sequences first")
    a = ap.parse_args()

    stage = ROOT / "submissions" / a.name
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copy(ROOT / "submit" / "solution.py", stage / "solution.py")
    shutil.copy(ROOT / "wnn" / "infer.py", stage / "infer.py")
    shutil.copy(a.run_dir / "weights.npz", stage / "weights.npz")
    if (a.run_dir / "model.onnx").exists():
        shutil.copy(a.run_dir / "model.onnx", stage / "model.onnx")
    (stage / "MODEL.md").write_text(f"run: {a.run_dir}\n")

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
