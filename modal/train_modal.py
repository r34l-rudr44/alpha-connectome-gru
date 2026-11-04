"""Run wnn/train.py on a Modal GPU against the data volume; results land in /data/runs/<name> and ./runs/<name>.

Usage: modal run modal/train_modal.py --name gru128 --extra "--hidden 128 --epochs 3"
"""
import modal

app = modal.App("wnn-connectome-train")
vol = modal.Volume.from_name("wnn-connectome-data")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy", "pyarrow", "onnx", "onnxruntime", "onnxscript")
         .add_local_python_source("wnn"))
PACK = "/data/wnn_connectome_starterpack"


@app.function(image=image, volumes={"/data": vol}, gpu="A10G", cpu=4, memory=16384, timeout=6 * 3600)
def run(name: str, extra: str):
    import shlex
    import subprocess

    from wnn.train import get_parser, train

    argv = ["--train", f"{PACK}/datasets/train.parquet", "--valid", f"{PACK}/datasets/valid.parquet",
            "--out", f"/data/runs/{name}", "--workers", "4"] + shlex.split(extra)
    print("args:", argv, flush=True)
    subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
    best = train(get_parser().parse_args(argv))
    vol.commit()
    files = {}
    for fn in ("best.pt", "weights.npz", "norm.npz", "hist.json"):
        p = f"/data/runs/{name}/{fn}"
        try:
            files[fn] = open(p, "rb").read()
        except FileNotFoundError:
            pass
    return best, files


@app.local_entrypoint()
def main(name: str = "gru128", extra: str = "", detach: bool = False):
    import pathlib

    if detach:  # modal run --detach modal/train_modal.py --name X --extra "..." --detach
        print("spawned", run.spawn(name, extra).object_id)  # outputs: modal volume get wnn-connectome-data runs/X/<file>
        return
    best, files = run.remote(name, extra)
    d = pathlib.Path("runs") / name
    d.mkdir(parents=True, exist_ok=True)
    for fn, b in files.items():
        (d / fn).write_bytes(b)
    print(f"best={best:.5f}; saved {list(files)} to {d}")
