"""Run wnn/train_local.py (sliding-window Transformer teacher) on a Modal A10G; results in /data/runs/<name>.

Usage: modal run --detach modal/train_local_modal.py --name lt128 --extra "--epochs 3 --weight-dir /data/maskw --weight-pow 0.5" --detach
"""
import modal

app = modal.App("wnn-connectome-train-local")
vol = modal.Volume.from_name("wnn-connectome-data")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy", "pyarrow", "numba")
         .add_local_python_source("wnn"))
PACK = "/data/wnn_connectome_starterpack"


@app.function(image=image, volumes={"/data": vol}, gpu="A10G", cpu=4, memory=16384, timeout=6 * 3600)
def run(name: str, extra: str):
    import shlex
    import subprocess

    from wnn.train_local import get_local_parser, train

    argv = ["--train", f"{PACK}/datasets/train.parquet", "--valid", f"{PACK}/datasets/valid.parquet",
            "--out", f"/data/runs/{name}", "--workers", "4"] + shlex.split(extra)
    print("args:", argv, flush=True)
    subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv", shell=True)
    best = train(get_local_parser().parse_args(argv))
    vol.commit()
    return best


@app.local_entrypoint()
def main(name: str = "lt128", extra: str = "", detach: bool = False):
    if detach:
        print("spawned", run.spawn(name, extra).object_id); return
    print(run.remote(name, extra))
