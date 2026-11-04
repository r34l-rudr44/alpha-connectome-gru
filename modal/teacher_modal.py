"""Train the bidirectional teacher on Modal and dump soft labels to /data/teacher/<name>/{train,valid}.npy.

Usage: modal run --detach modal/teacher_modal.py --name bi128 --extra "--hidden 128 --epochs 2" --detach
"""
import modal

app = modal.App("wnn-connectome-teacher")
vol = modal.Volume.from_name("wnn-connectome-data")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "numpy", "pyarrow", "numba")
         .add_local_python_source("wnn"))
PACK = "/data/wnn_connectome_starterpack"


@app.function(image=image, volumes={"/data": vol}, gpu="A10G", cpu=4, memory=16384, timeout=6 * 3600)
def run(name: str, extra: str):
    import shlex
    import sys

    from wnn import teacher

    sys.argv = ["teacher", "--train", f"{PACK}/datasets/train.parquet", "--valid", f"{PACK}/datasets/valid.parquet",
                "--out", f"/data/teacher/{name}", "--workers", "4"] + shlex.split(extra)
    teacher.main()
    vol.commit()
    return open(f"/data/teacher/{name}/hist.json").read()


@app.local_entrypoint()
def main(name: str = "bi128", extra: str = "", detach: bool = False):
    if detach:
        print("spawned", run.spawn(name, extra).object_id); return
    print(run.remote(name, extra))
