"""Tiny scheduling probe: does a GPU container start? Usage: modal run modal/gpu_probe.py --gpu L4"""
import modal

app = modal.App("wnn-gpu-probe")


def _smi():
    import subprocess

    return subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, capture_output=True, text=True).stdout


@app.function(gpu="L4", cpu=1, memory=2048, timeout=120)
def probe_l4():
    return _smi()


@app.function(gpu="A10G", cpu=1, memory=2048, timeout=120)
def probe_a10g():
    return _smi()


@app.function(gpu="T4", cpu=1, memory=2048, timeout=120)
def probe_t4():
    return _smi()


@app.function(cpu=1, memory=1024, timeout=120)
def probe_cpu():
    return "cpu ok"


@app.local_entrypoint()
def main(gpu: str = "L4"):
    import time

    f = {"L4": probe_l4, "A10G": probe_a10g, "T4": probe_t4, "CPU": probe_cpu}[gpu]
    t = time.time()
    print(gpu, "->", f.remote().strip(), f"in {time.time() - t:.0f}s")
