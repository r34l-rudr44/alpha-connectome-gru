"""Download and extract the starter pack into a Modal Volume.

Runs on Modal's network instead of uploading 31 GB from the laptop.
Usage: modal run modal/fetch_data.py
"""
import modal

app = modal.App("wnn-connectome")
vol = modal.Volume.from_name("wnn-connectome-data", create_if_missing=True)
image = modal.Image.debian_slim().apt_install("curl", "pigz")

URL = "<starter-pack URL from the competition site>"
ROOT = "/data"
DONE = f"{ROOT}/.fetch_done"


@app.function(image=image, volumes={ROOT: vol}, timeout=4 * 3600, cpu=4, memory=8192)
def fetch():
    import os
    import subprocess

    if os.path.exists(DONE):
        print("already fetched")
    else:
        subprocess.run(
            f"cd {ROOT} && curl -fL --retry 5 --retry-delay 10 {URL} | pigz -dc | tar -x",
            shell=True,
            check=True,
        )
        open(DONE, "w").close()
        vol.commit()
    subprocess.run(f"ls -la {ROOT}/wnn_connectome_starterpack; du -sh {ROOT}/wnn_connectome_starterpack/*", shell=True)


@app.local_entrypoint()
def main():
    fetch.remote()
