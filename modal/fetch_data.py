"""Download (resumable) and extract the starter pack into a Modal Volume.

Runs on Modal's network instead of uploading 31 GB from the laptop.
Usage: modal run modal/fetch_data.py
"""
import modal

app = modal.App("wnn-connectome")
vol = modal.Volume.from_name("wnn-connectome-data", create_if_missing=True)
image = modal.Image.debian_slim().apt_install("curl", "pigz")

URL = "<starter-pack URL from the competition site>"
TOTAL = 33737976954
ROOT = "/data"
TB = f"{ROOT}/dl/pack.tar.gz"
DONE = f"{ROOT}/.fetch_done"


@app.function(image=image, volumes={ROOT: vol}, timeout=4 * 3600, cpu=4, memory=8192)
def fetch():
    import os
    import shutil
    import subprocess
    import time

    def sh(cmd, check=True):
        print("$", cmd, flush=True)
        return subprocess.run(cmd, shell=True, check=check)

    if os.path.exists(DONE):
        print("already fetched")
    else:
        shutil.rmtree(f"{ROOT}/wnn_connectome_starterpack", ignore_errors=True)  # partial extract from a failed run
        os.makedirs(f"{ROOT}/dl", exist_ok=True)
        for attempt in range(40):
            have = os.path.getsize(TB) if os.path.exists(TB) else 0
            if have >= TOTAL:
                break
            print(f"attempt {attempt}: resuming at {have:,}/{TOTAL:,}", flush=True)
            sh(f"curl -f -sS -r {have}- '{URL}' >> '{TB}'", check=False)
            vol.commit()
            time.sleep(5)
        assert os.path.getsize(TB) == TOTAL, os.path.getsize(TB)
        sh(f"cd {ROOT} && pigz -dc '{TB}' | tar -x")
        os.remove(TB)
        open(DONE, "w").close()
        vol.commit()
    sh(f"ls -la {ROOT}/wnn_connectome_starterpack; du -sh {ROOT}/wnn_connectome_starterpack/*", check=False)


@app.local_entrypoint()
def main():
    fetch.remote()
