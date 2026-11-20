"""Paths and constants. The competition data is not part of this repository.

Point WNN_PACK_DIR at the extracted starter pack. Without it the code looks for ./data/wnn_connectome_starterpack
and then falls back to ./sample, a small slice written by modal/extract_sample.py.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_PACK = Path(os.environ.get("WNN_PACK_DIR", ROOT / "data" / "wnn_connectome_starterpack"))
_full = (_PACK / "datasets" / "valid_mask.parquet").exists()   # last file in the archive => extraction complete
PACK_DIR = _PACK if _full else ROOT / "sample"
DATA_DIR = PACK_DIR / "datasets"
TRAIN = DATA_DIR / ("train.parquet" if _full else "train_sample.parquet")
VALID = DATA_DIR / ("valid.parquet" if _full else "valid_sample.parquet")
VALID_MASK = DATA_DIR / "valid_mask.parquet"
MODAL_PACK_DIR = "/data/wnn_connectome_starterpack"            # inside the Modal volume

SEQ_LEN = 20_000
WARMUP = 99            # steps 0..98 have need_prediction == False
N_FEATURES = 112
TARGETS = ["t0", "t1"]
CLIP = 2.0

# Scoring budget: 60 min for 1,970 sequences x 20,000 steps on 1 vCPU.
TEST_STEPS = 1_970 * SEQ_LEN
BUDGET_US_PER_STEP = 60 * 60 * 1e6 / TEST_STEPS      # ~91 us
LOCAL_TARGET_US_PER_STEP = 45.0                      # 2x safety margin vs. the scorer's vCPU
