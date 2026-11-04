"""Paths and constants. Data lives outside the repo."""
from pathlib import Path

PACK_DIR = Path("data/wnn_connectome_starterpack")
DATA_DIR = PACK_DIR / "datasets"
TRAIN = DATA_DIR / "train.parquet"
VALID = DATA_DIR / "valid.parquet"
VALID_MASK = DATA_DIR / "valid_mask.parquet"
CACHE_DIR = Path("cache")
MODAL_PACK_DIR = "/data/wnn_connectome_starterpack"                # inside the Modal volume

SEQ_LEN = 20_000
WARMUP = 99            # steps 0..98 have need_prediction == False
N_FEATURES = 112
TARGETS = ["t0", "t1"]
CLIP = 2.0

# Scoring budget: 60 min for 1,970 sequences x 20,000 steps on 1 vCPU.
TEST_STEPS = 1_970 * SEQ_LEN
BUDGET_US_PER_STEP = 60 * 60 * 1e6 / TEST_STEPS      # ~91 us
LOCAL_TARGET_US_PER_STEP = 45.0                      # 2x safety margin vs. the scorer's vCPU
