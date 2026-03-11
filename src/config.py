from pathlib import Path

# ============================================================
# Model / data
# ============================================================

MODEL_NAME = "google/gemma-2-2b-it"
DOLMA_SAMPLE_URL = "https://olmo-data.org/dolma-v1_6-8B-sample/v1_5r2_sample-0000.json.gz"

HOOK_LAYER_INDEX = 11
MAX_SEQ_LEN = 64
MIN_TEXT_CHARS = 20

# ============================================================
# Training
# ============================================================

TEXT_BATCH_SIZE_PER_RANK = 8
TRAIN_STEPS = 200
LOG_EVERY = 10
SAVE_EVERY = 100

LR = 1e-3
SEED = 42

# ============================================================
# SAE
# ============================================================

LATENT_FACTOR = 4
INIT_THRESHOLD = 1e-3
STE_BANDWIDTH = 1e-3
L0_COEFF = 1e-3
ACT_NORM_SCALE = 1.0

# ============================================================
# Mean initialisation for b_dec
# ============================================================

MEAN_INIT_BATCHES = 16

# ============================================================
# Runtime / IO
# ============================================================

HTTP_TIMEOUT = 30
USER_AGENT = "Mozilla/5.0"
REQUIRE_CUDA = True

OUTPUT_DIR = Path("./outputs/jumprelu_sae")
DATA_CACHE_DIR = Path("./data_cache")
DATA_CACHE_PATH = DATA_CACHE_DIR / "v1_5r2_sample-0000.json.gz"


def get_hf_token() -> str | None:
    return None if "HF_TOKEN" not in __import__("os").environ else __import__("os").environ["HF_TOKEN"]