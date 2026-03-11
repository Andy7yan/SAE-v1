from pathlib import Path

MODEL_NAME = "google/gemma-2-2b-it"
DOLMA_URL = "https://olmo-data.org/dolma-v1_6/books/books-0000.json.gz"
DATA_CACHE_PATH = Path("./data_cache/books-0000.json.gz")

HOOK_LAYER_INDEX = 12
MAX_SEQ_LEN = 64
MIN_TEXT_CHARS = 20

TEXT_BATCH_SIZE_PER_RANK = 8
SAE_BATCH_SIZE = 4096
BUFFER_CAPACITY = 32768

TRAIN_STEPS = 200
LOG_EVERY = 10
SAVE_EVERY = 100

LATENT_FACTOR = 4
INIT_THRESHOLD = 1e-3
STE_BANDWIDTH = 1e-3
L0_COEFF = 1e-3
LR = 1e-3
MEAN_INIT_BATCHES = 16

SEED = 42
HTTP_TIMEOUT = 30
USER_AGENT = "Mozilla/5.0"

OUTPUT_DIR = Path("./outputs/jumprelu_sae_raw_dolma")