import gzip
import json
import urllib.request
from pathlib import Path

from config import DATA_CACHE_PATH, DOLMA_URL, HTTP_TIMEOUT, MIN_TEXT_CHARS, USER_AGENT
from dist_utils import barrier, log0


def extract_text(row: dict) -> str:
    value = row.get("text", "")
    if isinstance(value, str):
        return value.strip()
    return ""


def ensure_local_dolma_shard():
    if not DATA_CACHE_PATH.exists():
        DATA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        log0(f"Downloading Dolma shard to {DATA_CACHE_PATH} ...")
        request = urllib.request.Request(DOLMA_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response, open(DATA_CACHE_PATH, "wb") as f:
            f.write(response.read())
    barrier()
    if not DATA_CACHE_PATH.exists():
        raise RuntimeError(f"Missing shard: {DATA_CACHE_PATH}")
    return DATA_CACHE_PATH


def iter_rank_text_batches(shard_path: Path, local_batch_size: int, rank: int, world_size: int):
    assigned = []
    group = []

    with gzip.open(shard_path, "rt", encoding="utf-8") as gz_file:
        for raw_line in gz_file:
            try:
                row = json.loads(raw_line)
            except Exception:
                continue

            text = extract_text(row)
            if not text or len(text) < MIN_TEXT_CHARS:
                continue

            group.append(text)

            if len(group) == world_size:
                assigned.append(group[rank])
                group.clear()

                if len(assigned) == local_batch_size:
                    yield assigned
                    assigned = []