import gzip
import json
import urllib.request
from pathlib import Path
from typing import Iterator

from config import (
    DOLMA_SAMPLE_URL,
    HTTP_TIMEOUT,
    MIN_TEXT_CHARS,
    USER_AGENT,
)
from dist_utils import barrier, get_rank, print0


def extract_text_from_example(example: dict) -> str:
    preferred_keys = ["text", "content", "document", "body"]

    for key in preferred_keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for value in example.values():
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError(f"Could not find a text field in example keys: {list(example.keys())}")


def download_url_to_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        with open(dst, "wb") as f:
            f.write(response.read())


def ensure_local_dolma_shard(path: Path) -> Path:
    if get_rank() == 0 and not path.exists():
        print0(f"[rank=0] Downloading Dolma shard to {path}")
        download_url_to_file(DOLMA_SAMPLE_URL, path)

    barrier()

    if not path.exists():
        raise RuntimeError(f"Dolma shard was expected at {path}, but it does not exist.")

    return path


def iter_rank_text_batches(
    shard_path: Path,
    local_batch_size: int,
    rank: int,
    world_size: int,
) -> Iterator[list[str]]:
    """
    Deterministic equal split across ranks.

    Rules:
    - Build complete groups of exactly world_size usable texts.
    - Within each complete group, rank k gets item k.
    - If the final group is incomplete, drop it.
    - Then build per-rank batches of size local_batch_size.
    - If the final batch is incomplete, drop it.

    Result:
    Every rank yields exactly the same number of steps.
    """
    assigned_texts: list[str] = []
    current_group: list[str] = []

    with gzip.open(shard_path, "rt", encoding="utf-8") as gz_file:
        for raw_line in gz_file:
            try:
                row = json.loads(raw_line)
                text = extract_text_from_example(row)
                if len(text) < MIN_TEXT_CHARS:
                    continue
            except Exception:
                continue

            current_group.append(text)

            if len(current_group) == world_size:
                assigned_texts.append(current_group[rank])
                current_group.clear()

                if len(assigned_texts) == local_batch_size:
                    yield assigned_texts
                    assigned_texts = []

    # Intentionally drop incomplete current_group and assigned_texts.