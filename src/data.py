import gzip
import json
import queue
import threading
import urllib.request
from pathlib import Path

from config import DATA_CACHE_PATH, DOLMA_URL, HTTP_TIMEOUT, MIN_TEXT_CHARS, USER_AGENT
from dist_utils import barrier, log0


def extract_text(row: dict) -> str:
    value = row.get("text", "")
    if isinstance(value, str):
        return value.strip()
    return ""


def ensure_local_dolma_shard(rank: int):
    if rank == 0 and not DATA_CACHE_PATH.exists():
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


def iter_prefetched_rank_text_batches(
    shard_path: Path,
    local_batch_size: int,
    rank: int,
    world_size: int,
    prefetch_batches: int = 4,
):
    if prefetch_batches <= 0:
        yield from iter_rank_text_batches(
            shard_path=shard_path,
            local_batch_size=local_batch_size,
            rank=rank,
            world_size=world_size,
        )
        return

    out_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=max(1, prefetch_batches))
    stop_event = threading.Event()

    def put_item(tag: str, payload: object) -> bool:
        while not stop_event.is_set():
            try:
                out_queue.put((tag, payload), timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def producer() -> None:
        try:
            for batch in iter_rank_text_batches(
                shard_path=shard_path,
                local_batch_size=local_batch_size,
                rank=rank,
                world_size=world_size,
            ):
                if not put_item("batch", batch):
                    return
        except BaseException as exc:
            put_item("error", exc)
            return

        put_item("done", None)

    worker = threading.Thread(
        target=producer,
        name=f"rank_text_prefetch_r{rank}",
        daemon=True,
    )
    worker.start()

    try:
        while True:
            tag, payload = out_queue.get()

            if tag == "batch":
                yield payload
                continue

            if tag == "error":
                raise RuntimeError("Background text prefetch failed.") from payload

            if tag == "done":
                break

            raise RuntimeError(f"Unknown prefetch queue tag: {tag}")
    finally:
        stop_event.set()
        worker.join(timeout=1.0)