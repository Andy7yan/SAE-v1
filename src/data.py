import gzip
import json
import multiprocessing as mp
import queue
import threading
import traceback
import urllib.request
from pathlib import Path

from config import (
    DATA_CACHE_PATH,
    DOLMA_URL,
    HTTP_TIMEOUT,
    MIN_TEXT_CHARS,
    TEXT_PREFETCH_BACKEND,
    TEXT_PREFETCH_BATCHES,
    USER_AGENT,
)
from dist_utils import barrier, log0


def extract_text(row: dict) -> str:
    value = row.get("text", "")
    if isinstance(value, str):
        return value.strip()
    return ""


def ensure_local_dolma_shard(rank: int) -> Path:
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
    assigned: list[str] = []
    group: list[str] = []

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


def _process_prefetch_producer(
    shard_path_str: str,
    local_batch_size: int,
    rank: int,
    world_size: int,
    out_queue,
) -> None:
    try:
        for batch in iter_rank_text_batches(
            shard_path=Path(shard_path_str),
            local_batch_size=local_batch_size,
            rank=rank,
            world_size=world_size,
        ):
            out_queue.put(("batch", batch))
    except BaseException:
        out_queue.put(("error", traceback.format_exc()))
        return

    out_queue.put(("done", None))


def iter_thread_prefetched_rank_text_batches(
    shard_path: Path,
    local_batch_size: int,
    rank: int,
    world_size: int,
    prefetch_batches: int,
):
    out_queue: queue.Queue = queue.Queue(maxsize=max(1, prefetch_batches))
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
        except BaseException:
            put_item("error", traceback.format_exc())
            return

        put_item("done", None)

    worker = threading.Thread(
        target=producer,
        name=f"rank_text_prefetch_thread_r{rank}",
        daemon=True,
    )
    worker.start()

    try:
        while True:
            try:
                tag, payload = out_queue.get(timeout=0.1)
            except queue.Empty:
                if not worker.is_alive():
                    raise RuntimeError("Background text prefetch thread exited unexpectedly.")
                continue

            if tag == "batch":
                yield payload
                continue

            if tag == "error":
                raise RuntimeError(f"Background text prefetch thread failed on rank {rank}:\n{payload}")

            if tag == "done":
                break

            raise RuntimeError(f"Unknown prefetch queue tag: {tag}")
    finally:
        stop_event.set()
        worker.join(timeout=1.0)


def iter_process_prefetched_rank_text_batches(
    shard_path: Path,
    local_batch_size: int,
    rank: int,
    world_size: int,
    prefetch_batches: int,
):
    ctx = mp.get_context("spawn")
    out_queue = ctx.Queue(maxsize=max(1, prefetch_batches))
    worker = ctx.Process(
        target=_process_prefetch_producer,
        args=(str(shard_path), local_batch_size, rank, world_size, out_queue),
        name=f"rank_text_prefetch_process_r{rank}",
        daemon=True,
    )
    worker.start()

    try:
        while True:
            try:
                tag, payload = out_queue.get(timeout=0.1)
            except queue.Empty:
                if not worker.is_alive():
                    exit_code = worker.exitcode
                    if exit_code == 0:
                        raise RuntimeError(
                            "Background text prefetch process exited without signalling completion."
                        )
                    raise RuntimeError(
                        f"Background text prefetch process exited unexpectedly with code {exit_code}."
                    )
                continue

            if tag == "batch":
                yield payload
                continue

            if tag == "error":
                raise RuntimeError(f"Background text prefetch process failed on rank {rank}:\n{payload}")

            if tag == "done":
                break

            raise RuntimeError(f"Unknown prefetch queue tag: {tag}")
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=1.0)
        out_queue.close()
        out_queue.join_thread()


def iter_text_batches(
    shard_path: Path,
    local_batch_size: int,
    rank: int,
    world_size: int,
    prefetch_batches: int = TEXT_PREFETCH_BATCHES,
    prefetch_backend: str = TEXT_PREFETCH_BACKEND,
):
    backend = prefetch_backend.strip().lower()

    if prefetch_batches <= 0 or backend == "none":
        yield from iter_rank_text_batches(
            shard_path=shard_path,
            local_batch_size=local_batch_size,
            rank=rank,
            world_size=world_size,
        )
        return

    if backend == "thread":
        yield from iter_thread_prefetched_rank_text_batches(
            shard_path=shard_path,
            local_batch_size=local_batch_size,
            rank=rank,
            world_size=world_size,
            prefetch_batches=prefetch_batches,
        )
        return

    if backend == "process":
        yield from iter_process_prefetched_rank_text_batches(
            shard_path=shard_path,
            local_batch_size=local_batch_size,
            rank=rank,
            world_size=world_size,
            prefetch_batches=prefetch_batches,
        )
        return

    raise ValueError(
        f"Unsupported TEXT_PREFETCH_BACKEND={prefetch_backend!r}. Use 'none', 'thread', or 'process'."
    )