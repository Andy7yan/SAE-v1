from pathlib import Path

import torch
import torch.distributed as dist

from config import (
    HOOK_LAYER_INDEX,
    INIT_STATS_CACHE_PATH,
    MAX_SEQ_LEN,
    MEAN_INIT_BATCHES,
    MODEL_NAME,
    TEXT_BATCH_SIZE_PER_RANK,
    TEXT_PREFETCH_BACKEND,
    TEXT_PREFETCH_BATCHES,
    TOKEN_STATS_BATCHES,
)
from data import iter_text_batches
from dist_utils import all_reduce_sum, barrier, log0


def profile_token_lengths(
    activation_model,
    shard_path,
    rank: int,
    world_size: int,
    device: torch.device,
):
    raw_tokens_sum = torch.zeros((), device=device, dtype=torch.float64)
    valid_tokens_sum = torch.zeros((), device=device, dtype=torch.float64)
    sequence_count = torch.zeros((), device=device, dtype=torch.float64)
    truncation_count = torch.zeros((), device=device, dtype=torch.float64)
    used = 0

    for texts in iter_text_batches(
        shard_path=shard_path,
        local_batch_size=TEXT_BATCH_SIZE_PER_RANK,
        rank=rank,
        world_size=world_size,
        prefetch_batches=TEXT_PREFETCH_BATCHES,
        prefetch_backend=TEXT_PREFETCH_BACKEND,
    ):
        stats = activation_model.length_stats(texts)

        raw_tokens_sum += torch.tensor(sum(stats["raw_lengths"]), device=device, dtype=torch.float64)
        valid_tokens_sum += torch.tensor(sum(stats["valid_lengths"]), device=device, dtype=torch.float64)
        sequence_count += torch.tensor(len(stats["raw_lengths"]), device=device, dtype=torch.float64)
        truncation_count += torch.tensor(stats["truncation_count"], device=device, dtype=torch.float64)
        used += 1

        if used >= TOKEN_STATS_BATCHES:
            break

    raw_tokens_sum = all_reduce_sum(raw_tokens_sum)
    valid_tokens_sum = all_reduce_sum(valid_tokens_sum)
    sequence_count = all_reduce_sum(sequence_count).clamp(min=1.0)
    truncation_count = all_reduce_sum(truncation_count)

    mean_raw_tokens = float((raw_tokens_sum / sequence_count).item())
    mean_valid_tokens = float((valid_tokens_sum / sequence_count).item())
    truncation_rate = float((truncation_count / sequence_count).item())

    log0(
        "Token profile | "
        f"batches={used} | "
        f"mean_raw_tokens={mean_raw_tokens:.2f} | "
        f"mean_valid_tokens={mean_valid_tokens:.2f} | "
        f"truncation_rate={100.0 * truncation_rate:.2f}%"
    )

    return {
        "mean_raw_tokens": mean_raw_tokens,
        "mean_valid_tokens": mean_valid_tokens,
        "truncation_rate": truncation_rate,
        "batches_used": used,
    }


def estimate_activation_stats(
    activation_model,
    shard_path,
    rank: int,
    world_size: int,
    device: torch.device,
):
    total_sum = None
    total_sq_sum = torch.zeros((), device=device, dtype=torch.float64)
    total_count = torch.zeros((), device=device, dtype=torch.float64)
    used = 0

    for texts in iter_text_batches(
        shard_path=shard_path,
        local_batch_size=TEXT_BATCH_SIZE_PER_RANK,
        rank=rank,
        world_size=world_size,
        prefetch_batches=TEXT_PREFETCH_BATCHES,
        prefetch_backend=TEXT_PREFETCH_BACKEND,
    ):
        act, mask = activation_model.capture_text_batch(texts)
        d_in = act.shape[-1]

        x = act.reshape(-1, d_in)
        mask_flat = mask.reshape(-1).to(device)
        valid = x[mask_flat]

        if valid.numel() == 0:
            continue

        batch_sum = valid.sum(dim=0, dtype=torch.float64)
        batch_sq_sum = valid.pow(2).sum(dtype=torch.float64)
        batch_count = torch.tensor(valid.shape[0], device=device, dtype=torch.float64)

        if total_sum is None:
            total_sum = torch.zeros(d_in, device=device, dtype=torch.float64)

        total_sum += batch_sum
        total_sq_sum += batch_sq_sum
        total_count += batch_count
        used += 1

        if used >= MEAN_INIT_BATCHES:
            break

    if total_sum is None:
        raise RuntimeError("No activation batches were available for initial statistics.")

    total_sum = all_reduce_sum(total_sum)
    total_sq_sum = all_reduce_sum(total_sq_sum)
    total_count = all_reduce_sum(total_count).clamp(min=1.0)

    mean = (total_sum / total_count).to(torch.float32)
    d_in = mean.numel()
    rms_scale = torch.sqrt(total_sq_sum / (total_count * d_in)).clamp(min=1e-8).to(torch.float32)

    log0(
        "Activation stats done | "
        f"batches={used} | "
        f"valid_tokens={int(total_count.item())} | "
        f"d_in={d_in} | "
        f"rms_scale={float(rms_scale.item()):.6f}"
    )

    return mean, float(rms_scale.item())


def _save_init_stats_cache(
    cache_path: Path,
    mean_vec: torch.Tensor,
    activation_scale: float,
    token_stats: dict,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "mean_vec": mean_vec.detach().cpu().to(torch.float32),
        "activation_scale": float(activation_scale),
        "token_stats": token_stats,
        "meta": {
            "model_name": MODEL_NAME,
            "hook_layer_index": HOOK_LAYER_INDEX,
            "max_seq_len": MAX_SEQ_LEN,
            "text_batch_size_per_rank": TEXT_BATCH_SIZE_PER_RANK,
            "token_stats_batches": TOKEN_STATS_BATCHES,
            "mean_init_batches": MEAN_INIT_BATCHES,
        },
    }
    torch.save(payload, cache_path)


def _load_init_stats_cache(
    cache_path: Path,
    device: torch.device,
):
    payload = torch.load(cache_path, map_location="cpu")
    mean_vec = payload["mean_vec"].to(device=device, dtype=torch.float32)
    activation_scale = float(payload["activation_scale"])
    token_stats = dict(payload["token_stats"])
    meta = dict(payload.get("meta", {}))
    return mean_vec, activation_scale, token_stats, meta


def load_or_compute_init_stats(
    activation_model,
    shard_path,
    rank: int,
    world_size: int,
    device: torch.device,
    cache_path: Path = INIT_STATS_CACHE_PATH,
):
    if rank == 0:
        cache_exists = cache_path.exists()
    else:
        cache_exists = False

    flag = torch.tensor(1 if cache_exists else 0, device=device, dtype=torch.int32)
    dist.broadcast(flag, src=0)
    cache_exists = bool(flag.item())

    if cache_exists:
        mean_vec, activation_scale, token_stats, _meta = _load_init_stats_cache(
            cache_path=cache_path,
            device=device,
        )
        log0(f"Loaded init stats cache from: {cache_path}")
        return mean_vec, activation_scale, token_stats

    log0(f"Init stats cache not found. Recomputing and saving to: {cache_path}")

    token_stats = profile_token_lengths(
        activation_model=activation_model,
        shard_path=shard_path,
        rank=rank,
        world_size=world_size,
        device=device,
    )

    mean_vec, activation_scale = estimate_activation_stats(
        activation_model=activation_model,
        shard_path=shard_path,
        rank=rank,
        world_size=world_size,
        device=device,
    )

    if rank == 0:
        _save_init_stats_cache(
            cache_path=cache_path,
            mean_vec=mean_vec,
            activation_scale=activation_scale,
            token_stats=token_stats,
        )
        log0(f"Saved init stats cache to: {cache_path}")

    barrier()
    return mean_vec, activation_scale, token_stats