import torch

from config import MEAN_INIT_BATCHES, TEXT_BATCH_SIZE_PER_RANK, TOKEN_STATS_BATCHES
from data import iter_rank_text_batches
from dist_utils import all_reduce_sum, log0


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

    for texts in iter_rank_text_batches(
        shard_path=shard_path,
        local_batch_size=TEXT_BATCH_SIZE_PER_RANK,
        rank=rank,
        world_size=world_size,
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
        f"batches={used} | mean_raw_tokens={mean_raw_tokens:.2f} | "
        f"mean_valid_tokens={mean_valid_tokens:.2f} | truncation_rate={100.0 * truncation_rate:.2f}%"
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

    for texts in iter_rank_text_batches(
        shard_path=shard_path,
        local_batch_size=TEXT_BATCH_SIZE_PER_RANK,
        rank=rank,
        world_size=world_size,
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
        f"batches={used} | valid_tokens={int(total_count.item())} | d_in={d_in} | rms_scale={float(rms_scale.item()):.6f}"
    )

    return mean, float(rms_scale.item())