import torch

from config import MEAN_INIT_BATCHES, TEXT_BATCH_SIZE_PER_RANK
from data import iter_rank_text_batches
from dist_utils import all_reduce_sum, log0


def estimate_activation_mean(
    activation_model,
    shard_path,
    rank: int,
    world_size: int,
    device: torch.device,
):
    total_sum = None
    total_count = torch.zeros((), device=device, dtype=torch.float32)
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

        batch_sum = x[mask_flat].sum(dim=0)
        batch_count = mask_flat.sum()

        if total_sum is None:
            total_sum = torch.zeros(d_in, device=device, dtype=torch.float32)

        total_sum += batch_sum
        total_count += batch_count
        used += 1

        if used >= MEAN_INIT_BATCHES:
            break

    if total_sum is None:
        raise RuntimeError("No activation batches were available for mean initialisation.")

    total_sum = all_reduce_sum(total_sum)
    total_count = all_reduce_sum(total_count).clamp(min=1.0)
    mean = total_sum / total_count

    log0(
        f"Mean initialisation done | batches={used} | valid_tokens={int(total_count.item())} | d_in={mean.numel()}"
    )

    return mean