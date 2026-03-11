import torch

from data import iter_rank_text_batches
from dist_utils import all_reduce_sum, print0


def estimate_activation_mean(
    activation_model,
    shard_path,
    local_batch_size: int,
    num_batches: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Estimate the mean activation vector over valid tokens only.
    This will be used to initialise SAE.b_dec before training.
    """
    local_sum: torch.Tensor | None = None
    local_count = torch.zeros((), device=device, dtype=torch.float32)

    batches_used = 0

    for texts in iter_rank_text_batches(
        shard_path=shard_path,
        local_batch_size=local_batch_size,
        rank=rank,
        world_size=world_size,
    ):
        act, attention_mask = activation_model.capture_text_batch(texts)

        if act.ndim != 3:
            raise RuntimeError(f"Expected hooked activation to be 3D, got {tuple(act.shape)}")

        d_in = int(act.shape[-1])

        x = act.to(dtype=torch.float32).reshape(-1, d_in)
        mask_flat = attention_mask.reshape(-1).to(device=device, dtype=torch.float32)

        batch_sum = (x * mask_flat.unsqueeze(-1)).sum(dim=0)
        batch_count = mask_flat.sum()

        if local_sum is None:
            local_sum = torch.zeros(d_in, device=device, dtype=torch.float32)

        local_sum += batch_sum
        local_count += batch_count

        batches_used += 1
        if batches_used >= num_batches:
            break

    if local_sum is None:
        raise RuntimeError(
            "No batches were available for activation-mean estimation. "
            "Reduce local_batch_size, reduce world_size, or use a larger shard."
        )

    global_sum = all_reduce_sum(local_sum)
    global_count = all_reduce_sum(local_count).clamp(min=1.0)

    mean_vec = global_sum / global_count

    print0(
        f"[rank=0] Mean initialisation done: "
        f"batches_used={batches_used}, "
        f"global_valid_tokens={int(global_count.item())}, "
        f"d_in={mean_vec.numel()}"
    )

    return mean_vec