import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from activation_store import FrozenActivationModel
from config import (
    ADAM_BETA1,
    ADAM_BETA2,
    ADAM_EPS,
    BUFFER_CAPACITY,
    HOOK_LAYER_INDEX,
    L0_COEFF,
    L0_WARMUP_STEPS,
    LOG_EVERY,
    LR,
    LR_WARMUP_START_FACTOR,
    LR_WARMUP_STEPS,
    MAX_SEQ_LEN,
    MODEL_NAME,
    OUTPUT_DIR,
    SAE_BATCH_SIZE,
    SAVE_EVERY,
    STE_BANDWIDTH,
    TEXT_BATCH_SIZE_PER_RANK,
    TRAIN_STEPS,
)
from data import ensure_local_dolma_shard, iter_prefetched_rank_text_batches, iter_rank_text_batches
from dist_utils import all_reduce_mean, all_reduce_min_int, cleanup, log0, setup
from init_stats import estimate_activation_stats, profile_token_lengths
from sae import TinyJumpReLUSAE, module_of, step_ste


class ActivationBuffer:
    def __init__(self):
        self.chunks: list[torch.Tensor] = []
        self.size = 0

    def add(self, x: torch.Tensor, mask: torch.Tensor, scale: float) -> None:
        valid = x[mask]
        if valid.numel() == 0:
            return
        valid = (valid / scale).detach().cpu()
        self.chunks.append(valid)
        self.size += valid.shape[0]

    def ready(self, device: torch.device) -> bool:
        return all_reduce_min_int(self.size, device) >= BUFFER_CAPACITY

    def pop_batches(self, device: torch.device):
        all_x = torch.cat(self.chunks, dim=0)
        perm = torch.randperm(all_x.shape[0])
        all_x = all_x[perm]

        take = all_x[:BUFFER_CAPACITY]
        left = all_x[BUFFER_CAPACITY:]

        self.chunks = [left] if left.shape[0] > 0 else []
        self.size = int(left.shape[0])

        for i in range(0, BUFFER_CAPACITY, SAE_BATCH_SIZE):
            yield take[i : i + SAE_BATCH_SIZE].to(device, non_blocking=True)


def lr_for_step(step: int) -> float:
    if step <= 0:
        return LR * LR_WARMUP_START_FACTOR

    if LR_WARMUP_STEPS > 0 and step <= LR_WARMUP_STEPS:
        progress = step / LR_WARMUP_STEPS
        factor = LR_WARMUP_START_FACTOR + (1.0 - LR_WARMUP_START_FACTOR) * progress
        return LR * factor

    if TRAIN_STEPS <= LR_WARMUP_STEPS:
        return LR

    progress = (step - LR_WARMUP_STEPS) / max(1, TRAIN_STEPS - LR_WARMUP_STEPS)
    progress = min(max(progress, 0.0), 1.0)
    return LR * 0.5 * (1.0 + math.cos(math.pi * progress))


def l0_coeff_for_step(step: int) -> float:
    if L0_WARMUP_STEPS <= 0:
        return L0_COEFF
    progress = min(max(step / L0_WARMUP_STEPS, 0.0), 1.0)
    return L0_COEFF * progress


def save_checkpoint(
    path: Path,
    step: int,
    sae_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    activation_scale: float,
    token_stats: dict,
) -> None:
    if dist.get_rank() != 0:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "sae_state_dict": module_of(sae_model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_name": MODEL_NAME,
            "hook_layer_index": HOOK_LAYER_INDEX,
            "max_seq_len": MAX_SEQ_LEN,
            "activation_scale": activation_scale,
            "token_stats": token_stats,
        },
        path,
    )


def train() -> None:
    assert BUFFER_CAPACITY % SAE_BATCH_SIZE == 0

    rank, local_rank, world_size, device, model_dtype = setup()
    log0(f"Rank {rank}/{world_size} | device: {device}")

    shard_path = ensure_local_dolma_shard(rank)

    if rank == 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    activation_model = None

    try:
        activation_model = FrozenActivationModel(
            device=device,
            model_dtype=model_dtype,
            hook_layer_index=HOOK_LAYER_INDEX,
        )

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

        d_in = int(mean_vec.numel())

        base_sae = TinyJumpReLUSAE(d_in=d_in).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            base_sae.b_dec.copy_((mean_vec / activation_scale).to(device=device, dtype=torch.float32))

        sae_model = DDP(
            base_sae,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

        optimizer = torch.optim.Adam(
            sae_model.parameters(),
            lr=LR * LR_WARMUP_START_FACTOR,
            betas=(ADAM_BETA1, ADAM_BETA2),
            eps=ADAM_EPS,
        )

        buffer = ActivationBuffer()
        step = 0
        epoch = 0
        last_log_time = time.perf_counter()

        log0(
            "Training | jumprelu | "
            f"total_steps={TRAIN_STEPS} | "
            f"text_batch={TEXT_BATCH_SIZE_PER_RANK} | "
            f"sae_batch={SAE_BATCH_SIZE} | "
            f"buffer={BUFFER_CAPACITY} | "
            f"activation_scale={activation_scale:.6f} | "
            "text_prefetch=on"
        )

        while step < TRAIN_STEPS:
            epoch += 1
            local_batches_this_epoch = 0

            for texts in iter_prefetched_rank_text_batches(
                shard_path=shard_path,
                local_batch_size=TEXT_BATCH_SIZE_PER_RANK,
                rank=rank,
                world_size=world_size,
                prefetch_batches=4,
            ):
                local_batches_this_epoch += 1

                act, mask = activation_model.capture_text_batch(texts)
                x = act.reshape(-1, d_in)
                mask_flat = mask.reshape(-1)
                buffer.add(x, mask_flat, activation_scale)

                if not buffer.ready(device):
                    continue

                for x_batch in buffer.pop_batches(device):
                    current_step = step + 1
                    current_lr = lr_for_step(current_step)
                    current_l0_coeff = l0_coeff_for_step(current_step)

                    for group in optimizer.param_groups:
                        group["lr"] = current_lr

                    x_hat, pre = sae_model(x_batch)
                    sae_base: TinyJumpReLUSAE = module_of(sae_model)

                    recon_loss = ((x_hat - x_batch) ** 2).mean()
                    l0 = step_ste(pre, sae_base.get_threshold(), STE_BANDWIDTH).sum(dim=-1).mean()
                    loss = recon_loss + current_l0_coeff * l0

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    sae_base.remove_decoder_grad_parallel()
                    optimizer.step()
                    sae_base.normalise_decoder()

                    step = current_step

                    if step == 1 or step % LOG_EVERY == 0 or step == TRAIN_STEPS:
                        now = time.perf_counter()
                        dt = max(now - last_log_time, 1e-8)
                        steps_per_sec = LOG_EVERY / dt if step > 1 else 0.0
                        last_log_time = now

                        recon_mean = float(all_reduce_mean(recon_loss).item())
                        l0_mean = float(all_reduce_mean(l0).item())
                        theta_mean = float(sae_base.get_threshold().mean().item())
                        max_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

                        log0(
                            f"step={step:06d}/{TRAIN_STEPS} "
                            f"epoch={epoch} "
                            f"lr={current_lr:.2e} "
                            f"lambda={current_l0_coeff:.2e} "
                            f"recon={recon_mean:.6f} "
                            f"avg_l0={l0_mean:.3f} "
                            f"theta_mean={theta_mean:.6f} "
                            f"steps_per_sec={steps_per_sec:.2f} "
                            f"max_mem_gb={max_mem_gb:.2f}"
                        )

                    if step % SAVE_EVERY == 0 or step == TRAIN_STEPS:
                        save_checkpoint(
                            OUTPUT_DIR / f"sae_step_{step:06d}.pt",
                            step,
                            sae_model,
                            optimizer,
                            activation_scale,
                            token_stats,
                        )

                    if step >= TRAIN_STEPS:
                        break

                if step >= TRAIN_STEPS:
                    break

            if local_batches_this_epoch == 0:
                raise RuntimeError("No complete per-rank batches were produced from the Dolma shard.")

        dist.barrier()
        log0(f"TRAINING COMPLETE. Outputs saved under: {OUTPUT_DIR}")

    finally:
        if activation_model is not None:
            activation_model.close()
        cleanup()


if __name__ == "__main__":
    train()