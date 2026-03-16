import argparse
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
    INIT_STATS_CACHE_PATH,
    L0_COEFF,
    L0_WARMUP_STEPS,
    LOG_EVERY,
    LR,
    LR_WARMUP_START_FACTOR,
    LR_WARMUP_STEPS,
    MAX_SEQ_LEN,
    MODEL_NAME,
    OUTPUT_DIR,
    PIN_MEMORY,
    SAE_BATCH_SIZE,
    SAVE_EVERY,
    STE_BANDWIDTH,
    TEXT_BATCH_SIZE_PER_RANK,
    TEXT_PREFETCH_BACKEND,
    TEXT_PREFETCH_BATCHES,
    TRAIN_STEPS,
)
from data import ensure_local_dolma_shard, iter_text_batches
from dist_utils import all_reduce_mean, all_reduce_min_int, cleanup, log0, setup
from init_stats import load_or_compute_init_stats
from sae import TinyJumpReLUSAE, module_of, step_ste


class ActivationBuffer:
    def __init__(self):
        self.chunks: list[torch.Tensor] = []
        self.size = 0

    def add(self, x: torch.Tensor, mask: torch.Tensor, scale: float) -> None:
        valid = x[mask]
        if valid.numel() == 0:
            return
        valid = (valid / scale).detach().to(device="cpu", dtype=torch.float32)
        if PIN_MEMORY:
            valid = valid.pin_memory()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-steps",
        type=int,
        default=TRAIN_STEPS,
        help="Total optimiser steps to reach.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path to resume from.",
    )
    parser.add_argument(
        "--dataset-step",
        type=int,
        default=0,
        help="Number of text batches from iter_text_batches() to skip at the start of the first data pass.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Checkpoint output directory override.",
    )
    return parser.parse_args()


def lr_for_step(step: int, total_train_steps: int) -> float:
    if step <= 0:
        return LR * LR_WARMUP_START_FACTOR

    if LR_WARMUP_STEPS > 0 and step <= LR_WARMUP_STEPS:
        progress = step / LR_WARMUP_STEPS
        factor = LR_WARMUP_START_FACTOR + (1.0 - LR_WARMUP_START_FACTOR) * progress
        return LR * factor

    if total_train_steps <= LR_WARMUP_STEPS:
        return LR

    progress = (step - LR_WARMUP_STEPS) / max(1, total_train_steps - LR_WARMUP_STEPS)
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


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device, non_blocking=True)


def load_resume_checkpoint(
    checkpoint_path: Path,
    sae_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    module_of(sae_model).load_state_dict(payload["sae_state_dict"], strict=True)

    optimizer_state_dict = payload.get("optimizer_state_dict")
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        move_optimizer_state_to_device(optimizer, device)

    start_step = int(payload.get("step", 0))
    activation_scale = float(payload.get("activation_scale", 1.0))
    token_stats = dict(payload.get("token_stats", {}))
    return start_step, activation_scale, token_stats


def build_first_pass_iterator(
    shard_path: Path,
    rank: int,
    world_size: int,
    skip_batches: int,
):
    skipped = 0
    for texts in iter_text_batches(
        shard_path=shard_path,
        local_batch_size=TEXT_BATCH_SIZE_PER_RANK,
        rank=rank,
        world_size=world_size,
        prefetch_batches=TEXT_PREFETCH_BATCHES,
        prefetch_backend=TEXT_PREFETCH_BACKEND,
    ):
        if skipped < skip_batches:
            skipped += 1
            continue
        yield texts


def train() -> None:
    assert BUFFER_CAPACITY % SAE_BATCH_SIZE == 0

    args = parse_args()
    if args.dataset_step < 0:
        raise ValueError("--dataset-step must be >= 0")
    if args.train_steps <= 0:
        raise ValueError("--train-steps must be > 0")

    rank, local_rank, world_size, device, model_dtype = setup()
    log0(f"Rank {rank}/{world_size} | device: {device}")

    resume_checkpoint_path = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    if resume_checkpoint_path is not None and not resume_checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint_path}")

    if args.output_dir is not None:
        run_output_dir = Path(args.output_dir)
    elif resume_checkpoint_path is not None:
        run_output_dir = resume_checkpoint_path.parent
    else:
        run_output_dir = OUTPUT_DIR

    shard_path = ensure_local_dolma_shard(rank)

    if rank == 0:
        run_output_dir.mkdir(parents=True, exist_ok=True)

    activation_model = None

    try:
        activation_model = FrozenActivationModel(
            device=device,
            model_dtype=model_dtype,
            hook_layer_index=HOOK_LAYER_INDEX,
        )

        if resume_checkpoint_path is None:
            mean_vec, activation_scale, token_stats = load_or_compute_init_stats(
                activation_model=activation_model,
                shard_path=shard_path,
                rank=rank,
                world_size=world_size,
                device=device,
                cache_path=INIT_STATS_CACHE_PATH,
            )
            d_in = int(mean_vec.numel())
        else:
            mean_vec = None
            activation_scale = 1.0
            token_stats = {}

            checkpoint_payload = torch.load(resume_checkpoint_path, map_location="cpu")
            state_dict = checkpoint_payload["sae_state_dict"]
            d_in = int(state_dict["b_dec"].numel())
            del checkpoint_payload

        base_sae = TinyJumpReLUSAE(d_in=d_in).to(device=device, dtype=torch.float32)
        if mean_vec is not None:
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

        if resume_checkpoint_path is None:
            start_step = 0
        else:
            start_step, activation_scale, token_stats = load_resume_checkpoint(
                checkpoint_path=resume_checkpoint_path,
                sae_model=sae_model,
                optimizer=optimizer,
                device=device,
            )
            log0(
                f"Resumed checkpoint | path={resume_checkpoint_path} | "
                f"start_step={start_step} | dataset_step={args.dataset_step}"
            )

        if start_step >= args.train_steps:
            log0(
                f"Nothing to do: checkpoint step {start_step} already reached target train steps {args.train_steps}."
            )
            dist.barrier()
            return

        buffer = ActivationBuffer()
        step = start_step
        epoch = 0
        first_epoch_dataset_step = args.dataset_step
        last_log_time = time.perf_counter()

        log0(
            "Training | jumprelu | "
            f"start_step={start_step} | "
            f"target_steps={args.train_steps} | "
            f"dataset_step={args.dataset_step} | "
            f"text_batch={TEXT_BATCH_SIZE_PER_RANK} | "
            f"sae_batch={SAE_BATCH_SIZE} | "
            f"buffer={BUFFER_CAPACITY} | "
            f"activation_scale={activation_scale:.6f} | "
            f"text_prefetch={TEXT_PREFETCH_BACKEND}({TEXT_PREFETCH_BATCHES}) | "
            f"pin_memory={PIN_MEMORY} | "
            f"output_dir={run_output_dir} | "
            f"init_stats_cache={INIT_STATS_CACHE_PATH}"
        )

        while step < args.train_steps:
            epoch += 1
            local_batches_this_epoch = 0

            if epoch == 1:
                text_iter = build_first_pass_iterator(
                    shard_path=shard_path,
                    rank=rank,
                    world_size=world_size,
                    skip_batches=first_epoch_dataset_step,
                )
            else:
                text_iter = iter_text_batches(
                    shard_path=shard_path,
                    local_batch_size=TEXT_BATCH_SIZE_PER_RANK,
                    rank=rank,
                    world_size=world_size,
                    prefetch_batches=TEXT_PREFETCH_BATCHES,
                    prefetch_backend=TEXT_PREFETCH_BACKEND,
                )

            for texts in text_iter:
                local_batches_this_epoch += 1

                act, mask = activation_model.capture_text_batch(texts)
                x = act.reshape(-1, d_in)
                mask_flat = mask.reshape(-1)
                buffer.add(x, mask_flat, activation_scale)

                if not buffer.ready(device):
                    continue

                for x_batch in buffer.pop_batches(device):
                    current_step = step + 1
                    current_lr = lr_for_step(current_step, args.train_steps)
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

                    if step == start_step + 1 or step % LOG_EVERY == 0 or step == args.train_steps:
                        now = time.perf_counter()
                        dt = max(now - last_log_time, 1e-8)
                        steps_per_sec = LOG_EVERY / dt if step > start_step + 1 else 0.0
                        last_log_time = now

                        recon_mean = float(all_reduce_mean(recon_loss).item())
                        l0_mean = float(all_reduce_mean(l0).item())
                        theta_mean = float(sae_base.get_threshold().mean().item())
                        active_frac = l0_mean / max(sae_base.d_latent, 1)
                        max_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

                        log0(
                            f"step={step:06d}/{args.train_steps} "
                            f"epoch={epoch} "
                            f"lr={current_lr:.2e} "
                            f"lambda={current_l0_coeff:.2e} "
                            f"recon={recon_mean:.6f} "
                            f"avg_l0={l0_mean:.1f} "
                            f"active_frac={100.0 * active_frac:.2f}% "
                            f"theta_mean={theta_mean:.6f} "
                            f"steps_per_sec={steps_per_sec:.2f} "
                            f"max_mem_gb={max_mem_gb:.2f}"
                        )

                    if step % SAVE_EVERY == 0 or step == args.train_steps:
                        save_checkpoint(
                            run_output_dir / f"sae_step_{step:06d}.pt",
                            step,
                            sae_model,
                            optimizer,
                            activation_scale,
                            token_stats,
                        )

                    if step >= args.train_steps:
                        break

                if step >= args.train_steps:
                    break

            if local_batches_this_epoch == 0:
                raise RuntimeError(
                    "No complete per-rank batches were produced from the Dolma shard after applying dataset_step."
                )

        dist.barrier()
        log0(f"TRAINING COMPLETE. Outputs saved under: {run_output_dir}")

    finally:
        if activation_model is not None:
            activation_model.close()
        cleanup()


if __name__ == "__main__":
    train()
