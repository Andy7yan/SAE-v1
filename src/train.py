import math
import time
from dataclasses import dataclass
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
    CHECKPOINT_DIR,
    FINAL_WEIGHTS_PATH,
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
    THRESHOLD_LOG_QUANTILES,
    TRAIN_STEPS,
    WEIGHTS_DIR,
)
from data import ensure_local_dolma_shard, iter_text_batches
from dist_utils import all_reduce_min_int, cleanup, log0, setup
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

    def pop_cpu_batches(self):
        all_x = torch.cat(self.chunks, dim=0)
        perm = torch.randperm(all_x.shape[0])
        all_x = all_x[perm]

        take = all_x[:BUFFER_CAPACITY]
        left = all_x[BUFFER_CAPACITY:]

        self.chunks = [left] if left.shape[0] > 0 else []
        self.size = int(left.shape[0])

        for i in range(0, BUFFER_CAPACITY, SAE_BATCH_SIZE):
            yield take[i : i + SAE_BATCH_SIZE]


@dataclass
class MonitorWindow:
    step_count: int = 0
    text_batch_count: int = 0
    buffer_not_ready_count: int = 0
    captured_valid_tokens: int = 0
    sae_tokens: int = 0
    recon_sum: float = 0.0
    l0_sum: float = 0.0
    capture_time_s: float = 0.0
    h2d_time_s: float = 0.0
    train_step_time_s: float = 0.0

    def add_text_batch(self, valid_tokens: int, capture_time_s: float, buffer_ready: bool) -> None:
        self.text_batch_count += 1
        self.captured_valid_tokens += int(valid_tokens)
        self.capture_time_s += float(capture_time_s)
        if not buffer_ready:
            self.buffer_not_ready_count += 1

    def add_train_step(
        self,
        recon_loss: float,
        l0_value: float,
        h2d_time_s: float,
        train_step_time_s: float,
        sae_tokens: int,
    ) -> None:
        self.step_count += 1
        self.recon_sum += float(recon_loss)
        self.l0_sum += float(l0_value)
        self.h2d_time_s += float(h2d_time_s)
        self.train_step_time_s += float(train_step_time_s)
        self.sae_tokens += int(sae_tokens)

    def reset(self) -> None:
        self.step_count = 0
        self.text_batch_count = 0
        self.buffer_not_ready_count = 0
        self.captured_valid_tokens = 0
        self.sae_tokens = 0
        self.recon_sum = 0.0
        self.l0_sum = 0.0
        self.capture_time_s = 0.0
        self.h2d_time_s = 0.0
        self.train_step_time_s = 0.0


class CUDATimer:
    def __init__(self, device: torch.device):
        self.device = device

    def measure(self, fn):
        if self.device.type != "cuda":
            start = time.perf_counter()
            out = fn()
            end = time.perf_counter()
            return out, end - start

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        out = fn()
        end_event.record()
        end_event.synchronize()
        elapsed_s = start_event.elapsed_time(end_event) / 1000.0
        return out, elapsed_s


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


def all_reduce_sum_float(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def all_reduce_sum_int(value: int, device: torch.device) -> int:
    tensor = torch.tensor(value, device=device, dtype=torch.int64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


@torch.no_grad()
def get_threshold_stats(sae_base: TinyJumpReLUSAE) -> dict[str, float]:
    threshold = sae_base.get_threshold()
    quantiles = torch.tensor(THRESHOLD_LOG_QUANTILES, device=threshold.device, dtype=threshold.dtype)
    q = torch.quantile(threshold, quantiles)
    return {
        "thr_min": float(threshold.min().item()),
        "thr_p05": float(q[0].item()),
        "thr_p50": float(q[1].item()),
        "thr_p95": float(q[2].item()),
        "thr_max": float(threshold.max().item()),
        "thr_mean": float(threshold.mean().item()),
    }


@torch.no_grad()
def get_gpu_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"peak_alloc_gb": 0.0, "peak_reserved_gb": 0.0}

    return {
        "peak_alloc_gb": torch.cuda.max_memory_allocated(device) / (1024 ** 3),
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
    }


@torch.no_grad()
def reset_gpu_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def build_common_payload(
    step: int,
    sae_model: nn.Module,
    activation_scale: float,
    token_stats: dict,
) -> dict:
    return {
        "step": step,
        "sae_state_dict": module_of(sae_model).state_dict(),
        "model_name": MODEL_NAME,
        "hook_layer_index": HOOK_LAYER_INDEX,
        "max_seq_len": MAX_SEQ_LEN,
        "activation_scale": activation_scale,
        "token_stats": token_stats,
    }


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

    payload = build_common_payload(
        step=step,
        sae_model=sae_model,
        activation_scale=activation_scale,
        token_stats=token_stats,
    )
    payload["optimizer_state_dict"] = optimizer.state_dict()

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def save_final_weights(
    path: Path,
    step: int,
    sae_model: nn.Module,
    activation_scale: float,
    token_stats: dict,
) -> None:
    if dist.get_rank() != 0:
        return

    payload = build_common_payload(
        step=step,
        sae_model=sae_model,
        activation_scale=activation_scale,
        token_stats=token_stats,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train() -> None:
    assert BUFFER_CAPACITY % SAE_BATCH_SIZE == 0

    rank, local_rank, world_size, device, model_dtype = setup()
    log0(f"Rank {rank}/{world_size} | device: {device}")

    shard_path = ensure_local_dolma_shard(rank)

    if rank == 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    activation_model = None

    try:
        activation_model = FrozenActivationModel(
            device=device,
            model_dtype=model_dtype,
            hook_layer_index=HOOK_LAYER_INDEX,
        )

        mean_vec, activation_scale, token_stats = load_or_compute_init_stats(
            activation_model=activation_model,
            shard_path=shard_path,
            rank=rank,
            world_size=world_size,
            device=device,
            cache_path=INIT_STATS_CACHE_PATH,
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
        timer = CUDATimer(device=device)
        monitor = MonitorWindow()
        step = 0
        epoch = 0
        last_log_time = time.perf_counter()
        reset_gpu_peak_memory_stats(device)

        log0(
            "Training | sae-v1 | "
            f"total_steps={TRAIN_STEPS} | "
            f"text_batch={TEXT_BATCH_SIZE_PER_RANK} | "
            f"sae_batch={SAE_BATCH_SIZE} | "
            f"buffer={BUFFER_CAPACITY} | "
            f"activation_scale={activation_scale:.6f} | "
            f"text_prefetch={TEXT_PREFETCH_BACKEND}({TEXT_PREFETCH_BATCHES}) | "
            f"pin_memory={PIN_MEMORY} | "
            f"log_every={LOG_EVERY} | "
            f"checkpoint_dir={CHECKPOINT_DIR} | "
            f"final_weights={FINAL_WEIGHTS_PATH} | "
            f"init_stats_cache={INIT_STATS_CACHE_PATH}"
        )

        while step < TRAIN_STEPS:
            epoch += 1
            local_batches_this_epoch = 0

            for texts in iter_text_batches(
                shard_path=shard_path,
                local_batch_size=TEXT_BATCH_SIZE_PER_RANK,
                rank=rank,
                world_size=world_size,
                prefetch_batches=TEXT_PREFETCH_BATCHES,
                prefetch_backend=TEXT_PREFETCH_BACKEND,
            ):
                local_batches_this_epoch += 1

                (act, mask), capture_time_s = timer.measure(
                    lambda: activation_model.capture_text_batch(texts)
                )
                x = act.reshape(-1, d_in)
                mask_flat = mask.reshape(-1)
                valid_tokens = int(mask_flat.sum().item())
                buffer.add(x, mask_flat, activation_scale)

                buffer_is_ready = buffer.ready(device)
                monitor.add_text_batch(
                    valid_tokens=valid_tokens,
                    capture_time_s=capture_time_s,
                    buffer_ready=buffer_is_ready,
                )

                if not buffer_is_ready:
                    continue

                for x_batch_cpu in buffer.pop_cpu_batches():
                    current_step = step + 1
                    current_lr = lr_for_step(current_step)
                    current_l0_coeff = l0_coeff_for_step(current_step)

                    for group in optimizer.param_groups:
                        group["lr"] = current_lr

                    x_batch, h2d_time_s = timer.measure(
                        lambda batch=x_batch_cpu: batch.to(device, non_blocking=True)
                    )

                    def run_train_step():
                        optimizer.zero_grad(set_to_none=True)
                        x_hat, pre = sae_model(x_batch)
                        sae_base_local: TinyJumpReLUSAE = module_of(sae_model)

                        recon_loss_local = ((x_hat - x_batch) ** 2).mean()
                        l0_local = step_ste(
                            pre,
                            sae_base_local.get_threshold(),
                            STE_BANDWIDTH,
                        ).sum(dim=-1).mean()
                        loss_local = recon_loss_local + current_l0_coeff * l0_local

                        loss_local.backward()
                        sae_base_local.remove_decoder_grad_parallel()
                        optimizer.step()
                        sae_base_local.normalise_decoder()
                        return recon_loss_local, l0_local

                    (recon_loss, l0), train_step_time_s = timer.measure(run_train_step)
                    sae_base: TinyJumpReLUSAE = module_of(sae_model)

                    step = current_step
                    monitor.add_train_step(
                        recon_loss=float(recon_loss.detach().item()),
                        l0_value=float(l0.detach().item()),
                        h2d_time_s=h2d_time_s,
                        train_step_time_s=train_step_time_s,
                        sae_tokens=int(x_batch.shape[0]),
                    )

                    if step == 1 or step % LOG_EVERY == 0 or step == TRAIN_STEPS:
                        now = time.perf_counter()
                        wall_dt = max(now - last_log_time, 1e-8)
                        last_log_time = now

                        global_step_count = all_reduce_sum_int(monitor.step_count, device)
                        global_text_batch_count = all_reduce_sum_int(monitor.text_batch_count, device)
                        global_buffer_not_ready_count = all_reduce_sum_int(monitor.buffer_not_ready_count, device)
                        global_captured_valid_tokens = all_reduce_sum_int(monitor.captured_valid_tokens, device)
                        global_sae_tokens = all_reduce_sum_int(monitor.sae_tokens, device)
                        global_recon_sum = all_reduce_sum_float(monitor.recon_sum, device)
                        global_l0_sum = all_reduce_sum_float(monitor.l0_sum, device)
                        global_capture_time_s = all_reduce_sum_float(monitor.capture_time_s, device)
                        global_h2d_time_s = all_reduce_sum_float(monitor.h2d_time_s, device)
                        global_train_step_time_s = all_reduce_sum_float(monitor.train_step_time_s, device)

                        steps_in_window = max(global_step_count // max(world_size, 1), 1)
                        avg_recon = global_recon_sum / max(global_step_count, 1)
                        avg_l0 = global_l0_sum / max(global_step_count, 1)
                        active_frac = avg_l0 / max(sae_base.d_latent, 1)

                        avg_capture_ms = 1000.0 * global_capture_time_s / max(global_text_batch_count, 1)
                        avg_h2d_ms = 1000.0 * global_h2d_time_s / max(global_step_count, 1)
                        avg_train_step_ms = 1000.0 * global_train_step_time_s / max(global_step_count, 1)
                        avg_valid_tokens_per_text_batch = (
                            global_captured_valid_tokens / max(global_text_batch_count, 1)
                        )
                        avg_text_batches_per_step = global_text_batch_count / max(global_step_count, 1)
                        buffer_wait_frac = global_buffer_not_ready_count / max(global_text_batch_count, 1)
                        total_sae_tokens_per_sec = global_sae_tokens / wall_dt
                        steps_per_sec = steps_in_window / wall_dt

                        threshold_stats = get_threshold_stats(sae_base)
                        mem_stats = get_gpu_memory_stats(device)
                        buffer_min_tokens = all_reduce_min_int(buffer.size, device)

                        log0(
                            f"step={step:06d}/{TRAIN_STEPS} "
                            f"epoch={epoch} "
                            f"lr={current_lr:.2e} "
                            f"lambda={current_l0_coeff:.2e} "
                            f"recon_avg={avg_recon:.6f} "
                            f"avg_l0={avg_l0:.1f} "
                            f"active_frac={100.0 * active_frac:.2f}% "
                            f"thr_p05={threshold_stats['thr_p05']:.6f} "
                            f"thr_p50={threshold_stats['thr_p50']:.6f} "
                            f"thr_p95={threshold_stats['thr_p95']:.6f} "
                            f"thr_max={threshold_stats['thr_max']:.6f} "
                            f"capture_ms={avg_capture_ms:.1f} "
                            f"h2d_ms={avg_h2d_ms:.1f} "
                            f"train_step_ms={avg_train_step_ms:.1f} "
                            f"text_batches_per_step={avg_text_batches_per_step:.2f} "
                            f"valid_tokens_per_text_batch={avg_valid_tokens_per_text_batch:.1f} "
                            f"buffer_wait_frac={100.0 * buffer_wait_frac:.1f}% "
                            f"buffer_min_tokens={buffer_min_tokens} "
                            f"steps_per_sec={steps_per_sec:.2f} "
                            f"sae_tokens_per_sec={total_sae_tokens_per_sec:.1f} "
                            f"peak_alloc_gb={mem_stats['peak_alloc_gb']:.2f} "
                            f"peak_reserved_gb={mem_stats['peak_reserved_gb']:.2f}"
                        )

                        monitor.reset()
                        reset_gpu_peak_memory_stats(device)

                    if step % SAVE_EVERY == 0 or step == TRAIN_STEPS:
                        save_checkpoint(
                            CHECKPOINT_DIR / f"sae-v1_step_{step:06d}.pt",
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

        save_final_weights(
            FINAL_WEIGHTS_PATH,
            step,
            sae_model,
            activation_scale,
            token_stats,
        )

        dist.barrier()
        log0(f"TRAINING COMPLETE. Outputs saved under: {OUTPUT_DIR}")

    finally:
        if activation_model is not None:
            activation_model.close()
        cleanup()


if __name__ == "__main__":
    train()