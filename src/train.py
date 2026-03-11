import json

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from activation_store import FrozenActivationModel
from config import (
    ACT_NORM_SCALE,
    DATA_CACHE_PATH,
    HOOK_LAYER_INDEX,
    INIT_THRESHOLD,
    L0_COEFF,
    LATENT_FACTOR,
    LOG_EVERY,
    LR,
    MAX_SEQ_LEN,
    MEAN_INIT_BATCHES,
    MODEL_NAME,
    OUTPUT_DIR,
    SAVE_EVERY,
    SEED,
    STE_BANDWIDTH,
    TEXT_BATCH_SIZE_PER_RANK,
    TRAIN_STEPS,
    REQUIRE_CUDA,
)
from data import ensure_local_dolma_shard, iter_rank_text_batches
from dist_utils import (
    all_reduce_sum,
    barrier,
    cleanup_process,
    find_free_port,
    get_model_dtype,
    get_rank,
    launched_with_torchrun,
    print0,
    set_seed,
    setup_process,
)
from init_stats import estimate_activation_mean
from sae import TinyJumpReLUSAE, compute_grad_norm, module_of, step_ste


def save_json(path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def append_jsonl(path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def save_checkpoint(
    output_dir,
    step: int,
    sae_model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    if get_rank() != 0:
        return

    sae = module_of(sae_model)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / f"sae_step_{step:06d}.pt"
    payload = {
        "step": step,
        "sae_state_dict": sae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": {
            "model_name": MODEL_NAME,
            "hook_layer_index": HOOK_LAYER_INDEX,
            "max_seq_len": MAX_SEQ_LEN,
            "latent_factor": LATENT_FACTOR,
            "init_threshold": INIT_THRESHOLD,
            "ste_bandwidth": STE_BANDWIDTH,
            "l0_coeff": L0_COEFF,
            "act_norm_scale": ACT_NORM_SCALE,
            "lr": LR,
            "text_batch_size_per_rank": TEXT_BATCH_SIZE_PER_RANK,
            "train_steps": TRAIN_STEPS,
            "mean_init_batches": MEAN_INIT_BATCHES,
        },
    }
    torch.save(payload, ckpt_path)
    print0(f"[rank=0] Saved checkpoint: {ckpt_path}")


def run_training(rank: int, local_rank: int, world_size: int, init_method: str | None) -> None:
    os = __import__("os")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = setup_process(rank=rank, local_rank=local_rank, world_size=world_size, init_method=init_method)
    set_seed(SEED)

    prefix = f"[rank={rank} local_rank={local_rank}]"
    model_dtype = get_model_dtype(device)

    print(f"{prefix} device={device} world_size={world_size} model_dtype={model_dtype}", flush=True)
    print(f"{prefix} HF_HOME={os.environ.get('HF_HOME', '(not set)')}", flush=True)
    print(f"{prefix} HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE', '(not set)')}", flush=True)
    print(f"{prefix} HF_TOKEN_present={'HF_TOKEN' in os.environ}", flush=True)

    shard_path = ensure_local_dolma_shard(DATA_CACHE_PATH)

    if get_rank() == 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_json(
            OUTPUT_DIR / "train_config.json",
            {
                "model_name": MODEL_NAME,
                "hook_layer_index": HOOK_LAYER_INDEX,
                "max_seq_len": MAX_SEQ_LEN,
                "text_batch_size_per_rank": TEXT_BATCH_SIZE_PER_RANK,
                "train_steps": TRAIN_STEPS,
                "latent_factor": LATENT_FACTOR,
                "init_threshold": INIT_THRESHOLD,
                "ste_bandwidth": STE_BANDWIDTH,
                "l0_coeff": L0_COEFF,
                "act_norm_scale": ACT_NORM_SCALE,
                "lr": LR,
                "mean_init_batches": MEAN_INIT_BATCHES,
            },
        )

    barrier()

    activation_model = FrozenActivationModel(device=device, model_dtype=model_dtype)
    print(f"{prefix} hook_module=model.model.layers[{HOOK_LAYER_INDEX}]", flush=True)

    sae_model: nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None

    step = 0
    epoch = 0

    try:
        # --------------------------------------------------
        # 1) Estimate activation mean and initialise b_dec
        # --------------------------------------------------
        mean_vec = estimate_activation_mean(
            activation_model=activation_model,
            shard_path=shard_path,
            local_batch_size=TEXT_BATCH_SIZE_PER_RANK,
            num_batches=MEAN_INIT_BATCHES,
            rank=rank,
            world_size=world_size,
            device=device,
        )

        d_in = int(mean_vec.numel())

        base_sae = TinyJumpReLUSAE(
            d_in=d_in,
            latent_factor=LATENT_FACTOR,
            input_scale=ACT_NORM_SCALE,
            init_threshold=INIT_THRESHOLD,
            ste_bandwidth=STE_BANDWIDTH,
        ).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            base_sae.b_dec.copy_(mean_vec.to(device=device, dtype=torch.float32))

        if world_size > 1:
            sae_model = DDP(
                base_sae,
                device_ids=[local_rank],
                output_device=local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            sae_model = base_sae

        optimizer = torch.optim.Adam(sae_model.parameters(), lr=LR, betas=(0.0, 0.999))

        print0(
            f"[rank=0] SAE initialised: "
            f"d_in={d_in}, "
            f"d_latent={module_of(sae_model).d_latent}, "
            f"global_text_batch={TEXT_BATCH_SIZE_PER_RANK * world_size}"
        )

        # --------------------------------------------------
        # 2) Training loop
        # --------------------------------------------------
        while step < TRAIN_STEPS:
            epoch += 1
            local_batches_this_epoch = 0

            for texts in iter_rank_text_batches(
                shard_path=shard_path,
                local_batch_size=TEXT_BATCH_SIZE_PER_RANK,
                rank=rank,
                world_size=world_size,
            ):
                local_batches_this_epoch += 1
                step += 1

                act, attention_mask = activation_model.capture_text_batch(texts)

                if act.ndim != 3:
                    raise RuntimeError(f"Expected hooked activation to be 3D, got {tuple(act.shape)}")

                x = act.to(dtype=torch.float32).reshape(-1, d_in)
                mask_flat = attention_mask.reshape(-1).to(device=device, dtype=torch.float32)

                x_hat, z, pre_acts, x_scaled = sae_model(x)

                recon_per_token = ((x_hat - x_scaled) ** 2).mean(dim=-1)
                recon_sum_local = (recon_per_token * mask_flat).sum()

                theta = module_of(sae_model).threshold()
                l0_proxy = step_ste(pre_acts, theta, module_of(sae_model).ste_bandwidth).sum(dim=-1)
                l0_sum_local = (l0_proxy * mask_flat).sum()

                valid_tokens_local = mask_flat.sum().clamp(min=1.0)
                valid_tokens_global = all_reduce_sum(valid_tokens_local.detach())

                scale = float(world_size) / valid_tokens_global

                recon_loss = recon_sum_local * scale
                l0_loss = L0_COEFF * l0_sum_local * scale
                loss = recon_loss + l0_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                sae_base = module_of(sae_model)
                sae_base.remove_decoder_grad_parallel_()

                grad_norm = compute_grad_norm(sae_base)

                optimizer.step()
                sae_base._normalise_decoder_()

                recon_sum_global = all_reduce_sum(recon_sum_local.detach())
                l0_sum_global = all_reduce_sum(l0_sum_local.detach())

                logged_recon = float((recon_sum_global / valid_tokens_global).item())
                logged_l0 = float((L0_COEFF * l0_sum_global / valid_tokens_global).item())
                logged_total = logged_recon + logged_l0
                avg_l0 = float((l0_sum_global / valid_tokens_global).item())
                theta_mean = float(sae_base.threshold().mean().item())

                if step == 1 or step % LOG_EVERY == 0 or step == TRAIN_STEPS:
                    print0(
                        f"[rank=0] step={step:05d}/{TRAIN_STEPS} "
                        f"epoch={epoch} "
                        f"valid_tokens_global={int(valid_tokens_global.item())} "
                        f"recon_loss={logged_recon:.6f} "
                        f"l0_loss={logged_l0:.6f} "
                        f"total_loss={logged_total:.6f} "
                        f"avg_l0={avg_l0:.3f} "
                        f"theta_mean={theta_mean:.6f} "
                        f"grad_norm={grad_norm:.6f}"
                    )

                    append_jsonl(
                        OUTPUT_DIR / "train_log.jsonl",
                        {
                            "step": step,
                            "epoch": epoch,
                            "valid_tokens_global": int(valid_tokens_global.item()),
                            "recon_loss": logged_recon,
                            "l0_loss": logged_l0,
                            "total_loss": logged_total,
                            "avg_l0": avg_l0,
                            "theta_mean": theta_mean,
                            "grad_norm": grad_norm,
                        },
                    )

                if step % SAVE_EVERY == 0 or step == TRAIN_STEPS:
                    save_checkpoint(
                        output_dir=OUTPUT_DIR,
                        step=step,
                        sae_model=sae_model,
                        optimizer=optimizer,
                    )

                if step >= TRAIN_STEPS:
                    break

            if local_batches_this_epoch == 0:
                raise RuntimeError(
                    "No complete per-rank batches were produced from the Dolma shard. "
                    "Reduce TEXT_BATCH_SIZE_PER_RANK, reduce WORLD_SIZE, or use a larger shard."
                )

        barrier()
        print0(f"[rank=0] TRAINING COMPLETE. Outputs saved under: {OUTPUT_DIR}")

    finally:
        activation_model.close()
        cleanup_process()


def spawn_entry(local_rank: int, world_size: int, port: int) -> None:
    run_training(
        rank=local_rank,
        local_rank=local_rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )


def main() -> None:
    if not torch.cuda.is_available():
        if REQUIRE_CUDA:
            raise RuntimeError(
                "CUDA is not available. Run this on a GPU compute node, not the login node."
            )
        run_training(rank=0, local_rank=0, world_size=1, init_method=None)
        return

    if launched_with_torchrun():
        os = __import__("os")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        run_training(rank=rank, local_rank=local_rank, world_size=world_size, init_method="env://")
        return

    visible_gpu_count = torch.cuda.device_count()

    if visible_gpu_count <= 1:
        run_training(rank=0, local_rank=0, world_size=1, init_method=None)
        return

    port = find_free_port()
    print(f"[main] Detected {visible_gpu_count} visible GPUs. Auto-spawning DDP on port {port}.", flush=True)

    mp.spawn(
        spawn_entry,
        args=(visible_gpu_count, port),
        nprocs=visible_gpu_count,
        join=True,
    )


if __name__ == "__main__":
    main()