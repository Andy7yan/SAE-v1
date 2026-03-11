import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from activation_store import FrozenActivationModel
from config import (
    BUFFER_CAPACITY,
    HOOK_LAYER_INDEX,
    LATENT_FACTOR,
    LOG_EVERY,
    LR,
    L0_COEFF,
    MAX_SEQ_LEN,
    MODEL_NAME,
    OUTPUT_DIR,
    SAE_BATCH_SIZE,
    SAVE_EVERY,
    STE_BANDWIDTH,
    TEXT_BATCH_SIZE_PER_RANK,
    TRAIN_STEPS,
)
from data import ensure_local_dolma_shard, iter_rank_text_batches
from dist_utils import all_reduce_mean, all_reduce_min_int, cleanup, log0, setup
from init_stats import estimate_activation_mean
from sae import TinyJumpReLUSAE, module_of, step_ste


class ActivationBuffer:
    def __init__(self):
        self.chunks = []
        self.size = 0

    def add(self, x: torch.Tensor, mask: torch.Tensor):
        valid = x[mask]
        if valid.numel() == 0:
            return
        self.chunks.append(valid.detach().cpu())
        self.size += valid.shape[0]

    def ready(self, device: torch.device):
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
            yield take[i:i + SAE_BATCH_SIZE].to(device, non_blocking=True)


def save_checkpoint(path, step: int, sae_model: nn.Module, optimizer: torch.optim.Optimizer):
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
            "latent_factor": LATENT_FACTOR,
        },
        path,
    )


def train():
    assert BUFFER_CAPACITY % SAE_BATCH_SIZE == 0

    rank, local_rank, world_size, device, model_dtype = setup()
    log0(f"Rank {rank}/{world_size} | device: {device}")

    shard_path = ensure_local_dolma_shard()

    if rank == 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    activation_model = None

    try:
        activation_model = FrozenActivationModel(device=device, model_dtype=model_dtype)

        mean_vec = estimate_activation_mean(
            activation_model=activation_model,
            shard_path=shard_path,
            rank=rank,
            world_size=world_size,
            device=device,
        )

        d_in = int(mean_vec.numel())

        base_sae = TinyJumpReLUSAE(d_in=d_in).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            base_sae.b_dec.copy_(mean_vec.to(device=device, dtype=torch.float32))

        sae_model = DDP(
            base_sae,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

        optimizer = torch.optim.Adam(sae_model.parameters(), lr=LR, betas=(0.0, 0.999))
        buffer = ActivationBuffer()
        step = 0
        epoch = 0

        log0(
            f"Training | jumprelu | total_steps={TRAIN_STEPS} | text_batch={TEXT_BATCH_SIZE_PER_RANK} | sae_batch={SAE_BATCH_SIZE} | buffer={BUFFER_CAPACITY}"
        )

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

                act, mask = activation_model.capture_text_batch(texts)
                x = act.reshape(-1, d_in)
                mask_flat = mask.reshape(-1)
                buffer.add(x, mask_flat)

                if not buffer.ready(device):
                    continue

                for x_batch in buffer.pop_batches(device):
                    step += 1

                    x_hat, pre = sae_model(x_batch)
                    sae_base = module_of(sae_model)

                    recon_loss = ((x_hat - x_batch) ** 2).mean()
                    l0 = step_ste(pre, sae_base.threshold(), STE_BANDWIDTH).sum(dim=-1).mean()
                    loss = recon_loss + L0_COEFF * l0

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    sae_base.remove_decoder_grad_parallel()
                    optimizer.step()
                    sae_base.normalise_decoder()

                    if step == 1 or step % LOG_EVERY == 0 or step == TRAIN_STEPS:
                        recon_mean = float(all_reduce_mean(recon_loss).item())
                        l0_mean = float(all_reduce_mean(l0).item())
                        theta_mean = float(sae_base.threshold().mean().item())
                        log0(
                            f"step={step:05d}/{TRAIN_STEPS} epoch={epoch} recon={recon_mean:.6f} avg_l0={l0_mean:.3f} theta_mean={theta_mean:.6f}"
                        )

                    if step % SAVE_EVERY == 0 or step == TRAIN_STEPS:
                        save_checkpoint(
                            OUTPUT_DIR / f"sae_step_{step:06d}.pt",
                            step,
                            sae_model,
                            optimizer,
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