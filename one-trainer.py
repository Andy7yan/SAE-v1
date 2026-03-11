import gzip
import json
import math
import os
import time
import urllib.request
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-2-2b-it"
DOLMA_URL = "https://olmo-data.org/dolma-v1_6/books/books-0000.json.gz"
DATA_CACHE_PATH = Path("./data_cache/books-0000.json.gz")

HOOK_LAYER_INDEX = 12
MAX_SEQ_LEN = 64
MIN_TEXT_CHARS = 20

TEXT_BATCH_SIZE_PER_RANK = 8
SAE_BATCH_SIZE = 4096
BUFFER_CAPACITY = 32768

TRAIN_STEPS = 200
LOG_EVERY = 10
SAVE_EVERY = 100

LATENT_FACTOR = 4
INIT_THRESHOLD = 1e-3
STE_BANDWIDTH = 1e-3
L0_COEFF = 1e-3
LR = 1e-3
MEAN_INIT_BATCHES = 16

SEED = 42
HTTP_TIMEOUT = 30
USER_AGENT = "Mozilla/5.0"

OUTPUT_DIR = Path("./outputs/jumprelu_sae_raw_dolma")


class StopForward(Exception):
    pass


def now():
    return time.strftime("%H:%M:%S")


def log0(msg: str):
    if dist.get_rank() == 0:
        print(f"[{now()}][sae_train] {msg}", flush=True)


def setup():
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Use torchrun.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return rank, local_rank, world_size, device, dtype


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_sum(x: torch.Tensor):
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


def all_reduce_mean(x: torch.Tensor):
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= dist.get_world_size()
    return y


def all_reduce_min_int(value: int, device: torch.device):
    t = torch.tensor(value, device=device, dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return int(t.item())


def extract_text(row: dict) -> str:
    value = row.get("text", "")
    if isinstance(value, str):
        return value.strip()
    return ""


def ensure_local_dolma_shard():
    if dist.get_rank() == 0 and not DATA_CACHE_PATH.exists():
        DATA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        log0(f"Downloading Dolma shard to {DATA_CACHE_PATH} ...")
        request = urllib.request.Request(DOLMA_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response, open(DATA_CACHE_PATH, "wb") as f:
            f.write(response.read())
    dist.barrier()
    if not DATA_CACHE_PATH.exists():
        raise RuntimeError(f"Missing shard: {DATA_CACHE_PATH}")
    return DATA_CACHE_PATH


def iter_rank_text_batches(shard_path: Path, local_batch_size: int, rank: int, world_size: int):
    assigned = []
    group = []
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


class FrozenActivationModel:
    def __init__(self, device: torch.device, model_dtype: torch.dtype):
        self.device = device
        self.token = os.environ.get("HF_TOKEN")

        log0("Loading tokeniser ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            token=self.token,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        log0(f"Loading {MODEL_NAME} (frozen) ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            token=self.token,
            dtype=model_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        self.activation_store = {}

        def hook_fn(module, inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Hook output is not a tensor: {type(output)}")
            self.activation_store["act"] = output.detach()
            raise StopForward()

        self.handle = self.model.model.layers[HOOK_LAYER_INDEX].register_forward_hook(hook_fn)
        log0(f"Residual-stream hook registered at layer {HOOK_LAYER_INDEX}")

    @torch.no_grad()
    def capture_text_batch(self, texts: list[str]):
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        self.activation_store.clear()

        try:
            self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
        except StopForward:
            pass

        if "act" not in self.activation_store:
            raise RuntimeError("Hook did not capture activation.")

        act = self.activation_store["act"].to(torch.float32)
        mask = batch["attention_mask"].bool()
        mask[:, 0] = False
        return act, mask

    def close(self):
        self.handle.remove()


def estimate_activation_mean(activation_model, shard_path, rank: int, world_size: int, device: torch.device):
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


def rectangle_window(x: torch.Tensor):
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)


class StepSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, threshold = ctx.saved_tensors
        rect = rectangle_window((x - threshold) / ctx.bandwidth)
        threshold_grad = -(1.0 / ctx.bandwidth) * rect * grad_output
        while threshold_grad.ndim > threshold.ndim:
            threshold_grad = threshold_grad.sum(dim=0)
        return torch.zeros_like(x), threshold_grad, None


class JumpReLUSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, threshold = ctx.saved_tensors
        x_grad = (x > threshold).to(x.dtype) * grad_output
        rect = rectangle_window((x - threshold) / ctx.bandwidth)
        threshold_grad = -(threshold / ctx.bandwidth) * rect * grad_output
        while threshold_grad.ndim > threshold.ndim:
            threshold_grad = threshold_grad.sum(dim=0)
        return x_grad, threshold_grad, None


def step_ste(x: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
    return StepSTE.apply(x, threshold, bandwidth)


def jumprelu_ste(x: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
    return JumpReLUSTE.apply(x, threshold, bandwidth)


class TinyJumpReLUSAE(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        d_latent = d_in * LATENT_FACTOR
        self.d_in = d_in
        self.d_latent = d_latent

        self.W_enc = nn.Parameter(torch.empty(d_latent, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_latent))
        self.W_dec = nn.Parameter(torch.empty(d_in, d_latent))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.log_threshold = nn.Parameter(
            torch.full((d_latent,), math.log(INIT_THRESHOLD), dtype=torch.float32)
        )

        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        self.normalise_decoder()

        with torch.no_grad():
            self.W_enc.copy_(self.W_dec.T)

    def threshold(self):
        return torch.exp(self.log_threshold)

    @torch.no_grad()
    def normalise_decoder(self):
        self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8))

    @torch.no_grad()
    def remove_decoder_grad_parallel(self):
        if self.W_dec.grad is None:
            return
        unit = self.W_dec / self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.grad.sub_((self.W_dec.grad * unit).sum(dim=0, keepdim=True) * unit)

    def forward(self, x: torch.Tensor):
        x_in = x - self.b_dec
        pre = F.relu(x_in @ self.W_enc.T + self.b_enc)
        z = jumprelu_ste(pre, self.threshold(), STE_BANDWIDTH)
        x_hat = z @ self.W_dec.T + self.b_dec
        return x_hat, pre


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


def module_of(model: nn.Module):
    if isinstance(model, DDP):
        return model.module
    return model


def save_checkpoint(path: Path, step: int, sae_model: nn.Module, optimizer: torch.optim.Optimizer):
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

    activation_model = FrozenActivationModel(device=device, model_dtype=model_dtype)

    try:
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
        activation_model.close()
        cleanup()


if __name__ == "__main__":
    train()