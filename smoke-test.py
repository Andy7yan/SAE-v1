from __future__ import annotations

import gzip
import json
import math
import os
import socket
import urllib.request
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Config
# ============================================================

MODEL_NAME = "google/gemma-2-2b-it"

# Public Dolma sample shard
DOLMA_SAMPLE_URL = "https://olmo-data.org/dolma-v1_6-8B-sample/v1_5r2_sample-0000.json.gz"

HOOK_LAYER_INDEX = 12
MAX_SEQ_LEN = 64

# Per-rank text batch size
# Effective global text batch size = TEXT_BATCH_SIZE_PER_RANK * WORLD_SIZE
TEXT_BATCH_SIZE_PER_RANK = 8

# Training length
TRAIN_STEPS = 200
LOG_EVERY = 10
SAVE_EVERY = 100

# JumpReLU SAE config
LATENT_FACTOR = 4
INIT_THRESHOLD = 1e-3
STE_BANDWIDTH = 1e-3
L0_COEFF = 1e-3
ACT_NORM_SCALE = 1.0

LR = 1e-3
MIN_TEXT_CHARS = 20
HTTP_TIMEOUT = 30
USER_AGENT = "Mozilla/5.0"
REQUIRE_CUDA = True
SEED = 42

# Paths
OUTPUT_DIR = Path("./outputs/jumprelu_sae")
DATA_CACHE_DIR = Path("./data_cache")
DATA_CACHE_PATH = DATA_CACHE_DIR / "v1_5r2_sample-0000.json.gz"


# ============================================================
# Distributed helpers
# ============================================================

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return 0

def get_local_rank_from_env() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def rank_prefix() -> str:
    return f"[rank={get_rank()} local_rank={get_local_rank_from_env()}]"


def print0(msg: str) -> None:
    if get_rank() == 0:
        print(msg, flush=True)


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    if is_distributed():
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


def setup_process(rank: int, local_rank: int, world_size: int, init_method: str | None) -> torch.device:
    if not torch.cuda.is_available():
        if REQUIRE_CUDA:
            raise RuntimeError(
                "CUDA is not available. Run this on a GPU compute node, not the login node."
            )
        return torch.device("cpu")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method=init_method if init_method is not None else "env://",
            rank=rank,
            world_size=world_size,
        )

    return device


def cleanup_process() -> None:
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


# ============================================================
# Data helpers
# ============================================================

def extract_text_from_example(example: dict) -> str:
    preferred_keys = ["text", "content", "document", "body"]

    for key in preferred_keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for value in example.values():
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise ValueError(f"Could not find a text field in example keys: {list(example.keys())}")


def get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN", None)


def download_url_to_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
        with open(dst, "wb") as f:
            f.write(response.read())


def ensure_local_dolma_shard(path: Path) -> Path:
    # Only rank 0 downloads; everyone else waits.
    if get_rank() == 0 and not path.exists():
        print0(f"[rank=0] Downloading Dolma shard to {path}")
        download_url_to_file(DOLMA_SAMPLE_URL, path)

    barrier()

    if not path.exists():
        raise RuntimeError(f"Dolma shard was expected at {path}, but it does not exist.")

    return path


def iter_rank_text_batches(
    shard_path: Path,
    local_batch_size: int,
    rank: int,
    world_size: int,
) -> Iterator[list[str]]:
    """
    Deterministic equal split across ranks.

    Rules:
    - Build complete groups of exactly `world_size` usable texts.
    - Within each complete group, rank k gets item k.
    - If the final group is incomplete, drop it.
    - Then build per-rank batches of size `local_batch_size`.
    - If the final batch is incomplete, drop it.

    Result:
    Every rank yields exactly the same number of steps.
    """
    assigned_texts: list[str] = []
    current_group: list[str] = []

    with gzip.open(shard_path, "rt", encoding="utf-8") as gz_file:
        for raw_line in gz_file:
            try:
                row = json.loads(raw_line)
                text = extract_text_from_example(row)
                if len(text) < MIN_TEXT_CHARS:
                    continue
            except Exception:
                continue

            current_group.append(text)

            if len(current_group) == world_size:
                assigned_texts.append(current_group[rank])
                current_group.clear()

                if len(assigned_texts) == local_batch_size:
                    yield assigned_texts
                    assigned_texts = []

    # Drop incomplete current_group and incomplete assigned_texts on purpose.


# ============================================================
# JumpReLU STE helpers
# ============================================================

def rectangle_window(x: torch.Tensor) -> torch.Tensor:
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
        bandwidth = ctx.bandwidth

        x_grad = torch.zeros_like(x)
        rect = rectangle_window((x - threshold) / bandwidth)
        threshold_grad = -(1.0 / bandwidth) * rect * grad_output

        while threshold_grad.ndim > threshold.ndim:
            threshold_grad = threshold_grad.sum(dim=0)

        return x_grad, threshold_grad, None


class JumpReLUSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        x_grad = (x > threshold).to(x.dtype) * grad_output

        rect = rectangle_window((x - threshold) / bandwidth)
        threshold_grad = -(threshold / bandwidth) * rect * grad_output

        while threshold_grad.ndim > threshold.ndim:
            threshold_grad = threshold_grad.sum(dim=0)

        return x_grad, threshold_grad, None


def step_ste(x: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
    return StepSTE.apply(x, threshold, bandwidth)


def jumprelu_ste(x: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
    return JumpReLUSTE.apply(x, threshold, bandwidth)


# ============================================================
# SAE
# ============================================================

class TinyJumpReLUSAE(nn.Module):
    def __init__(
        self,
        d_in: int,
        latent_factor: int = 4,
        input_scale: float = 1.0,
        init_threshold: float = 1e-3,
        ste_bandwidth: float = 1e-3,
    ):
        super().__init__()
        d_latent = d_in * latent_factor

        self.d_in = d_in
        self.d_latent = d_latent
        self.ste_bandwidth = ste_bandwidth

        self.register_buffer(
            "input_scale",
            torch.tensor(float(input_scale), dtype=torch.float32),
        )

        self.W_enc = nn.Parameter(torch.empty(d_latent, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_latent))

        self.W_dec = nn.Parameter(torch.empty(d_in, d_latent))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        init_log_threshold = math.log(init_threshold)
        self.log_threshold = nn.Parameter(
            torch.full((d_latent,), init_log_threshold, dtype=torch.float32)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        self._normalise_decoder_()

        with torch.no_grad():
            self.W_enc.copy_(self.W_dec.T)

    @torch.no_grad()
    def _normalise_decoder_(self) -> None:
        norms = self.W_dec.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    @torch.no_grad()
    def remove_decoder_grad_parallel_(self) -> None:
        if self.W_dec.grad is None:
            return

        col_norms = self.W_dec.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        unit_cols = self.W_dec.data / col_norms
        parallel = (self.W_dec.grad * unit_cols).sum(dim=0, keepdim=True) * unit_cols
        self.W_dec.grad.sub_(parallel)

    def threshold(self) -> torch.Tensor:
        return torch.exp(self.log_threshold)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_scaled = x / self.input_scale
        x_in = x_scaled - self.b_dec

        pre_acts = F.relu(x_in @ self.W_enc.T + self.b_enc)
        theta = self.threshold()
        z = jumprelu_ste(pre_acts, theta, self.ste_bandwidth)
        x_hat = z @ self.W_dec.T + self.b_dec

        return x_hat, z, pre_acts, x_scaled


# ============================================================
# Logging / checkpoint helpers
# ============================================================

def module_of(model: nn.Module) -> nn.Module:
    if isinstance(model, DDP):
        return model.module
    return model


def compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(torch.sum(p.grad.detach() ** 2).item())
    return total ** 0.5


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def save_checkpoint(
    output_dir: Path,
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
        },
    }
    torch.save(payload, ckpt_path)
    print0(f"[rank=0] Saved checkpoint: {ckpt_path}")


# ============================================================
# Training
# ============================================================

def run_training(rank: int, local_rank: int, world_size: int, init_method: str | None) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = setup_process(rank=rank, local_rank=local_rank, world_size=world_size, init_method=init_method)
    set_seed(SEED)

    prefix = f"[rank={rank} local_rank={local_rank}]"
    model_dtype = get_model_dtype(device)
    hf_token = get_hf_token()

    print(f"{prefix} device={device} world_size={world_size} model_dtype={model_dtype}", flush=True)
    print(f"{prefix} HF_HOME={os.environ.get('HF_HOME', '(not set)')}", flush=True)
    print(f"{prefix} HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE', '(not set)')}", flush=True)
    print(f"{prefix} HF_TOKEN_present={hf_token is not None}", flush=True)

    shard_path = ensure_local_dolma_shard(DATA_CACHE_PATH)

    if get_rank() == 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        save_json(
            OUTPUT_DIR / "train_config.json",
            {
                "model_name": MODEL_NAME,
                "dolma_sample_url": DOLMA_SAMPLE_URL,
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
            },
        )

    barrier()

    # ----------------------------
    # Load tokenizer
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----------------------------
    # Load frozen base model
    # One full frozen model replica per rank/GPU.
    # ----------------------------
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=hf_token,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ----------------------------
    # Hook activation
    # ----------------------------
    activation_store: dict[str, torch.Tensor] = {}

    def hook_fn(module, inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]

        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Hook output is not a tensor: {type(output)}")

        activation_store["act"] = output.detach()

    target_module = model.model.layers[HOOK_LAYER_INDEX]
    handle = target_module.register_forward_hook(hook_fn)

    print(f"{prefix} hook_module=model.model.layers[{HOOK_LAYER_INDEX}]", flush=True)

    sae_model: nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None

    step = 0
    epoch = 0

    try:
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

                batch = tokenizer(
                    texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                    padding="max_length",
                )
                batch = {k: v.to(device) for k, v in batch.items()}

                activation_store.clear()

                with torch.no_grad():
                    _ = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    )

                if "act" not in activation_store:
                    raise RuntimeError("Hook fired zero times; no activation was captured.")

                act = activation_store["act"]  # [B, S, D]
                if act.ndim != 3:
                    raise RuntimeError(f"Expected hooked activation to be 3D, got {tuple(act.shape)}")

                d_in = int(act.shape[-1])

                if sae_model is None:
                    base_sae = TinyJumpReLUSAE(
                        d_in=d_in,
                        latent_factor=LATENT_FACTOR,
                        input_scale=ACT_NORM_SCALE,
                        init_threshold=INIT_THRESHOLD,
                        ste_bandwidth=STE_BANDWIDTH,
                    ).to(device=device, dtype=torch.float32)

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
                        f"[rank=0] SAE initialised: d_in={d_in}, d_latent={module_of(sae_model).d_latent}, "
                        f"global_text_batch={TEXT_BATCH_SIZE_PER_RANK * world_size}"
                    )

                assert sae_model is not None
                assert optimizer is not None

                x = act.to(dtype=torch.float32).reshape(-1, d_in)               # [B*S, D]
                mask_flat = batch["attention_mask"].reshape(-1).to(
                    device=device, dtype=torch.float32
                )                                                               # [B*S]

                x_hat, z, pre_acts, x_scaled = sae_model(x)

                # --------------------------------------------------
                # Exact global-token normalisation for DDP
                # --------------------------------------------------
                recon_per_token = ((x_hat - x_scaled) ** 2).mean(dim=-1)         # [B*S]
                recon_sum_local = (recon_per_token * mask_flat).sum()

                theta = module_of(sae_model).threshold()
                l0_proxy = step_ste(pre_acts, theta, module_of(sae_model).ste_bandwidth).sum(dim=-1)
                l0_sum_local = (l0_proxy * mask_flat).sum()

                valid_tokens_local = mask_flat.sum().clamp(min=1.0)
                valid_tokens_global = all_reduce_sum(valid_tokens_local.detach())

                # Because DDP averages gradients across ranks, multiply by world_size
                # so the resulting gradient matches the true global-token objective.
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

                # --------------------------------------------------
                # Logging values computed from global sums
                # --------------------------------------------------
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
        handle.remove()
        cleanup_process()


# ============================================================
# Launch logic
# ============================================================

def main() -> None:
    if not torch.cuda.is_available():
        if REQUIRE_CUDA:
            raise RuntimeError(
                "CUDA is not available. Run this on a GPU compute node, not the login node."
            )
        run_training(rank=0, local_rank=0, world_size=1, init_method=None)
        return

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    run_training(rank=rank, local_rank=local_rank, world_size=world_size, init_method="env://")
    return


if __name__ == "__main__":
    main()