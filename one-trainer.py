#!/usr/bin/env python3
"""
Sparse Autoencoder (SAE) — Gemma 2-2b-it · Layer 12 · Dolma (AllenAI)
=======================================================================
Top-K SAE with Ghost Gradients for dead-neuron revival.

Activation:  Top-K (exactly K features active per token)
Dead-neuron: Ghost Gradients — dead neurons receive gradient signal
             from the live reconstruction residual, pushing them to
             capture unexplained variance (Anthropic, 2024).

Loss = MSE(x_hat, x)  +  ghost_coeff · MSE(x_ghost_scaled, residual·0.5)

With Top-K, sparsity is structurally enforced (no L1 needed).

Usage (single node, N GPUs):
    torchrun --standalone --nproc_per_node=<N_GPUS> train_sae.py [--args]

Usage (SLURM / multi-node):
    torchrun --nnodes=$SLURM_NNODES \
             --nproc_per_node=$SLURM_GPUS_ON_NODE \
             --rdzv_id=$SLURM_JOB_ID \
             --rdzv_backend=c10d \
             --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
             train_sae.py [--args]
"""

# ===========================================================================
# Imports  (must come before the macros that reference torch)
# ===========================================================================
import os
import sys
import math
import time
import logging
import argparse
from pathlib import Path
from contextlib import nullcontext
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


# ===========================================================================
# MACROS — edit before running
# ===========================================================================
HF_TOKEN: str = "YOUR_HF_TOKEN_HERE"       # HuggingFace token (gated model)

# Model
MODEL_NAME: str  = "google/gemma-2-2b-it"
HOOK_LAYER: int  = 12                       # 0-indexed transformer layer
HIDDEN_DIM: int  = 2304                     # Gemma 2-2b residual-stream dim

# SAE architecture
EXPANSION_FACTOR: int = 8                   # SAE width = HIDDEN_DIM × factor
SAE_DIM: int          = HIDDEN_DIM * EXPANSION_FACTOR   # 18 432
TOP_K: int            = 64                  # active features per token (L0)

# Ghost-gradient hyper-parameters
GHOST_THRESHOLD: int    = 10_000  # steps without firing → neuron is "dead"
GHOST_GRAD_COEFF: float = 1.0     # weight of ghost loss term

# Training hyper-parameters
LEARNING_RATE: float = 1e-4
BATCH_SIZE: int      = 4096       # activation vectors per global step
MAX_SEQ_LEN: int     = 128        # tokens per document chunk
GRAD_CLIP: float     = 1.0
TOTAL_STEPS: int     = 200_000
WARMUP_STEPS: int    = 1_000
SAVE_EVERY: int      = 5_000
LOG_EVERY: int       = 100

# Dataset
DATASET_NAME: str   = "allenai/dolma"
DATASET_SUBSET: str = "v1_6"               # Dolma config/subset name
DATASET_SPLIT: str  = "train"
TEXT_COLUMN: str    = "text"
NUM_WORKERS: int    = 4                    # DataLoader CPU workers per rank

# Paths
CHECKPOINT_DIR: str = "./sae_checkpoints"
LOG_DIR: str        = "./sae_logs"

# Mixed precision  (bfloat16 — native to Gemma, no GradScaler needed)
USE_AMP: bool = True


# ===========================================================================
# Distributed helpers
# ===========================================================================

def setup_distributed() -> tuple[int, int, int]:
    """Initialise torch.distributed when launched via torchrun / SLURM."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
    else:
        local_rank = rank = 0
        world_size = 1
    return local_rank, rank, world_size


def is_main(rank: int) -> bool:
    return rank == 0


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_min_(t: torch.Tensor) -> None:
    """In-place MIN all-reduce across all DDP ranks (no-op if single process)."""
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.MIN)


# ===========================================================================
# Logging
# ===========================================================================

def get_logger(rank: int, log_dir: str) -> logging.Logger:
    logger = logging.getLogger("sae_train")
    logger.setLevel(logging.DEBUG if rank == 0 else logging.WARNING)
    fmt = logging.Formatter("[%(asctime)s][%(name)s] %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if rank == 0:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / "train.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ===========================================================================
# Top-K activation
# ===========================================================================

def topk_activation(pre_acts: torch.Tensor, k: int) -> torch.Tensor:
    """
    Enforce exactly k active (positive) features per token.

    Algorithm
    ---------
    1. Find the k largest pre-activation values for each token.
    2. Zero out all other positions.
    3. Apply ReLU to the kept values so negative top-k entries are also
       suppressed (can occur near initialisation).

    Shape: [N, latent_dim] → [N, latent_dim]  (k non-zero entries per row)
    """
    topk_vals, topk_idx = torch.topk(pre_acts, k, dim=-1)  # [N, k]
    z = torch.zeros_like(pre_acts)
    z.scatter_(-1, topk_idx, F.relu(topk_vals))
    return z


# ===========================================================================
# Sparse Autoencoder  (Top-K + Ghost Gradients)
# ===========================================================================

class SparseAutoencoder(nn.Module):
    """
    Top-K Sparse Autoencoder with Ghost Gradient support for dead neurons.

    Live forward path
    -----------------
        pre_acts = W_enc · (x − b_pre) + b_enc        encoder pre-activations
        z        = TopK(pre_acts)                      exactly k active features
        x_hat    = W_dec · z  +  b_dec  +  b_pre      reconstruction

    Ghost gradient path  (only for dead neurons, only during training)
    ------------------
    A neuron is dead if it has not fired on any DDP rank for ≥ ghost_threshold
    consecutive training steps.  Dead neurons cannot receive gradients through
    the Top-K operator (they are never selected), so we bypass it:

        ghost_pre  = exp(pre_acts) * dead_mask          soft, always > 0
        ghost_norm = ghost_pre / Σ_dead(ghost_pre)      normalise across dead
        x_ghost    = W_dec_unit · ghost_norm            unit-norm decoder
        scale      = ‖residual‖ / ‖x_ghost‖  *  0.5
        x_ghost_sc = x_ghost * scale

    Ghost loss = MSE( x_ghost_sc ,  stop_grad(residual) * 0.5 )

    Comparing x_ghost_sc (which depends on pre_acts for dead neurons) against
    the detached half-residual creates a gradient that steers W_enc/b_enc of
    dead neurons toward capturing unexplained variance without destabilising
    the live reconstruction.

    Total loss = MSE(x_hat, x)  +  ghost_coeff * ghost_loss

    Dead-neuron state
    -----------------
    steps_since_fired  — registered buffer, shape [latent_dim], int64.
    Persisted in checkpoints so resumption does not reset the counters.
    After each step the buffer is MIN-reduced across all DDP ranks: a neuron
    is only dead if it has not fired on *any* rank.
    """

    def __init__(self, input_dim: int, latent_dim: int, top_k: int):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.top_k      = top_k

        # Learned centring of the activation space
        self.b_pre = nn.Parameter(torch.zeros(input_dim))

        # Encoder weights & bias
        self.W_enc = nn.Parameter(torch.empty(latent_dim, input_dim))
        self.b_enc = nn.Parameter(torch.zeros(latent_dim))

        # Decoder weights & bias
        self.W_dec = nn.Parameter(torch.empty(input_dim, latent_dim))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        # Dead-neuron counter — updated and synced every training step
        self.register_buffer(
            "steps_since_fired",
            torch.zeros(latent_dim, dtype=torch.long),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        self._normalise_decoder_()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _normalise_decoder_(self) -> None:
        """Project each decoder column onto the unit sphere (in-place)."""
        norms = self.W_dec.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_dead_neurons(self, z: torch.Tensor) -> None:
        """
        Update steps_since_fired after each training step.

        z : [N, latent_dim] — sparse activations from the current batch.

        Steps
        -----
        1. Increment every counter.
        2. Reset counters for neurons that fired this step.
        3. MIN all-reduce across DDP ranks so a neuron is considered alive
           if it fired on at least one rank.
        """
        fired = (z > 0).any(dim=0)          # [latent_dim] bool
        self.steps_since_fired.add_(1)
        self.steps_since_fired[fired] = 0
        all_reduce_min_(self.steps_since_fired)

    # ------------------------------------------------------------------
    def dead_mask(self, threshold: int) -> torch.Tensor:
        """Return a boolean mask: True for neurons dead >= threshold steps."""
        return self.steps_since_fired >= threshold  # [latent_dim]

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        ghost_threshold: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x               : [N, input_dim]  batch of activation vectors
        ghost_threshold : when provided, run the ghost path for neurons that
                          have been dead for >= ghost_threshold steps.
                          Pass None to disable ghost gradients (e.g. eval).

        Returns (dict)
        --------------
        pre_acts    [N, latent_dim]  raw encoder output before Top-K
        z           [N, latent_dim]  sparse Top-K activations
        x_hat       [N, input_dim]   live reconstruction
        x_ghost     [N, input_dim]   scaled ghost reconstruction  (optional)
        ghost_tgt   [N, input_dim]   detached residual * 0.5      (optional)
        n_dead      scalar int       number of dead neurons
        """
        # ── Live path ──────────────────────────────────────────────────
        x_c      = x - self.b_pre
        pre_acts = x_c @ self.W_enc.T + self.b_enc     # [N, latent_dim]
        z        = topk_activation(pre_acts, self.top_k)
        x_hat    = z @ self.W_dec.T + self.b_dec + self.b_pre

        out: dict[str, torch.Tensor] = {
            "pre_acts": pre_acts,
            "z":        z,
            "x_hat":    x_hat,
            "n_dead":   torch.tensor(0, device=x.device),
        }

        # ── Ghost path ─────────────────────────────────────────────────
        if ghost_threshold is not None:
            dmask  = self.dead_mask(ghost_threshold)    # [latent_dim] bool
            n_dead = int(dmask.sum().item())
            out["n_dead"] = torch.tensor(n_dead, device=x.device)

            if n_dead > 0:
                # Detach residual: ghost loss must not pull the live
                # reconstruction off course.
                residual = (x - x_hat).detach()        # [N, input_dim]

                # --- Ghost activations ---
                # Use exp (not ReLU / Top-K) so every dead neuron receives
                # a non-zero, differentiable gradient signal regardless of
                # the sign of its pre-activation.
                ghost_pre = torch.exp(pre_acts) * dmask.float()  # [N, latent_dim]

                # Normalise across the dead-neuron dimension per token so
                # magnitudes remain comparable to unit-norm decoder columns.
                ghost_sum = ghost_pre.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                ghost_pre_n = ghost_pre / ghost_sum               # [N, latent_dim]

                # Ghost reconstruction — W_dec columns are unit-norm
                x_ghost = ghost_pre_n @ self.W_dec.T              # [N, input_dim]

                # Scale x_ghost so ‖x_ghost‖ ≈ 0.5 · ‖residual‖ per token.
                # This keeps the ghost loss on the same scale as the MSE.
                res_norm    = residual.norm(dim=-1, keepdim=True)           # [N, 1]
                ghost_norm_ = x_ghost.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                x_ghost_sc  = x_ghost * (res_norm / ghost_norm_) * 0.5    # [N, input_dim]

                out["x_ghost"]   = x_ghost_sc
                out["ghost_tgt"] = residual * 0.5                 # already detached

        return out

    # ------------------------------------------------------------------
    @staticmethod
    def compute_loss(
        x:           torch.Tensor,
        out:         dict[str, torch.Tensor],
        ghost_coeff: float,
    ) -> dict[str, torch.Tensor]:
        """
        Compute training loss from a forward-pass output dict.

        Components
        ----------
        mse   : reconstruction loss (always)
        ghost : ghost-gradient loss  (0 when there are no dead neurons)
        total : mse + ghost_coeff * ghost
        """
        mse = F.mse_loss(out["x_hat"], x, reduction="mean")

        ghost = torch.zeros(1, device=x.device, dtype=mse.dtype).squeeze()
        if "x_ghost" in out:
            ghost = F.mse_loss(out["x_ghost"], out["ghost_tgt"], reduction="mean")

        total = mse + ghost_coeff * ghost
        return {"total": total, "mse": mse, "ghost": ghost}


# ===========================================================================
# Activation hook — residual-stream output at a single transformer layer
# ===========================================================================

class ResidualStreamHook:
    """
    Captures hidden states at the output of transformer layer `layer_idx`.

    For Gemma 2, model.model.layers[i]'s forward returns a tuple whose first
    element is the updated hidden-state tensor of shape [B, T, D].
    """

    def __init__(self):
        self._handle                             = None
        self.activations: Optional[torch.Tensor] = None

    def _hook_fn(self, module, inputs, output) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self.activations = hidden.detach()   # no backprop through frozen LLM

    def register(self, model: nn.Module, layer_idx: int) -> None:
        layer        = model.model.layers[layer_idx]
        self._handle = layer.register_forward_hook(self._hook_fn)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ===========================================================================
# Streaming dataset — Dolma (AllenAI)
# ===========================================================================

class DolmaStreamDataset(IterableDataset):
    """
    Streams tokenised text from Dolma and yields fixed-length token chunks.

    Two-level sharding guarantees no overlap across workers/ranks:
      1. HuggingFace `.shard()` — one shard per DDP rank.
      2. DataLoader worker round-robin — further divides each rank's shard.
    """

    def __init__(
        self,
        tokenizer,
        max_seq_len:  int,
        rank:         int,
        world_size:   int,
        dataset_name: str = DATASET_NAME,
        subset:       str = DATASET_SUBSET,
        split:        str = DATASET_SPLIT,
        text_column:  str = TEXT_COLUMN,
        hf_token:     str = HF_TOKEN,
    ):
        self.tokenizer    = tokenizer
        self.max_seq_len  = max_seq_len
        self.rank         = rank
        self.world_size   = world_size
        self.dataset_name = dataset_name
        self.subset       = subset
        self.split        = split
        self.text_column  = text_column
        self.hf_token     = hf_token

    def _build_stream(self) -> Iterator:
        ds = load_dataset(
            self.dataset_name,
            self.subset,
            split=self.split,
            streaming=True,
            token=self.hf_token,
            trust_remote_code=True,
        )
        if self.world_size > 1:
            ds = ds.shard(num_shards=self.world_size, index=self.rank)
        return iter(ds)

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        stream      = self._build_stream()

        if worker_info is not None:
            n_w, w_id = worker_info.num_workers, worker_info.id
            stream = (ex for i, ex in enumerate(stream) if i % n_w == w_id)

        token_buffer: list[int] = []

        for example in stream:
            text = example.get(self.text_column, "")
            if not text or not text.strip():
                continue

            ids = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]

            token_buffer.extend(ids)

            while len(token_buffer) >= self.max_seq_len:
                chunk        = token_buffer[: self.max_seq_len]
                token_buffer = token_buffer[self.max_seq_len :]
                yield torch.tensor(chunk, dtype=torch.long)


# ===========================================================================
# LR schedule: linear warm-up → cosine decay
# ===========================================================================

def get_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ===========================================================================
# Checkpoint helpers
# ===========================================================================

def save_checkpoint(
    sae:            nn.Module,
    optimizer:      torch.optim.Optimizer,
    step:           int,
    metrics:        dict,
    checkpoint_dir: str,
    rank:           int,
) -> None:
    if not is_main(rank):
        return
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    raw_sae   = sae.module if isinstance(sae, DDP) else sae
    ckpt_path = Path(checkpoint_dir) / f"step_{step:09d}.pt"
    torch.save(
        {
            "step":            step,
            "model_state":     raw_sae.state_dict(),  # includes steps_since_fired
            "optimizer_state": optimizer.state_dict(),
            "metrics":         metrics,
            "config": {
                "model_name":       MODEL_NAME,
                "hook_layer":       HOOK_LAYER,
                "hidden_dim":       HIDDEN_DIM,
                "sae_dim":          SAE_DIM,
                "top_k":            TOP_K,
                "ghost_threshold":  GHOST_THRESHOLD,
                "ghost_grad_coeff": GHOST_GRAD_COEFF,
                "learning_rate":    LEARNING_RATE,
                "batch_size":       BATCH_SIZE,
                "max_seq_len":      MAX_SEQ_LEN,
            },
        },
        ckpt_path,
    )
    latest = Path(checkpoint_dir) / "latest.pt"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(ckpt_path.name)


def load_checkpoint(
    path:      str,
    sae:       nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
) -> int:
    ckpt    = torch.load(path, map_location=device)
    raw_sae = sae.module if isinstance(sae, DDP) else sae
    raw_sae.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return int(ckpt["step"])


# ===========================================================================
# Training
# ===========================================================================

def train(args: argparse.Namespace) -> None:

    # ── Distributed setup ────────────────────────────────────────────────
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    logger = get_logger(rank, args.log_dir)
    logger.info(f"Rank {rank}/{world_size} | device: {device}")

    torch.manual_seed(42 + rank)

    # ── Tokeniser ────────────────────────────────────────────────────────
    logger.info("Loading tokeniser …")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, token=HF_TOKEN, use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Frozen LLM ───────────────────────────────────────────────────────
    logger.info(f"Loading {MODEL_NAME} (frozen) …")
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    llm.eval()
    for p in llm.parameters():
        p.requires_grad_(False)

    # ── Hook ─────────────────────────────────────────────────────────────
    hook = ResidualStreamHook()
    hook.register(llm, args.hook_layer)
    logger.info(f"Residual-stream hook registered at layer {args.hook_layer}")

    # ── SAE ──────────────────────────────────────────────────────────────
    sae = SparseAutoencoder(args.hidden_dim, args.sae_dim, args.top_k).to(device)

    if world_size > 1:
        # find_unused_parameters=True because the ghost path may not activate
        # every step (no dead neurons early in training)
        sae = DDP(sae, device_ids=[local_rank], find_unused_parameters=True)

    raw_sae: SparseAutoencoder = sae.module if isinstance(sae, DDP) else sae

    # ── Optimiser + scheduler ────────────────────────────────────────────
    optimizer = Adam(sae.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=get_lr_lambda(args.warmup_steps, args.total_steps),
    )

    # ── Optional resume ──────────────────────────────────────────────────
    start_step = 0
    if args.resume:
        resume_path = (
            args.resume
            if args.resume != "latest"
            else str(Path(args.checkpoint_dir) / "latest.pt")
        )
        if Path(resume_path).exists():
            start_step = load_checkpoint(resume_path, sae, optimizer, device)
            logger.info(f"Resumed from step {start_step}")
        else:
            logger.warning(f"Resume path not found: {resume_path}")

    # ── AMP context ──────────────────────────────────────────────────────
    # The SAE itself runs in float32 for numerical stability.
    # bfloat16 autocast is only applied to the frozen LLM forward pass.
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if USE_AMP and device.type == "cuda"
        else nullcontext()
    )

    # ── Dataset + DataLoader ─────────────────────────────────────────────
    logger.info("Building Dolma streaming dataset …")
    assert args.batch_size % world_size == 0, \
        "BATCH_SIZE must be divisible by world_size"
    per_rank_batch = args.batch_size // world_size

    dataset = DolmaStreamDataset(
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        rank=rank,
        world_size=world_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=per_rank_batch,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    # ── Training loop ────────────────────────────────────────────────────
    logger.info(
        f"Training | top_k={args.top_k} | ghost_threshold={args.ghost_threshold} "
        f"| ghost_coeff={args.ghost_coeff} | total_steps={args.total_steps}"
    )

    sae.train()
    step    = start_step
    running = {"total": 0.0, "mse": 0.0, "ghost": 0.0}
    t0      = time.perf_counter()

    for batch_tokens in dataloader:
        if step >= args.total_steps:
            break

        # ── Collect activations from frozen LLM ──────────────────────
        batch_tokens = batch_tokens.to(device, non_blocking=True)
        with torch.no_grad(), amp_ctx:
            _ = llm(input_ids=batch_tokens, use_cache=False)

        acts = hook.activations                          # [B, T, D]
        if acts is None:
            logger.warning("Hook returned None — skipping batch")
            continue

        # Flatten (B, T, D) → (B·T, D): each token is one sample
        B, T, D   = acts.shape
        acts_flat = acts.reshape(B * T, D).to(torch.float32)  # SAE in fp32

        # ── SAE forward + loss ───────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)

        out = sae(acts_flat, ghost_threshold=args.ghost_threshold)

        loss_dict = SparseAutoencoder.compute_loss(
            acts_flat, out, args.ghost_coeff
        )
        loss_dict["total"].backward()

        nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        # Maintain unit-norm decoder columns after every gradient step
        raw_sae._normalise_decoder_()

        # ── Update dead-neuron counters (synced across ranks) ────────
        with torch.no_grad():
            raw_sae.update_dead_neurons(out["z"])

        step += 1
        for k in running:
            running[k] += loss_dict[k].item()

        # ── Logging ──────────────────────────────────────────────────
        if step % args.log_every == 0 and is_main(rank):
            elapsed  = time.perf_counter() - t0
            avg      = {k: v / args.log_every for k, v in running.items()}
            lr_now   = scheduler.get_last_lr()[0]
            n_dead   = int(raw_sae.dead_mask(args.ghost_threshold).sum().item())
            dead_pct = 100.0 * n_dead / args.sae_dim
            # L0 is fixed == top_k with Top-K activation
            frac_act = 100.0 * args.top_k / args.sae_dim

            logger.info(
                f"step {step:>8d}/{args.total_steps} | "
                f"loss {avg['total']:.4f} | "
                f"mse {avg['mse']:.4f} | "
                f"ghost {avg['ghost']:.6f} | "
                f"dead {dead_pct:.1f}% ({n_dead}/{args.sae_dim}) | "
                f"L0={args.top_k} ({frac_act:.2f}%) | "
                f"lr {lr_now:.2e} | "
                f"{args.log_every / elapsed:.1f} it/s"
            )
            running = {k: 0.0 for k in running}
            t0 = time.perf_counter()

        # ── Checkpoint ───────────────────────────────────────────────
        if step % args.save_every == 0:
            metrics = {k: v / args.save_every for k, v in running.items()}
            save_checkpoint(sae, optimizer, step, metrics,
                            args.checkpoint_dir, rank)
            if is_main(rank):
                logger.info(f"Checkpoint saved at step {step}")

    # ── Final checkpoint ──────────────────────────────────────────────────
    save_checkpoint(sae, optimizer, step, {}, args.checkpoint_dir, rank)
    if is_main(rank):
        logger.info(f"Training complete at step {step}.")

    hook.remove()
    cleanup_distributed()


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Top-K SAE with Ghost Gradients on Gemma 2-2b-it"
    )

    # Architecture
    p.add_argument("--hook_layer", type=int,   default=HOOK_LAYER)
    p.add_argument("--hidden_dim", type=int,   default=HIDDEN_DIM)
    p.add_argument("--sae_dim",    type=int,   default=SAE_DIM)
    p.add_argument("--top_k",      type=int,   default=TOP_K,
                   help="Number of active latents per token (fixes L0)")

    # Ghost gradients
    p.add_argument("--ghost_threshold", type=int,   default=GHOST_THRESHOLD,
                   help="Steps without firing before ghost gradients are applied")
    p.add_argument("--ghost_coeff",     type=float, default=GHOST_GRAD_COEFF,
                   help="Weight of the ghost gradient loss term")

    # Training
    p.add_argument("--lr",           type=float, default=LEARNING_RATE)
    p.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    p.add_argument("--max_seq_len",  type=int,   default=MAX_SEQ_LEN)
    p.add_argument("--grad_clip",    type=float, default=GRAD_CLIP)
    p.add_argument("--total_steps",  type=int,   default=TOTAL_STEPS)
    p.add_argument("--warmup_steps", type=int,   default=WARMUP_STEPS)
    p.add_argument("--save_every",   type=int,   default=SAVE_EVERY)
    p.add_argument("--log_every",    type=int,   default=LOG_EVERY)
    p.add_argument("--num_workers",  type=int,   default=NUM_WORKERS)

    # Paths
    p.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR)
    p.add_argument("--log_dir",        type=str, default=LOG_DIR)
    p.add_argument("--resume",         type=str, default=None,
                   help="Path to checkpoint file, or 'latest' to auto-resume")

    return p.parse_args()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    args = parse_args()
    train(args)