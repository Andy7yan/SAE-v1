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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Config
# ============================================================

MODEL_NAME = "google/gemma-2-2b-it"
DOLMA_SAMPLE_URL = "https://olmo-data.org/dolma-v1_6-8B-sample/v1_5r2_sample-0000.json.gz"

HOOK_LAYER_INDEX = 12
MAX_SEQ_LEN = 64

# LLM 推理阶段的 Batch Size (用于快速榨取激活值)
TEXT_BATCH_SIZE_PER_RANK = 16 

# ============================================================
# SOTA 架构新增：激活缓冲区与 SAE 独立 Batch 配置
# ============================================================
SAE_BATCH_SIZE = 4096       # 真正喂给 SAE 进行一次梯度下降的独立批次大小
BUFFER_CAPACITY = 131072    # 全局激活缓冲区容量 (13万个 Token 约等于打乱 32 个 SAE Batch)

# Training length
TRAIN_STEPS = 500           # 这里的 Step 现在严格指代 SAE 的参数更新次数
LOG_EVERY = 10
SAVE_EVERY = 250

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

OUTPUT_DIR = Path("./outputs/jumprelu_sae_decoupled")
DATA_CACHE_DIR = Path("./data_cache")
DATA_CACHE_PATH = DATA_CACHE_DIR / "v1_5r2_sample-0000.json.gz"


# ============================================================
# Early Exit Mechanism
# ============================================================

class EarlyExitException(Exception):
    """
    截获目标层激活后强行阻断 LLM 后续前向传播的自定义异常。
    """
    pass


# ============================================================
# Distributed helpers
# ============================================================

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0

def get_local_rank_from_env() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

def print0(msg: str) -> None:
    if get_rank() == 0:
        print(msg, flush=True)

def barrier() -> None:
    if is_distributed():
        dist.barrier()

def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    y = x.clone()
    if is_distributed():
        dist.all_reduce(y, op=dist.ReduceOp.AVG)
    return y

def setup_process(rank: int, local_rank: int, world_size: int, init_method: str | None) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method=init_method or "env://", rank=rank, world_size=world_size)
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
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# ============================================================
# Data & Buffer helpers
# ============================================================

class ActivationBuffer:
    """
    全局激活缓冲区：实现了 LLM 数据流与 SAE 训练流的彻底解耦。
    """
    def __init__(self, capacity: int, d_in: int, device: torch.device):
        self.capacity = capacity
        self.d_in = d_in
        self.device = device
        
        # self.acts: 仅存储通过掩码筛选的有效 Token 激活
        # Type: list[torch.Tensor], 每个元素的 Shape: [V_i, d_in], Dtype: torch.float32
        self.acts: list[torch.Tensor] = []
        self.current_size = 0

    def add(self, act: torch.Tensor, mask: torch.Tensor) -> None:
        # act Type: torch.Tensor, Shape: [B*S, d_in]
        # mask Type: torch.Tensor, Shape: [B*S], Dtype: torch.bool
        
        # 利用布尔索引提取有效 Token，这一步直接剥离了大量的无用计算
        # valid_acts Type: torch.Tensor, Shape: [V, d_in], (V 为有效 Token 数量)
        valid_acts = act[mask] 
        self.acts.append(valid_acts)
        self.current_size += valid_acts.shape[0]

    def is_ready(self) -> bool:
        return self.current_size >= self.capacity

    def get_shuffled_batches(self, batch_size: int) -> Iterator[torch.Tensor]:
        """
        拼接所有缓存的有效张量，执行全局打乱，并切分为标准的 SAE Batch。
        """
        # all_acts Type: torch.Tensor, Shape: [current_size, d_in]
        all_acts = torch.cat(self.acts, dim=0)
        
        # 核心：消除序列相关性的全局随机打乱
        indices = torch.randperm(self.current_size, device=self.device)
        shuffled_acts = all_acts[indices]

        for i in range(0, self.current_size, batch_size):
            end_idx = min(i + batch_size, self.current_size)
            # 严格保证每个传递给 SAE 的 Batch Size 一致，方便 DDP 梯度对齐
            if end_idx - i == batch_size:
                # yield Type: torch.Tensor, Shape: [batch_size, d_in]
                yield shuffled_acts[i:end_idx]

        # 保留不够一个 Batch 的尾部数据，结转到下一轮
        leftover_size = self.current_size % batch_size
        if leftover_size > 0:
            self.acts = [shuffled_acts[-leftover_size:]]
            self.current_size = leftover_size
        else:
            self.acts = []
            self.current_size = 0

def extract_text_from_example(example: dict) -> str:
    for key in ["text", "content", "document", "body"]:
        if isinstance((value := example.get(key)), str) and value.strip():
            return value.strip()
    raise ValueError("Could not find text field.")

def ensure_local_dolma_shard(path: Path) -> Path:
    if get_rank() == 0 and not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(DOLMA_SAMPLE_URL, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response, open(path, "wb") as f:
            f.write(response.read())
    barrier()
    return path

def iter_rank_text_batches(shard_path: Path, local_batch_size: int, rank: int, world_size: int) -> Iterator[list[str]]:
    assigned_texts, current_group = [], []
    with gzip.open(shard_path, "rt", encoding="utf-8") as gz_file:
        for raw_line in gz_file:
            try:
                text = extract_text_from_example(json.loads(raw_line))
                if len(text) < MIN_TEXT_CHARS: continue
            except Exception: continue

            current_group.append(text)
            if len(current_group) == world_size:
                assigned_texts.append(current_group[rank])
                current_group.clear()
                if len(assigned_texts) == local_batch_size:
                    yield assigned_texts
                    assigned_texts = []

# ============================================================
# JumpReLU STE helpers
# ============================================================

def rectangle_window(x: torch.Tensor) -> torch.Tensor:
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)

class StepSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(x, threshold); ctx.bandwidth = bandwidth
        return (x > threshold).to(x.dtype)
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, threshold = ctx.saved_tensors
        threshold_grad = -(1.0 / ctx.bandwidth) * rectangle_window((x - threshold) / ctx.bandwidth) * grad_output
        while threshold_grad.ndim > threshold.ndim: threshold_grad = threshold_grad.sum(dim=0)
        return torch.zeros_like(x), threshold_grad, None

class JumpReLUSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(x, threshold); ctx.bandwidth = bandwidth
        return x * (x > threshold).to(x.dtype)
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, threshold = ctx.saved_tensors
        x_grad = (x > threshold).to(x.dtype) * grad_output
        threshold_grad = -(threshold / ctx.bandwidth) * rectangle_window((x - threshold) / ctx.bandwidth) * grad_output
        while threshold_grad.ndim > threshold.ndim: threshold_grad = threshold_grad.sum(dim=0)
        return x_grad, threshold_grad, None

def step_ste(x, t, b): return StepSTE.apply(x, t, b)
def jumprelu_ste(x, t, b): return JumpReLUSTE.apply(x, t, b)

# ============================================================
# SAE Module
# ============================================================

class TinyJumpReLUSAE(nn.Module):
    def __init__(self, d_in: int, latent_factor: int = 4, input_scale: float = 1.0, init_threshold: float = 1e-3, ste_bandwidth: float = 1e-3):
        super().__init__()
        self.d_in, self.d_latent, self.ste_bandwidth = d_in, d_in * latent_factor, ste_bandwidth
        self.register_buffer("input_scale", torch.tensor(float(input_scale), dtype=torch.float32))
        self.W_enc = nn.Parameter(torch.empty(self.d_latent, d_in))
        self.b_enc = nn.Parameter(torch.zeros(self.d_latent))
        self.W_dec = nn.Parameter(torch.empty(d_in, self.d_latent))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.log_threshold = nn.Parameter(torch.full((self.d_latent,), math.log(init_threshold), dtype=torch.float32))
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        self._normalise_decoder_()
        with torch.no_grad(): self.W_enc.copy_(self.W_dec.T)

    @torch.no_grad()
    def _normalise_decoder_(self):
        norms = self.W_dec.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    @torch.no_grad()
    def remove_decoder_grad_parallel_(self):
        if self.W_dec.grad is None: return
        unit_cols = self.W_dec.data / self.W_dec.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.grad.sub_((self.W_dec.grad * unit_cols).sum(dim=0, keepdim=True) * unit_cols)

    def threshold(self): return torch.exp(self.log_threshold)

    def forward(self, x: torch.Tensor):
        x_scaled = x / self.input_scale
        x_in = x_scaled - self.b_dec
        pre_acts = F.relu(x_in @ self.W_enc.T + self.b_enc)
        z = jumprelu_ste(pre_acts, self.threshold(), self.ste_bandwidth)
        return z @ self.W_dec.T + self.b_dec, z, pre_acts, x_scaled

# ============================================================
# Main Training Logic
# ============================================================

def module_of(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model

def run_training(rank: int, local_rank: int, world_size: int, init_method: str | None) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True
    
    device = setup_process(rank, local_rank, world_size, init_method)
    set_seed(SEED)
    model_dtype = get_model_dtype(device)
    hf_token = os.environ.get("HF_TOKEN", None)

    shard_path = ensure_local_dolma_shard(DATA_CACHE_PATH)
    if rank == 0:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=hf_token, dtype=model_dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    activation_store = {}
    def hook_fn(module, inputs, output):
        activation_store["act"] = output[0] if isinstance(output, (tuple, list)) else output
        raise EarlyExitException() # 物理截断 LLM 计算图
    model.model.layers[HOOK_LAYER_INDEX].register_forward_hook(hook_fn)

    sae_model, optimizer, buffer = None, None, None
    sae_step = 0

    print0("[rank=0] System ready. Starting decoupled training pipeline...")

    while sae_step < TRAIN_STEPS:
        for texts in iter_rank_text_batches(shard_path, TEXT_BATCH_SIZE_PER_RANK, rank, world_size):
            if sae_step >= TRAIN_STEPS: break
            
            # =================================================================
            # 管线 1：推断与数据收集 (LLM Forward)
            # =================================================================
            batch = tokenizer(texts, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
            batch = {k: v.to(device) for k, v in batch.items()}
            
            activation_store.clear()
            try:
                with torch.no_grad():
                    _ = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            except EarlyExitException: pass

            act = activation_store["act"] # Type: torch.Tensor, Shape: [B, S, d_in]
            d_in = act.shape[-1]

            if buffer is None:
                buffer = ActivationBuffer(capacity=BUFFER_CAPACITY, d_in=d_in, device=device)

            # SOTA 工程规范：丢弃 BOS (序列第一个词元) 以防止 Attention Sink 破坏 SAE
            mask = batch["attention_mask"].clone()
            mask[:, 0] = 0 # 将第 0 列置零
            
            # 展平以便装入 Buffer
            mask_flat = mask.reshape(-1).to(device=device, dtype=torch.bool) # Shape: [B*S]
            raw_x = act.to(dtype=torch.float32).reshape(-1, d_in)            # Shape: [B*S, d_in]
            
            buffer.add(raw_x, mask_flat)

            # =================================================================
            # 管线 2：解耦后的 SAE 独立更新 (SAE Backward)
            # =================================================================
            if buffer.is_ready():
                for x_batch in buffer.get_shuffled_batches(SAE_BATCH_SIZE):
                    if sae_step >= TRAIN_STEPS: break
                    
                    if sae_model is None:
                        base_sae = TinyJumpReLUSAE(d_in, LATENT_FACTOR, ACT_NORM_SCALE, INIT_THRESHOLD, STE_BANDWIDTH).to(device)
                        sae_model = DDP(base_sae, device_ids=[local_rank], output_device=local_rank) if world_size > 1 else base_sae
                        optimizer = torch.optim.Adam(sae_model.parameters(), lr=LR, betas=(0.0, 0.999))
                        print0(f"[rank=0] SAE Init. Decoupled SAE Batch: {SAE_BATCH_SIZE}")

                    sae_step += 1
                    sae_base = module_of(sae_model)
                    
                    # x_batch Type: torch.Tensor, Shape: [SAE_BATCH_SIZE, d_in], Dtype: torch.float32
                    x_hat, z, pre_acts, x_scaled = sae_model(x_batch)

                    # 计算重构损失 (均方误差)
                    recon_loss = ((x_hat - x_scaled) ** 2).mean()

                    # 计算 L0 稀疏性代理损失
                    theta = sae_base.threshold()
                    l0_proxy = step_ste(pre_acts, theta, sae_base.ste_bandwidth).sum(dim=-1)
                    l0_loss = l0_proxy.mean() * L0_COEFF

                    loss = recon_loss + l0_loss

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    sae_base.remove_decoder_grad_parallel_()
                    optimizer.step()
                    sae_base._normalise_decoder_()

                    # 日志记录 (同步各节点的全局均值)
                    if sae_step == 1 or sae_step % LOG_EVERY == 0:
                        global_recon = float(all_reduce_mean(recon_loss.detach()).item())
                        global_l0 = float(all_reduce_mean(l0_loss.detach()).item()) / L0_COEFF
                        print0(f"[rank=0] SAE Step={sae_step:05d}/{TRAIN_STEPS} | "
                               f"Recon_MSE={global_recon:.6f} | "
                               f"Avg_L0={global_l0:.3f} | "
                               f"Theta_Mean={float(theta.mean()):.6f}")

    cleanup_process()
    print0(f"[rank=0] TRAINING COMPLETE.")

def main() -> None:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Launch Violation: Must be launched using torchrun.")
    run_training(int(os.environ["RANK"]), int(os.environ.get("LOCAL_RANK", "0")), int(os.environ["WORLD_SIZE"]), "env://")

if __name__ == "__main__":
    main()