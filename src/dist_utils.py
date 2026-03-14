import os
import time

import torch
import torch.distributed as dist

from config import (
    RAYON_NUM_THREADS,
    SEED,
    TOKENIZERS_PARALLELISM,
    TORCH_NUM_INTEROP_THREADS,
    TORCH_NUM_THREADS,
)


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

    os.environ.setdefault("TOKENIZERS_PARALLELISM", TOKENIZERS_PARALLELISM)
    os.environ.setdefault("RAYON_NUM_THREADS", str(RAYON_NUM_THREADS))

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if TORCH_NUM_THREADS > 0:
        torch.set_num_threads(TORCH_NUM_THREADS)
    if TORCH_NUM_INTEROP_THREADS > 0:
        torch.set_num_interop_threads(TORCH_NUM_INTEROP_THREADS)

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    return rank, local_rank, world_size, device, dtype


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    dist.barrier()


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
