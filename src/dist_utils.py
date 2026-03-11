import os
import socket

import torch
import torch.distributed as dist

from config import REQUIRE_CUDA


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


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


def launched_with_torchrun() -> bool:
    return (
        "RANK" in os.environ
        and "LOCAL_RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and int(os.environ["WORLD_SIZE"]) >= 1
    )


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