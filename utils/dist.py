"""Minimal multi-GPU helpers.

Training stages run under `torchrun` / `accelerate launch` and rely on the HF Trainer for DDP.
Inference stages (annotation, MS weights, generation, reward scoring) shard the dataset across
ranks, each rank writes its shard, and rank 0 merges the shards in the original order.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.distributed as dist


def rank() -> int:
    return int(os.environ.get("RANK", 0))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main() -> bool:
    return rank() == 0


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank())
    return torch.device("cpu")


def init() -> None:
    """Initialise the process group when launched with torchrun (no-op for single process)."""
    if world_size() > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank())
        dist.init_process_group(backend=backend)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@contextmanager
def main_first():
    """Let rank 0 run a block (e.g. downloads / caching) before the other ranks."""
    if not is_main():
        barrier()
    yield
    if is_main():
        barrier()


def shard_indices(n: int) -> list[int]:
    """Strided shard of range(n) for the current rank."""
    return list(range(rank(), n, world_size()))
