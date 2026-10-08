"""JSONL I/O and shard merging."""

from __future__ import annotations

import json
import os
from typing import Iterable

from utils import dist


def write_jsonl(path: str, records: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = 0
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def shard_path(path: str, r: int) -> str:
    return f"{path}.shard{r}"


def gather_shards(path: str, local_records: list[dict]) -> list[dict] | None:
    """Merge per-rank records (with `_idx`) into `path` in order; rank 0 returns them."""
    write_jsonl(shard_path(path, dist.rank()), local_records)
    dist.barrier()
    merged = None
    if dist.is_main():
        merged = []
        for r in range(dist.world_size()):
            sp = shard_path(path, r)
            merged.extend(read_jsonl(sp))
            os.remove(sp)
        merged.sort(key=lambda x: x["_idx"])
        for m in merged:
            m.pop("_idx")
        write_jsonl(path, merged)
    dist.barrier()
    return merged
