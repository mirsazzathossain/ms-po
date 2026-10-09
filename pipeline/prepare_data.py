"""Load, filter and split a dataset.  main.py stage=prepare_data dataset=<name>"""

from __future__ import annotations

import logging
import os

from omegaconf import DictConfig

from dataset import build_splits
from utils.common import is_done, setup
from utils import dist, hub
from utils.io import write_json, write_jsonl

log = logging.getLogger("mspo")


def run(cfg: DictConfig) -> None:
    setup(cfg, "prepare_data")
    out = cfg.paths.processed_dir
    files = [os.path.join(out, f"{s}.jsonl") for s in ("labeled", "unlabeled", "validation", "test")]
    if is_done(os.path.join(out, "stats.json"), cfg, also=tuple(files)) or not dist.is_main():
        return
    splits = build_splits(cfg.dataset, seed=cfg.seed, num_proc=4)
    if cfg.debug_max_samples:  # smoke tests only: keep the first N samples of every split
        splits = {k: ds.select(range(min(cfg.debug_max_samples, len(ds)))) for k, ds in splits.items()}
    stats = {}
    for name, ds in splits.items():
        stats[name] = write_jsonl(os.path.join(out, f"{name}.jsonl"), ds)
    write_json(os.path.join(out, "stats.json"), stats)
    log.info("Wrote %s: %s", out, stats)
    hub.push_files(cfg, files + [os.path.join(out, "stats.json")], f"prepare_data {cfg.dataset.name}")

