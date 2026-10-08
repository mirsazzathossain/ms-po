"""Stage 0: download, parse, length-filter and split a preference dataset.

Writes data/processed/<dataset>/{labeled,unlabeled,test}.jsonl and stats.json.

    python main.py stage=prepare_data dataset=hh_rlhf
"""

from __future__ import annotations

import logging
import os

from omegaconf import DictConfig

from dataset import build_splits
from utils.common import is_done, setup
from utils import dist
from utils.io import write_json, write_jsonl

log = logging.getLogger("mspo")


def run(cfg: DictConfig) -> None:
    setup(cfg, "prepare_data")
    out = cfg.paths.processed_dir
    if not dist.is_main() or is_done(os.path.join(out, "stats.json"), cfg):
        return
    splits = build_splits(cfg.dataset, seed=cfg.seed, num_proc=4)
    stats = {}
    for name, ds in splits.items():
        stats[name] = write_jsonl(os.path.join(out, f"{name}.jsonl"), ds)
    write_json(os.path.join(out, "stats.json"), stats)
    log.info("Wrote %s: %s", out, stats)

