"""Entry point for all stages.

    python main.py stage=<stage> [overrides]
    torchrun --nproc_per_node=N main.py stage=<stage> [overrides]

Stages: preflight, prepare_data, train_weak, annotate, train_strong_sft, compute_ms_weights,
train_strong_po, evaluate, collect_results."""

from __future__ import annotations

import importlib

import hydra
from omegaconf import DictConfig

from pipeline import STAGES
from utils.config import CONFIG_PATH


@hydra.main(config_path=CONFIG_PATH, config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.stage not in STAGES:
        raise ValueError(f"Unknown stage '{cfg.stage}'. Choose one of: {', '.join(STAGES)}")
    importlib.import_module(f"pipeline.{cfg.stage}").run(cfg)


if __name__ == "__main__":
    main()
