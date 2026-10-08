"""Weights & Biases integration."""

from __future__ import annotations

import logging
import os

from omegaconf import DictConfig

from utils import dist
from utils.config import to_container

log = logging.getLogger("mspo")


def setup_wandb(cfg: DictConfig, run_name: str, job_type: str):
    """Start a W&B run on rank 0 (HF Trainer reuses it). Returns the run or None."""
    if not cfg.logger.enabled:
        os.environ["WANDB_DISABLED"] = "true"
        return None
    os.environ.setdefault("WANDB_PROJECT", cfg.logger.project)
    if not dist.is_main():
        return None
    import wandb

    return wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        group=cfg.logger.group,
        tags=list(cfg.logger.tags),
        name=run_name,
        job_type=job_type,
        config=to_container(cfg),
        reinit=True,
    )


def report_to(cfg: DictConfig) -> list[str]:
    return ["wandb"] if cfg.logger.enabled else []


def wandb_log(metrics: dict, summary: bool = True) -> None:
    if not dist.is_main():
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    wandb.log(metrics)
    if summary:
        for k, v in metrics.items():
            wandb.run.summary[k] = v


def finish_wandb() -> None:
    if not dist.is_main():
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is not None:
        wandb.finish()
