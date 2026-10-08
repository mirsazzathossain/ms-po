"""Single entry point for every MS-PO pipeline stage.

    python main.py stage=<stage> [hydra overrides]
    torchrun --nproc_per_node=<N> main.py stage=<stage> [hydra overrides]     # multi-GPU (DDP)

Stages (see pipeline/__init__.py):
    prepare_data        download, length-filter and 30/70 split a dataset
    train_weak          weak teacher SFT + DPO on D_labeled
    annotate            weak pseudo-labels + teacher confidence C_weak on D_unlabeled
    train_strong_sft    strong student SFT (LoRA), human or weak labels
    compute_ms_weights  teacher-student token KL S(x, y+), S(x, y-) for MS-PO
    train_strong_po     strong preference optimisation (human / ws_po / cw_po / ms_po x dpo / ipo / rdpo / simpo)
    evaluate            Gold Reward Accuracy vs. the SFT model
    collect_results     aggregate GRA results into a table

Example:
    python main.py stage=train_strong_po dataset=hh_rlhf model=opt method=ms_po loss=dpo
"""

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
