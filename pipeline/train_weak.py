"""Weak teacher: full fine-tuning, SFT then DPO on D_labeled.  main.py stage=train_weak"""

from __future__ import annotations

import logging

from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from trl import DPOTrainer, SFTTrainer

from models import load_causal_lm, load_tokenizer
from utils import dist
from utils.common import is_done, mark_done, processed_file, run_name, setup, validation_set
from utils.hub import push_folder
from utils.io import read_jsonl
from utils.logging import finish_wandb, setup_wandb
from utils.trainer import dpo_config, sft_config

log = logging.getLogger("mspo")


def _finish(trainer, tokenizer, out_dir, cfg, hub_name, msg):
    trainer.save_model(out_dir)
    if dist.is_main():
        tokenizer.save_pretrained(out_dir)
    dist.barrier()
    mark_done(out_dir)
    push_folder(cfg, out_dir, hub_name, msg)
    finish_wandb()


def train_sft(cfg, tokenizer, labeled):
    out_dir = cfg.paths.weak_sft_dir
    if is_done(out_dir, cfg):
        return
    to_sft = lambda ds: ds.map(  # noqa: E731
        lambda r: {"prompt": r["prompt"], "completion": r["chosen"]}, remove_columns=ds.column_names
    )
    name = run_name(cfg, f"weak-sft-{cfg.model.weak.short}")
    setup_wandb(cfg, name, "weak_sft")
    model = load_causal_lm(cfg.model.weak.name, cfg.model_dtype, pad_token_id=tokenizer.pad_token_id)
    trainer = SFTTrainer(
        model=model,
        args=sft_config(cfg, cfg.train.weak_sft, out_dir, name, cfg.dataset.max_length),
        train_dataset=to_sft(labeled),
        eval_dataset=to_sft(validation_set(cfg)),
        processing_class=tokenizer,
    )
    trainer.train()
    _finish(trainer, tokenizer, out_dir, cfg, f"{cfg.dataset.name}-{cfg.model.weak.short}-weak-sft", "weak SFT")


def train_dpo(cfg, tokenizer, labeled):
    out_dir = cfg.paths.weak_po_dir
    if is_done(out_dir, cfg):
        return
    name = run_name(cfg, f"weak-dpo-{cfg.model.weak.short}")
    setup_wandb(cfg, name, "weak_po")
    model = load_causal_lm(cfg.paths.weak_sft_dir, cfg.model_dtype, pad_token_id=tokenizer.pad_token_id)
    loss_cfg = OmegaConf.create({"beta": cfg.train.weak_po.beta, "trl_loss_type": "sigmoid"})  # Eq. 3
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL builds a frozen copy of the SFT model as pi_ref,w
        args=dpo_config(cfg, cfg.train.weak_po, out_dir, name, cfg.dataset.max_length, loss_cfg),
        train_dataset=labeled,
        eval_dataset=validation_set(cfg),
        processing_class=tokenizer,
    )
    trainer.train()
    _finish(trainer, tokenizer, out_dir, cfg, f"{cfg.dataset.name}-{cfg.model.weak.short}-weak-dpo", "weak DPO")


def run(cfg: DictConfig) -> None:
    setup(cfg, "train_weak")
    tokenizer = load_tokenizer(cfg.model.weak.name)
    labeled = Dataset.from_list(
        [{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in read_jsonl(processed_file(cfg, "labeled"))]
    )
    train_sft(cfg, tokenizer, labeled)
    train_dpo(cfg, tokenizer, labeled)
    dist.cleanup()
