"""Weighted preference optimisation of the strong student (Eq. 6 / 12).  main.py stage=train_strong_po method=<m> loss=<l>"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
from datasets import Dataset
from omegaconf import DictConfig

from models import copy_adapter, load_merged, load_tokenizer, lora_config, save_lineage
from utils import dist
from utils.common import is_done, mark_done, preference_records, require, run_name, setup, validation_set
from utils.hub import push_folder
from utils.io import read_jsonl
from utils.logging import finish_wandb, setup_wandb, wandb_log
from utils.losses import c_align, ms_confidence
from utils.trainer import ConfidenceCollator, CWCPOTrainer, CWDPOTrainer, cpo_config, dpo_config

log = logging.getLogger("mspo")


def confidences(cfg, records):
    """Per-pair confidence C(x, y+, y-)."""
    kind = cfg.method.weight
    if kind == "uniform":
        return [1.0] * len(records)
    if kind == "cw":
        return [r["c_weak"] for r in records]
    if kind == "ms":
        ms_rows = read_jsonl(require(cfg.paths.ms_weights_file, "main.py stage=compute_ms_weights"))
        assert len(ms_rows) == len(records), "MS-PO statistics do not match the annotated dataset"
        c_pos = c_align(torch.tensor([m["S_chosen"] for m in ms_rows]), cfg.ms.gamma)
        c_neg = c_align(torch.tensor([m["S_rejected"] for m in ms_rows]), cfg.ms.gamma)
        c_weak = torch.tensor([r["c_weak"] for r in records])
        return ms_confidence(c_pos, c_neg, cfg.ms.variant, c_weak).tolist()
    raise ValueError(f"Unknown method.weight '{kind}'")


def run(cfg: DictConfig) -> None:
    setup(cfg, "train_strong_po")
    out_dir = cfg.paths.strong_po_dir
    if is_done(out_dir, cfg):
        return
    sft_dir = cfg.paths.strong_sft_dir
    require(os.path.join(sft_dir, "_SUCCESS"), "main.py stage=train_strong_sft")
    st, loss = cfg.train.strong_po, cfg.loss
    tok = load_tokenizer(sft_dir)

    records = preference_records(cfg)
    conf = confidences(cfg, records)
    train_ds = Dataset.from_list(
        [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"], "confidence": c}
         for r, c in zip(records, conf)]
    )
    eval_ds = validation_set(cfg).map(lambda r: {"confidence": 1.0})

    name = run_name(cfg, f"{cfg.model.strong.short}-{cfg.method.name}-{loss.name}{cfg.method.run_suffix}")
    setup_wandb(cfg, name, "strong_po")
    c = np.array(conf)
    wandb_log({"data/confidence_mean": float(c.mean()), "data/confidence_std": float(c.std()), "data/n_pairs": len(c)})

    model = load_merged(sft_dir, cfg.model_dtype, pad_token_id=tok.pad_token_id)
    peft_config = lora_config(st.lora, cfg.model.lora_target_modules)
    if loss.trainer == "dpo":
        trainer = CWDPOTrainer(
            model=model,
            ref_model=None,
            args=dpo_config(cfg, st, out_dir, name, cfg.dataset.max_length, loss),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tok,
            data_collator=ConfidenceCollator(pad_token_id=tok.pad_token_id),
            peft_config=peft_config,
        )
    elif loss.trainer == "cpo":
        trainer = CWCPOTrainer(
            model=model,
            args=cpo_config(cfg, st, out_dir, name, cfg.dataset.max_length, loss),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tok,
            peft_config=peft_config,
        )
    else:
        raise ValueError(f"Unknown loss.trainer '{loss.trainer}'")

    trainer.train()
    trainer.save_model(out_dir)
    if dist.is_main():
        tok.save_pretrained(out_dir)
        copy_adapter(sft_dir, os.path.join(out_dir, "sft_adapter"))
        save_lineage(out_dir, cfg.model.strong.name, ["sft_adapter", "."])
    dist.barrier()
    mark_done(out_dir)
    push_folder(
        cfg, out_dir,
        f"{cfg.dataset.name}-{cfg.model.strong.short}-{cfg.method.name}-{loss.name}{cfg.method.run_suffix}",
        f"{cfg.method.name} / {loss.name}",
    )
    finish_wandb()
    dist.cleanup()
