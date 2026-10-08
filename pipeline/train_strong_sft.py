"""LoRA SFT of the strong student on human or weak labels.  main.py stage=train_strong_sft method=<m>"""

from __future__ import annotations

import logging

from datasets import Dataset
from omegaconf import DictConfig
from trl import SFTTrainer

from models import load_causal_lm, load_tokenizer, lora_config, save_lineage
from utils import dist
from utils.common import is_done, mark_done, preference_records, run_name, setup, validation_set
from utils.hub import push_folder
from utils.logging import finish_wandb, setup_wandb
from utils.trainer import sft_config

log = logging.getLogger("mspo")


def run(cfg: DictConfig) -> None:
    setup(cfg, "train_strong_sft")
    out_dir = cfg.paths.strong_sft_dir
    if is_done(out_dir, cfg):
        return
    st = cfg.train.strong_sft
    tok = load_tokenizer(cfg.model.strong.name)

    train_ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["chosen"]} for r in preference_records(cfg)])
    eval_ds = validation_set(cfg).map(
        lambda r: {"prompt": r["prompt"], "completion": r["chosen"]}, remove_columns=["chosen", "rejected"]
    )

    name = run_name(cfg, f"strong-sft-{cfg.model.strong.short}-{cfg.method.label_source}")
    setup_wandb(cfg, name, "strong_sft")
    model = load_causal_lm(cfg.model.strong.name, cfg.model_dtype, pad_token_id=tok.pad_token_id)
    trainer = SFTTrainer(
        model=model,
        args=sft_config(cfg, st, out_dir, name, cfg.dataset.max_length),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=lora_config(st.lora, cfg.model.lora_target_modules),
    )
    trainer.train()
    trainer.save_model(out_dir)
    if dist.is_main():
        tok.save_pretrained(out_dir)
        save_lineage(out_dir, cfg.model.strong.name, ["."])
    dist.barrier()
    mark_done(out_dir)
    push_folder(
        cfg, out_dir,
        f"{cfg.dataset.name}-{cfg.model.strong.short}-sft-{cfg.method.label_source}",
        f"strong SFT ({cfg.method.label_source} labels)",
    )
    finish_wandb()
    dist.cleanup()
