"""Model / tokenizer loading. Strong checkpoints are LoRA adapters with a `lineage.json`
listing the base model and the adapters to merge (relative paths)."""

from __future__ import annotations

import json
import logging
import os
import shutil

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger("mspo")

LINEAGE = "lineage.json"
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def load_tokenizer(name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(name_or_path: str, dtype: str = "fp32", device=None, pad_token_id: int | None = None):
    model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=DTYPES[dtype])
    if pad_token_id is not None:
        model.config.pad_token_id = pad_token_id
    if device is not None:
        model.to(device)
    return model


def lora_config(lora_cfg, target_modules) -> LoraConfig | None:
    """LoRA configuration from resources/ms_po_ours/utils/mange_config.py."""
    if lora_cfg is None:
        return None
    return LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(target_modules),
    )


def read_lineage(path: str) -> dict | None:
    f = os.path.join(path, LINEAGE)
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        return json.load(fh)


def save_lineage(out_dir: str, base: str, adapters: list[str]) -> None:
    with open(os.path.join(out_dir, LINEAGE), "w") as fh:
        json.dump({"base": base, "adapters": adapters}, fh, indent=2)


def copy_adapter(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        if name.startswith("adapter_"):
            shutil.copy2(os.path.join(src, name), os.path.join(dst, name))


def load_merged(path_or_name: str, dtype: str = "fp32", device=None, pad_token_id: int | None = None):
    """Full checkpoint, or base + merged LoRA adapters for a directory with lineage.json."""
    lineage = read_lineage(path_or_name) if os.path.isdir(path_or_name) else None
    if lineage is None:
        return load_causal_lm(path_or_name, dtype, device, pad_token_id)
    model = load_causal_lm(lineage["base"], dtype, None, pad_token_id)
    for a in lineage["adapters"]:
        adapter = a if os.path.isabs(a) else os.path.normpath(os.path.join(path_or_name, a))
        log.info("Merging adapter %s", adapter)
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    if device is not None:
        model.to(device)
    return model


def check_shared_vocab(tok_a, tok_b) -> None:
    """Token-level KL (Eq. 9) needs teacher and student to share token ids."""
    if tok_a.get_vocab() != tok_b.get_vocab():
        raise ValueError("Weak and strong tokenizers differ; use weak/strong models from the same family.")
