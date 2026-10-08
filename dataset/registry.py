"""Dataset registry and splits, following resources/ms_po_ours/utils/data_processing_hh_rlhf.py:

    labeled, rest   = train.train_test_split(test_size=0.7, seed=42)   # D_labeled = 30%
    unlabeled, val  = rest.train_test_split(test_size=0.01, seed=42)   # D_unlabeled, validation
    test            = filtered test split
"""

from __future__ import annotations

import logging

from transformers import AutoTokenizer

from dataset import hh_rlhf, tldr, ufb

log = logging.getLogger("mspo")

LOADERS = {"hh_rlhf": hh_rlhf.load, "tldr": tldr.load, "ufb": ufb.load}


def build_splits(cfg, seed=42, num_proc=4):
    if cfg.loader not in LOADERS:
        raise KeyError(f"Unknown dataset loader '{cfg.loader}'. Options: {list(LOADERS)}")
    load = LOADERS[cfg.loader]

    tokenizer = AutoTokenizer.from_pretrained(cfg.length_tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train = load(cfg, cfg.train_split, tokenizer, num_proc)
    test = load(cfg, cfg.test_split, tokenizer, num_proc)

    split = train.train_test_split(test_size=cfg.unlabeled_fraction, seed=seed)
    labeled, rest = split["train"], split["test"]
    val_split = rest.train_test_split(test_size=cfg.validation_fraction, seed=seed)
    unlabeled, validation = val_split["train"], val_split["test"]

    log.info(
        "%s sizes: labeled %d | unlabeled %d | validation %d | test %d",
        cfg.name, len(labeled), len(unlabeled), len(validation), len(test),
    )
    return {"labeled": labeled, "unlabeled": unlabeled, "validation": validation, "test": test}
