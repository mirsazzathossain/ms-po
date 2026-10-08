"""TL;DR preprocessing (from resources/ms_po_ours/datasets/tl_dr_dataset.py)."""

from __future__ import annotations

from datasets import load_dataset


def preprocess_tldr_for_dpo(row):
    """Raw comparison row -> {'prompt', 'chosen', 'rejected'}."""
    winning_index = row["choice"]
    losing_index = 1 - winning_index  # The opposite index is the rejected one

    prompt_text = row["info"]["post"]

    winning_summary = row["summaries"][winning_index]["text"]
    losing_summary = row["summaries"][losing_index]["text"]

    return {"prompt": prompt_text, "chosen": winning_summary, "rejected": losing_summary}


def length_filter(row, max_length=1024):
    token_count = len(row["prompt"].split())
    return token_count <= max_length


def load(cfg, split, tokenizer=None, num_proc=4):
    dataset = load_dataset(cfg.hf_path, cfg.hf_name, split=split)
    # validation also holds CNN/DM rows without a Reddit post
    dataset = dataset.filter(lambda row: row["info"]["post"] is not None, num_proc=num_proc)
    formatted = dataset.map(preprocess_tldr_for_dpo, num_proc=num_proc, remove_columns=dataset.column_names)
    return formatted.filter(length_filter, fn_kwargs={"max_length": cfg.max_length}, num_proc=num_proc)
