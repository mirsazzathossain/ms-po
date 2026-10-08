"""TL;DR preprocessing, ported from resources/ms_po_ours/datasets/tl_dr_dataset.py.

OpenAI 'summarize_from_feedback' comparisons (Stiennon et al., 2020):
prompt = info["post"], chosen/rejected = summaries[choice] / summaries[1 - choice];
samples whose prompt exceeds 1024 whitespace tokens are dropped.
"""

from __future__ import annotations

from datasets import load_dataset


def preprocess_tldr_for_dpo(row):
    """Takes a raw row from the TL;DR comparisons dataset and transforms it into the standard
    format required by DPO/IPO/rDPO trainers: {'prompt': ..., 'chosen': ..., 'rejected': ...}"""
    # 'row["choice"]' is an integer index (0 or 1) indicating which summary
    # the human annotator preferred over the other.
    winning_index = row["choice"]
    losing_index = 1 - winning_index  # The opposite index is the rejected one

    # Accessing the original Reddit post text via the 'info' column dictionary
    prompt_text = row["info"]["post"]

    # Accessing the competing text strings from the 'summaries' list column using indices
    winning_summary = row["summaries"][winning_index]["text"]
    losing_summary = row["summaries"][losing_index]["text"]

    return {"prompt": prompt_text, "chosen": winning_summary, "rejected": losing_summary}


def length_filter(row, max_length=1024):
    # A quick whitespace/token approximation check
    token_count = len(row["prompt"].split())
    return token_count <= max_length


def load(cfg, split, tokenizer=None, num_proc=4):
    dataset = load_dataset(cfg.hf_path, cfg.hf_name, split=split)
    # The validation split also contains CNN/DM articles that have no Reddit post (info["post"] is None);
    # they are not TL;DR samples and would crash the length filter.
    dataset = dataset.filter(lambda row: row["info"]["post"] is not None, num_proc=num_proc)
    formatted = dataset.map(preprocess_tldr_for_dpo, num_proc=num_proc, remove_columns=dataset.column_names)
    return formatted.filter(length_filter, fn_kwargs={"max_length": cfg.max_length}, num_proc=num_proc)
