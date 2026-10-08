"""UltraFeedback Binarized (HuggingFaceH4/ultrafeedback_binarized).

No preprocessing script exists for UFB in resources/, so it mirrors the HH-RLHF pipeline:
prompt = the user prompt, chosen/rejected = the final assistant message of each conversation,
and the same max-length filter (resources/.../data_processing_hh_rlhf.py).
"""

from __future__ import annotations

from datasets import load_dataset

from dataset.hh_rlhf import filter_max_length


def convert(row):
    return {
        "prompt": row["prompt"].strip(),
        "chosen": row["chosen"][-1]["content"].strip(),
        "rejected": row["rejected"][-1]["content"].strip(),
    }


def load(cfg, split, tokenizer, num_proc=4):
    raw = load_dataset(cfg.hf_path, split=split)
    processed = raw.map(convert, num_proc=num_proc, remove_columns=raw.column_names)
    processed = processed.filter(lambda x: x["prompt"] and x["chosen"] and x["rejected"], num_proc=num_proc)
    return filter_max_length(processed, tokenizer, cfg.max_length, num_proc)
