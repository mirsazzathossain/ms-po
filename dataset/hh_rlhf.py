"""Anthropic HH-RLHF preprocessing (from resources/ms_po_ours/utils/data_processing_hh_rlhf.py)."""

from __future__ import annotations

from datasets import concatenate_datasets, load_dataset


def split_hh_conversation(text):
    """Split an Anthropic HH-RLHF conversation into prompt and response."""
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)

    if idx == -1:
        return None, None

    prompt = text[:idx].strip()
    response = text[idx + len(marker) :].strip()

    return prompt, response


def preprocess_hh_rlhf(dataset_split, tokenizer, is_train, max_length=512, num_proc=4):
    """Preprocess Anthropic HH-RLHF for DPO with batched processing."""

    def convert_batch(batch):
        prompts, ch_responses, rej_responses, valid = [], [], [], []

        for ch_text, rej_text in zip(batch["chosen"], batch["rejected"]):
            ch_p, ch_r = split_hh_conversation(ch_text)
            rej_p, rej_r = split_hh_conversation(rej_text)

            if ch_p and rej_p and ch_p == rej_p and ch_r and rej_r:
                prompts.append(ch_p)
                ch_responses.append(ch_r)
                rej_responses.append(rej_r)
                valid.append(True)
            else:
                prompts.append("")
                ch_responses.append("")
                rej_responses.append("")
                valid.append(False)

        return {"prompt": prompts, "chosen": ch_responses, "rejected": rej_responses, "is_valid": valid}

    processed = dataset_split.map(
        convert_batch,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset_split.column_names,
        desc="Parsing HH-RLHF conversations",
    )
    processed = processed.filter(
        lambda x: x["is_valid"], num_proc=num_proc, desc="Filtering invalid examples"
    ).remove_columns(["is_valid"])

    return filter_max_length(processed, tokenizer, max_length, num_proc)


def filter_max_length(processed, tokenizer, max_length, num_proc=4):
    """Token length check (filtering out large sequences)."""

    def calculate_length_batch(batch):
        texts = [
            p + "\n\nAssistant: " + c + "\n\nAssistant: " + r
            for p, c, r in zip(batch["prompt"], batch["chosen"], batch["rejected"])
        ]
        text_encs = tokenizer(texts, truncation=False, padding=False)["input_ids"]
        return {"under_max": [len(enc) <= max_length for enc in text_encs]}

    processed = processed.map(
        calculate_length_batch, batched=True, num_proc=num_proc, desc="Filtering max sequence lengths"
    )
    return processed.filter(
        lambda x: x["under_max"], num_proc=num_proc, desc=f"Filtering samples larger than {max_length} tokens"
    ).remove_columns(["under_max"])


def load(cfg, split, tokenizer, num_proc=4):
    parts = [load_dataset(cfg.hf_path, data_dir=subset, split=split) for subset in cfg.subsets]
    raw = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    return preprocess_hh_rlhf(raw, tokenizer, split, max_length=cfg.max_length, num_proc=num_proc)
