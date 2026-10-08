"""Batched teacher/student scoring with the tokenization of Compute_MS_PO.py."""

from __future__ import annotations

import torch
from tqdm import tqdm

from utils import dist
from utils.losses import evaluate_sequence


def _encode(tokenizer, prompts, responses, max_length):
    full = [tokenizer(p + r, max_length=max_length, truncation=True)["input_ids"] for p, r in zip(prompts, responses)]
    plens = [len(tokenizer(p, max_length=max_length, truncation=True)["input_ids"]) for p in prompts]
    return full, plens


def _pad(seqs, pad_id):
    n = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        att[i, : len(s)] = 1
    return ids, att


@torch.no_grad()
def score_pairs(weak_model, other_model, tokenizer, prompts, responses, batch_size, max_length, device, other_is_student):
    """Returns (S(x,y) if other_is_student else other's log-prob, weak log-prob, token count)."""
    full, plens = _encode(tokenizer, prompts, responses, max_length)
    n = len(full)
    out_other, out_weak, out_len = [0.0] * n, [0.0] * n, [0] * n
    order = sorted(range(n), key=lambda i: len(full[i]))  # less padding; order restored below
    for s in tqdm(range(0, n, batch_size), desc="score", disable=not dist.is_main()):
        idx = order[s : s + batch_size]
        input_ids, attention_mask = _pad([full[i] for i in idx], tokenizer.pad_token_id)
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        p = [plens[i] for i in idx]

        weak_logits = weak_model(input_ids=input_ids, attention_mask=attention_mask).logits
        other_logits = other_model(input_ids=input_ids, attention_mask=attention_mask).logits
        if other_is_student:
            other, weak_lp, cnt = evaluate_sequence(weak_logits, other_logits, input_ids, attention_mask, p)
        else:
            _, weak_lp, cnt = evaluate_sequence(weak_logits, None, input_ids, attention_mask, p)
            _, other, _ = evaluate_sequence(other_logits, None, input_ids, attention_mask, p)
        for j, i in enumerate(idx):
            out_other[i], out_weak[i], out_len[i] = other[j], weak_lp[j], cnt[j]
    return out_other, out_weak, out_len
