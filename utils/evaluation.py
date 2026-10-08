"""Evaluation (Sec. 5.2): sampling, gold reward models and Gold Reward Accuracy.

* Sampling: temperature 0.95, at most 512 new tokens (Appendix C.3).
* Gold RMs: Skywork/Skywork-Reward-V2-Llama-3.1-8B for HH-RLHF and UFB, scored on the chat
  [user: prompt, assistant: response] as in its model card; OpenAssistant/reward-model-deberta-v3-large-v2
  for TL;DR, scored on the (prompt, response) pair as in its model card.
* GRA = (1/N) sum_i 1[R(x_i, y_aligned) > R(x_i, y_SFT)] (Eq. 13).
"""

from __future__ import annotations

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils import dist


@torch.no_grad()
def generate_responses(model, tokenizer, prompts, eval_cfg, device) -> list[str]:
    model.eval()
    tokenizer.padding_side = "left"
    out = [""] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    bs = eval_cfg.generation_batch_size
    for s in tqdm(range(0, len(order), bs), desc="generate", disable=not dist.is_main()):
        idx = order[s : s + bs]
        batch = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True).to(device)
        gen = model.generate(
            **batch,
            do_sample=True,
            temperature=eval_cfg.temperature,
            max_new_tokens=eval_cfg.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = gen[:, batch["input_ids"].size(1) :]
        for j, i in enumerate(idx):
            out[i] = tokenizer.decode(new_tokens[j], skip_special_tokens=True)
    return out


class GoldRewardModel:
    def __init__(self, kind: str, name: str, dtype, device):
        self.kind = kind
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=dtype, num_labels=1).to(device)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def _encode(self, prompts, responses):
        if self.kind == "skywork":
            texts = []
            for p, r in zip(prompts, responses):
                conv = [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
                t = self.tokenizer.apply_chat_template(conv, tokenize=False)
                bos = self.tokenizer.bos_token
                if bos and t.startswith(bos):  # the tokenizer adds BOS again (model card)
                    t = t[len(bos) :]
                texts.append(t)
            return self.tokenizer(texts, return_tensors="pt", padding=True)
        if self.kind == "deberta":
            return self.tokenizer(list(prompts), list(responses), return_tensors="pt", padding=True, truncation=True)
        raise ValueError(f"Unknown reward model kind '{self.kind}'")

    @torch.no_grad()
    def score(self, prompts: list[str], responses: list[str], batch_size: int) -> list[float]:
        out = []
        for s in tqdm(range(0, len(prompts), batch_size), desc=f"reward[{self.kind}]", disable=not dist.is_main()):
            enc = self._encode(prompts[s : s + batch_size], responses[s : s + batch_size]).to(self.device)
            out.extend(self.model(**enc).logits[:, 0].float().tolist())
        return out


def gold_reward_accuracy(aligned_scores: list[float], sft_scores: list[float]) -> float:
    """GRA (Eq. 13): strictly higher gold reward than the SFT response; ties count as losses."""
    assert len(aligned_scores) == len(sft_scores) and aligned_scores
    return sum(a > s for a, s in zip(aligned_scores, sft_scores)) / len(aligned_scores)
