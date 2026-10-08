"""Confidence scores: S, C_align, C_weak, C_MS (Eq. 5, 8-11, 25-28), from Compute_MS_PO.py."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def response_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, prompt_lens: list[int]) -> torch.Tensor:
    """Mask over shifted positions (labels = input_ids[:, 1:]) that predict response tokens:
    prompt positions and padding are zeroed, exactly as in Compute_MS_PO.evaluate_sequence."""
    mask = torch.ones_like(input_ids[:, 1:], dtype=torch.float32)
    for i, plen in enumerate(prompt_lens):
        mask[i, : (plen - 1)] = 0.0  # Zero out prompt tokens
    return mask * attention_mask[:, 1:]  # Zero out padding tokens


def evaluate_sequence(
    weak_logits: torch.Tensor,
    student_logits: torch.Tensor | None,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: list[int],
) -> tuple[list[float], list[float], list[int]]:
    """Sequence discrepancy S(x, y), weak sequence log-prob and response token count, per row.

    Computed row by row to bound memory; the math is Compute_MS_PO.evaluate_sequence.
    `student_logits=None` skips the KL (S returned as 0)."""
    labels = input_ids[:, 1:]
    mask = response_mask(input_ids, attention_mask, prompt_lens)
    # embedding sizes can differ within a family (Qwen2.5-0.5B vs 7B)
    vocab = weak_logits.size(-1) if student_logits is None else min(weak_logits.size(-1), student_logits.size(-1))

    s_xy, seq_logprobs, token_counts = [], [], []
    for i in range(input_ids.size(0)):
        weak_logprobs = F.log_softmax(weak_logits[i, :-1, :vocab].float(), dim=-1)
        m = mask[i]
        token_count = int(m.sum().item())

        if student_logits is not None:
            student_logprobs = F.log_softmax(student_logits[i, :-1, :vocab].float(), dim=-1)
            weak_probs = torch.exp(weak_logprobs)
            token_kl = torch.sum(weak_probs * (weak_logprobs - student_logprobs), dim=-1)
            s_xy.append(((token_kl * m).sum() / max(token_count, 1)).item())
        else:
            s_xy.append(0.0)

        selected = torch.gather(weak_logprobs, dim=-1, index=labels[i].unsqueeze(-1)).squeeze(-1)
        seq_logprobs.append((selected * m).sum().item())
        token_counts.append(token_count)
    return s_xy, seq_logprobs, token_counts


def compute_c_weak_length_normalized(
    logprob_chosen: float, logprob_rejected: float, len_chosen: int, len_rejected: int
) -> float:
    """Computes C_weak using geometric mean token probabilities to prevent underflow."""
    mean_prob_chosen = torch.exp(torch.tensor(logprob_chosen / max(len_chosen, 1)))
    mean_prob_rejected = torch.exp(torch.tensor(logprob_rejected / max(len_rejected, 1)))

    prob_margin = mean_prob_chosen - mean_prob_rejected
    conf = 2.0 * (torch.sigmoid(prob_margin).item() - 0.5)
    return max(0.0, min(1.0, conf))


def c_align(s: torch.Tensor, gamma: float) -> torch.Tensor:
    """Student-aware alignment confidence C_align(x, y) = exp(-gamma * S(x, y)) (Eq. 10)."""
    return torch.exp(-gamma * s)


def ms_confidence(
    c_align_chosen: torch.Tensor,
    c_align_rejected: torch.Tensor,
    variant: str = "direct",
    c_weak: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-granular confidence C_MS (Eq. 11) and the Appendix-B variants.

    direct / normalized : 2 * (sigmoid(C+ - C-) - 0.5)      Eq. 11 / Eq. 28 (Compute_MS_PO.py)
    bounded             : sigmoid(C+ - C-)                  Eq. 27
    marginal            : C+ - C-                           Eq. 26
    weak                : C_weak * C+ * C-                  Eq. 25 (MS_PO_1.py)
    """
    c_delta = c_align_chosen - c_align_rejected
    if variant in ("direct", "normalized"):
        return 2.0 * (torch.sigmoid(c_delta) - 0.5)
    if variant == "bounded":
        return torch.sigmoid(c_delta)
    if variant == "marginal":
        return c_delta
    if variant == "weak":
        if c_weak is None:
            raise ValueError("variant='weak' needs C_weak")
        return c_weak * c_align_chosen * c_align_rejected
    raise ValueError(f"Unknown MS-PO variant '{variant}'")
