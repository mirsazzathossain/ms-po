"""MS-PO / CW-PO confidence scores, ported from resources/ms_po_ours/utils/Compute_MS_PO.py
(and the weak variant of resources/ms_po_ours/utils/MS_PO_1.py).

Per response y of a pair (Eq. 8-10):
    S(x, y)       = (1/T) sum_t sum_v pi_w(v|y<t,x) (log pi_w(v|y<t,x) - log pi_s(v|y<t,x))
    C_align(x, y) = exp(-gamma * S(x, y))
Per pair:
    C_weak (CW-PO, Eq. 5) = clamp(2 * (sigmoid(p_w(y+) - p_w(y-)) - 0.5), 0, 1),
                            p_w = length-normalised (geometric-mean) token probability under pi_w
    C_MS   (Eq. 11)       = 2 * (sigmoid(C_align(y+) - C_align(y-)) - 0.5)
"""

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
    # Models of one family may pad their embedding matrices to different sizes
    # (e.g. Qwen2.5-0.5B vs 7B); every real token id lies in the common prefix.
    vocab = weak_logits.size(-1) if student_logits is None else min(weak_logits.size(-1), student_logits.size(-1))

    s_xy, seq_logprobs, token_counts = [], [], []
    for i in range(input_ids.size(0)):
        # Shift logits and labels for autoregressive sequence prediction
        weak_logprobs = F.log_softmax(weak_logits[i, :-1, :vocab].float(), dim=-1)
        m = mask[i]
        token_count = int(m.sum().item())

        if student_logits is not None:
            # Full Vocabulary Token-Level KL Divergence: D_t
            student_logprobs = F.log_softmax(student_logits[i, :-1, :vocab].float(), dim=-1)
            weak_probs = torch.exp(weak_logprobs)
            token_kl = torch.sum(weak_probs * (weak_logprobs - student_logprobs), dim=-1)
            s_xy.append(((token_kl * m).sum() / max(token_count, 1)).item())
        else:
            s_xy.append(0.0)

        # Log-probabilities of the actual response tokens under the weak model (for C_weak)
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
