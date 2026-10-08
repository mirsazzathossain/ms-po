"""Weak pseudo-labels y+ = argmax r_w (Eq. 2, 4) and teacher confidence C_weak (Eq. 5)."""

from __future__ import annotations

from utils.losses import compute_c_weak_length_normalized


def annotate_pairs(records, logp_w, logp_ref, lens, beta_w):
    """Relabel human pairs (y1, y2) with the weak teacher; inputs are (y1 values, y2 values)."""
    out = []
    for i, rec in enumerate(records):
        y = (rec["chosen"], rec["rejected"])
        r = [beta_w * (logp_w[k][i] - logp_ref[k][i]) for k in (0, 1)]
        pos = 0 if r[0] >= r[1] else 1
        neg = 1 - pos
        out.append(
            {
                "prompt": rec["prompt"],
                "chosen": y[pos],
                "rejected": y[neg],
                "human_chosen": rec["chosen"],
                "human_rejected": rec["rejected"],
                "weak_agrees_human": pos == 0,
                "r_w_chosen": r[pos],
                "r_w_rejected": r[neg],
                "c_weak": compute_c_weak_length_normalized(
                    logprob_chosen=logp_w[pos][i],
                    logprob_rejected=logp_w[neg][i],
                    len_chosen=lens[pos][i],
                    len_rejected=lens[neg][i],
                ),
            }
        )
    return out
