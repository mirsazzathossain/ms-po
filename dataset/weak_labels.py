"""Weak-teacher pseudo-labelling of D_unlabeled (Sec. 3.1) and teacher confidence C_weak (Eq. 5).

    r_w(x, y) = beta_w * log pi_w(y|x) / pi_ref,w(y|x)                      (Eq. 2)
    y+ = argmax_{y in {y1, y2}} r_w(x, y),  y- = argmin                     (Eq. 4)
    C_weak = compute_c_weak_length_normalized(pi_w on y+, pi_w on y-)       (Eq. 5, Compute_MS_PO.py)
"""

from __future__ import annotations

from utils.losses import compute_c_weak_length_normalized


def annotate_pairs(records, logp_w, logp_ref, lens, beta_w):
    """records carry the human pair (chosen, rejected) = (y1, y2); logp_w / logp_ref / lens are
    (values for y1, values for y2). Returns records relabelled by the weak teacher."""
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
