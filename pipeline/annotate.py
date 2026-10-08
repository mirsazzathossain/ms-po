"""Stage 2: the weak teacher pseudo-labels D_unlabeled (Eq. 2, 4) and scores C_weak (Eq. 5).

    torchrun --nproc_per_node=N main.py stage=annotate dataset=hh_rlhf model=opt

Writes data/annotated/<dataset>/<weak>/unlabeled.jsonl (+ stats.json).
"""

from __future__ import annotations

import logging
import os

import numpy as np
from omegaconf import DictConfig

from dataset.weak_labels import annotate_pairs
from models import load_causal_lm, load_tokenizer, score_pairs
from utils import dist
from utils.common import is_done, processed_file, require, run_name, setup
from utils.io import gather_shards, read_jsonl, write_json
from utils.logging import finish_wandb, setup_wandb, wandb_log

log = logging.getLogger("mspo")


def run(cfg: DictConfig) -> None:
    setup(cfg, "annotate")
    out_file = os.path.join(cfg.paths.annotated_dir, "unlabeled.jsonl")
    if is_done(out_file, cfg):
        return
    require(os.path.join(cfg.paths.weak_po_dir, "_SUCCESS"), "main.py stage=train_weak")

    records = read_jsonl(processed_file(cfg, "unlabeled"))
    idx = dist.shard_indices(len(records))
    local = [records[i] for i in idx]

    device = dist.device()
    tok = load_tokenizer(cfg.paths.weak_po_dir)
    policy = load_causal_lm(cfg.paths.weak_po_dir, cfg.model_dtype, device).eval()  # pi_w
    ref = load_causal_lm(cfg.paths.weak_sft_dir, cfg.model_dtype, device).eval()    # pi_ref,w

    prompts = [r["prompt"] for r in local]
    args = (cfg.infer_batch_size, cfg.dataset.max_length, device, False)
    ref_c, w_c, len_c = score_pairs(policy, ref, tok, prompts, [r["chosen"] for r in local], *args)
    ref_r, w_r, len_r = score_pairs(policy, ref, tok, prompts, [r["rejected"] for r in local], *args)

    annotated = annotate_pairs(local, (w_c, w_r), (ref_c, ref_r), (len_c, len_r), beta_w=cfg.train.weak_po.beta)
    for i, a in zip(idx, annotated):
        a["_idx"] = i
    merged = gather_shards(out_file, annotated)

    if dist.is_main():
        c = np.array([m["c_weak"] for m in merged])
        stats = {
            "annotate/n": len(merged),
            "annotate/weak_label_accuracy": float(np.mean([m["weak_agrees_human"] for m in merged])),
            "annotate/c_weak_mean": float(c.mean()),
            "annotate/c_weak_std": float(c.std()),
        }
        write_json(os.path.join(cfg.paths.annotated_dir, "stats.json"), stats)
        log.info("Annotation stats: %s", stats)
        setup_wandb(cfg, run_name(cfg, f"annotate-{cfg.model.weak.short}"), "annotate")
        wandb_log(stats)
        finish_wandb()
    dist.cleanup()
