"""S(x, y+), S(x, y-) between weak teacher and SFT student (Eq. 8-9).  main.py stage=compute_ms_weights"""

from __future__ import annotations

import logging
import os

import torch
from omegaconf import DictConfig, OmegaConf

from models import check_shared_vocab, load_causal_lm, load_merged, load_tokenizer, score_pairs
from utils import dist, hub
from utils.common import annotated_file, is_done, require, run_name, setup
from utils.io import gather_shards, read_jsonl, write_json
from utils.logging import finish_wandb, setup_wandb, wandb_log
from utils.losses import c_align, ms_confidence

log = logging.getLogger("mspo")


def run(cfg: DictConfig) -> None:
    OmegaConf.update(cfg, "method.label_source", "weak", force_add=True)
    setup(cfg, "compute_ms_weights")
    out_file = cfg.paths.ms_weights_file
    stats_file = out_file.replace(".jsonl", "_stats.json")
    if is_done(out_file, cfg, also=(stats_file,)):
        return
    sft_dir = cfg.paths.strong_sft_dir
    require(os.path.join(sft_dir, "_SUCCESS"), "main.py stage=train_strong_sft method=ms_po")

    records = read_jsonl(annotated_file(cfg))
    idx = dist.shard_indices(len(records))
    local = [records[i] for i in idx]

    device = dist.device()
    tok = load_tokenizer(sft_dir)
    check_shared_vocab(load_tokenizer(cfg.paths.weak_po_dir), tok)
    weak_model = load_causal_lm(cfg.paths.weak_po_dir, cfg.model_dtype, device).eval()
    student_model = load_merged(sft_dir, cfg.model_dtype, device).eval()

    prompts = [r["prompt"] for r in local]
    args = (cfg.infer_batch_size, cfg.dataset.max_length, device, True)
    s_c, _, _ = score_pairs(weak_model, student_model, tok, prompts, [r["chosen"] for r in local], *args)
    s_r, _, _ = score_pairs(weak_model, student_model, tok, prompts, [r["rejected"] for r in local], *args)

    rows = [{"_idx": i, "S_chosen": a, "S_rejected": b} for i, a, b in zip(idx, s_c, s_r)]
    merged = gather_shards(out_file, rows)

    if dist.is_main():
        S_c = torch.tensor([m["S_chosen"] for m in merged])
        S_r = torch.tensor([m["S_rejected"] for m in merged])
        c_ms = ms_confidence(c_align(S_c, cfg.ms.gamma), c_align(S_r, cfg.ms.gamma), "direct")
        stats = {
            "ms/S_chosen_mean": S_c.mean().item(),
            "ms/S_rejected_mean": S_r.mean().item(),
            "ms/c_ms_mean": c_ms.mean().item(),
            "ms/c_ms_frac_negative": (c_ms < 0).float().mean().item(),
        }
        write_json(stats_file, stats)
        log.info("MS-PO stats (gamma=%s): %s", cfg.ms.gamma, stats)
        hub.push_files(cfg, [out_file, stats_file], f"ms weights {cfg.dataset.name} {cfg.model.strong.short}")
        setup_wandb(cfg, run_name(cfg, f"ms-weights-{cfg.model.strong.short}"), "ms_weights")
        wandb_log(stats)
        finish_wandb()
    dist.cleanup()
