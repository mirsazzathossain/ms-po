"""Stage 6: Gold Reward Accuracy (Eq. 13) of an aligned strong model against its SFT baseline.

    torchrun --nproc_per_node=N main.py stage=evaluate dataset=hh_rlhf model=opt method=ms_po loss=dpo

1. sample responses on the test prompts from the SFT model (cached per label source) and from the
   aligned model (temperature 0.95, 512 new tokens),
2. score both with the gold reward model (Skywork-Reward-V2-Llama-3.1-8B or OA DeBERTa),
3. GRA = fraction of prompts where R(x, y_aligned) > R(x, y_SFT).
Outputs live in outputs/<dataset>/<strong>/{generations,results}/.
"""

from __future__ import annotations

import gc
import logging
import os

import torch
from omegaconf import DictConfig

from utils.evaluation import GoldRewardModel, generate_responses, gold_reward_accuracy
from models import load_merged, load_tokenizer
from models.loading import DTYPES
from utils import dist
from utils.common import is_done, processed_file, require, run_name, setup
from utils.io import gather_shards, read_jsonl, write_json
from utils.logging import finish_wandb, setup_wandb, wandb_log

log = logging.getLogger("mspo")


def _free(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generations(cfg, model_dir: str, out_file: str, prompts: list[str]) -> list[dict]:
    """Sample (or load cached) responses from `model_dir` for all prompts, sharded over ranks."""
    if os.path.exists(out_file) and not cfg.overwrite:
        return read_jsonl(out_file)
    idx = dist.shard_indices(len(prompts))
    device = dist.device()
    tok = load_tokenizer(model_dir)
    model = load_merged(model_dir, cfg.model_dtype, device, pad_token_id=tok.pad_token_id)
    torch.manual_seed(cfg.seed + dist.rank())
    local_prompts = [prompts[i] for i in idx]
    responses = generate_responses(model, tok, local_prompts, cfg.eval, device)
    _free(model)
    rows = [{"_idx": i, "prompt": p, "response": r} for i, p, r in zip(idx, local_prompts, responses)]
    gather_shards(out_file, rows)
    return read_jsonl(out_file)


def scores(cfg, gen_file: str, rows: list[dict], rm_holder: dict) -> list[float]:
    """Gold-reward scores (cached next to the generations)."""
    kind = cfg.dataset.reward_model
    out_file = gen_file.replace(".jsonl", f".reward_{kind}.jsonl")
    if os.path.exists(out_file) and not cfg.overwrite:
        return [r["reward"] for r in read_jsonl(out_file)]
    if "rm" not in rm_holder:
        rm_holder["rm"] = GoldRewardModel(
            kind, cfg.eval.reward_models[kind], DTYPES[cfg.model_dtype], dist.device()
        )
    idx = dist.shard_indices(len(rows))
    local = [rows[i] for i in idx]
    vals = rm_holder["rm"].score([r["prompt"] for r in local], [r["response"] for r in local], cfg.eval.reward_batch_size)
    gather_shards(out_file, [{"_idx": i, "reward": v} for i, v in zip(idx, vals)])
    return [r["reward"] for r in read_jsonl(out_file)]


def run(cfg: DictConfig) -> None:
    setup(cfg, "evaluate")
    tag = f"{cfg.method.name}_{cfg.loss.name}{cfg.method.run_suffix}"
    result_file = os.path.join(cfg.paths.eval_dir, "results", f"{tag}.json")
    if is_done(result_file, cfg):
        return
    sft_dir = cfg.paths.strong_sft_dir
    po_dir = cfg.paths.strong_po_dir
    require(os.path.join(sft_dir, "_SUCCESS"), "main.py stage=train_strong_sft")
    require(os.path.join(po_dir, "_SUCCESS"), "main.py stage=train_strong_po")

    test = read_jsonl(processed_file(cfg, "test"))
    if cfg.eval.num_samples:
        test = test[: cfg.eval.num_samples]
    prompts = [r["prompt"] for r in test]
    gen_dir = os.path.join(cfg.paths.eval_dir, "generations")
    n = len(prompts)

    sft_file = os.path.join(gen_dir, f"sft_{cfg.method.label_source}_n{n}.jsonl")
    po_file = os.path.join(gen_dir, f"{tag}_n{n}.jsonl")
    sft_rows = generations(cfg, sft_dir, sft_file, prompts)
    po_rows = generations(cfg, po_dir, po_file, prompts)

    rm_holder: dict = {}
    sft_scores = scores(cfg, sft_file, sft_rows, rm_holder)
    po_scores = scores(cfg, po_file, po_rows, rm_holder)

    if dist.is_main():
        gra = gold_reward_accuracy(po_scores, sft_scores)
        result = {
            "dataset": cfg.dataset.name,
            "weak": cfg.model.weak.short,
            "strong": cfg.model.strong.short,
            "method": cfg.method.name,
            "loss": cfg.loss.name,
            "run_suffix": cfg.method.run_suffix,
            "reward_model": cfg.eval.reward_models[cfg.dataset.reward_model],
            "n": n,
            "gra": 100.0 * gra,
            "reward_aligned_mean": sum(po_scores) / n,
            "reward_sft_mean": sum(sft_scores) / n,
        }
        write_json(result_file, result)
        log.info("GRA = %.2f%% (%s)", 100 * gra, result)
        setup_wandb(cfg, run_name(cfg, f"{cfg.model.strong.short}-{tag}-eval"), "evaluate")
        wandb_log({"eval/gra": result["gra"], "eval/reward_aligned": result["reward_aligned_mean"],
                   "eval/reward_sft": result["reward_sft_mean"]})
        try:
            import wandb

            if wandb.run is not None:
                table = wandb.Table(columns=["prompt", "sft", "aligned", "r_sft", "r_aligned"])
                for i in range(min(50, n)):
                    table.add_data(prompts[i], sft_rows[i]["response"], po_rows[i]["response"], sft_scores[i], po_scores[i])
                wandb.log({"eval/samples": table})
        except ImportError:
            pass
        finish_wandb()
    dist.cleanup()

