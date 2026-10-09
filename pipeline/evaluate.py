"""Gold Reward Accuracy of the aligned model vs. its SFT model (Eq. 13).  main.py stage=evaluate"""

from __future__ import annotations

import gc
import logging
import os
import random

import torch
from omegaconf import DictConfig

from utils.evaluation import GoldRewardModel, generate_responses, gold_reward_accuracy
from models import load_merged, load_tokenizer
from models.loading import DTYPES
from utils import dist, hub
from utils.common import is_done, processed_file, require, run_name, setup
from utils.io import gather_shards, read_jsonl, write_json
from utils.logging import finish_wandb, log_artifact, log_table, setup_wandb, wandb_log

log = logging.getLogger("mspo")


def _free(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cached(cfg, path: str) -> bool:
    """Local file, or restored from the Hub artifacts repo."""
    if not os.path.exists(path) and dist.is_main():
        hub.restore_file(cfg, path)
    dist.barrier()
    return os.path.exists(path)


def generations(cfg, model_dir: str, out_file: str, prompts: list[str]) -> list[dict]:
    """Sample (or load cached) responses from `model_dir` for all prompts, sharded over ranks."""
    if not cfg.overwrite and _cached(cfg, out_file):
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
    hub.push_files(cfg, [out_file], f"generations {os.path.basename(out_file)}")
    return read_jsonl(out_file)


def scores(cfg, gen_file: str, rows: list[dict], rm_holder: dict) -> list[float]:
    """Gold-reward scores (cached next to the generations)."""
    kind = cfg.dataset.reward_model
    out_file = gen_file.replace(".jsonl", f".reward_{kind}.jsonl")
    if not cfg.overwrite and _cached(cfg, out_file):
        return [r["reward"] for r in read_jsonl(out_file)]
    if "rm" not in rm_holder:
        rm_holder["rm"] = GoldRewardModel(
            kind, cfg.eval.reward_models[kind], DTYPES[cfg.model_dtype], dist.device()
        )
    idx = dist.shard_indices(len(rows))
    local = [rows[i] for i in idx]
    vals = rm_holder["rm"].score([r["prompt"] for r in local], [r["response"] for r in local], cfg.eval.reward_batch_size)
    gather_shards(out_file, [{"_idx": i, "reward": v} for i, v in zip(idx, vals)])
    hub.push_files(cfg, [out_file], f"rewards {os.path.basename(out_file)}")
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
    n_eval = cfg.eval.num_samples or cfg.dataset.get("eval_num_samples")
    if n_eval and n_eval < len(test):
        # same fixed-seed subset for every model / method
        keep = sorted(random.Random(cfg.seed).sample(range(len(test)), n_eval))
        test = [test[i] for i in keep]
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
        hub.push_files(cfg, [result_file], f"result {cfg.dataset.name} {cfg.model.strong.short} {tag}")
        setup_wandb(cfg, run_name(cfg, f"{cfg.model.strong.short}-{tag}-eval"), "evaluate")
        wandb_log({
            "results/gra": result["gra"],
            "results/reward_aligned_mean": result["reward_aligned_mean"],
            "results/reward_sft_mean": result["reward_sft_mean"],
            "results/n": n,
        })
        log_table(
            "results/per_prompt",
            ["prompt", "sft_response", "aligned_response", "reward_sft", "reward_aligned", "aligned_wins"],
            [[prompts[i], sft_rows[i]["response"], po_rows[i]["response"], sft_scores[i], po_scores[i],
              po_scores[i] > sft_scores[i]] for i in range(n)],
        )
        log_artifact(
            f"eval-{cfg.dataset.name}-{cfg.model.strong.short}-{tag}",
            "evaluation",
            [result_file, sft_file, po_file] + [f.replace(".jsonl", f".reward_{cfg.dataset.reward_model}.jsonl")
                                                for f in (sft_file, po_file)],
            metadata=result,
        )
        finish_wandb()
    dist.cleanup()

