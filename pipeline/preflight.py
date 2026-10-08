"""Pre-run checks, so a long cluster job does not fail hours in.

    python main.py stage=preflight

Checks: GPUs, free disk, W&B login, HF token, access to every dataset / weak / strong / gold-RM
repo used in the paper (gated Skywork included), and that each weak/strong pair shares a
tokenizer vocabulary (needed for the token-level KL, Eq. 9). Downloads only configs/tokenizers.
Exits non-zero if any check fails.
"""

from __future__ import annotations

import os
import shutil
import sys

import torch
from omegaconf import DictConfig, OmegaConf

from utils.config import CONFIG_PATH

FAMILIES = ("opt", "qwen2_5", "qwen3")
DATASETS = ("hh_rlhf", "tldr", "ufb")
# Students of Table 2 besides the Table 1 defaults (scripts/table2.sh).
TABLE2_STUDENTS = {
    "opt": ("facebook/opt-1.3b", "facebook/opt-2.7b"),
    "qwen2_5": ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"),
    "qwen3": (),
}


def _load(group: str, name: str) -> DictConfig:
    return OmegaConf.load(os.path.join(CONFIG_PATH, group, f"{name}.yaml"))


def run(cfg: DictConfig) -> None:
    results: list[tuple[str, bool, str]] = []

    def check(name, fn):
        try:
            detail = fn()
            results.append((name, True, str(detail or "")))
        except Exception as e:  # noqa: BLE001 - report every failure, keep checking
            results.append((name, False, f"{type(e).__name__}: {e}"[:200]))

    # --- hardware / disk
    def cuda_check():
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA device")
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return f"{len(names)} GPU(s): {', '.join(names)}"

    check("cuda", cuda_check)
    check("disk", lambda: f"{shutil.disk_usage(cfg.paths.root_dir).free / 2**30:.0f} GB free at {cfg.paths.root_dir}")

    # --- logging / hub credentials
    def wandb_check():
        if not cfg.logger.enabled:
            return "logger disabled"
        import wandb

        return f"logged in as {wandb.Api().viewer.username}, project {cfg.logger.project}"

    check("wandb", wandb_check)

    from huggingface_hub import HfApi, auth_check

    api = HfApi()
    check("hf_token", lambda: f"logged in as {api.whoami()['name']}")
    if cfg.hub.push:
        def hub_user_check():
            if not cfg.hub.user:
                raise ValueError("hub.push=true needs HF_USERNAME")
            return cfg.hub.user

        check("hub.user", hub_user_check)

    # --- repo access (configs only; gated repos raise here)
    repos = {s for students in TABLE2_STUDENTS.values() for s in students}
    for fam in FAMILIES:
        m = _load("model", fam)
        repos |= {m.weak.name, m.strong.name}
    repos |= set(cfg.eval.reward_models.values())
    for repo in sorted(repos):
        check(f"model {repo}", lambda r=repo: auth_check(r) or "accessible")
    for ds in DATASETS:
        d = _load("dataset", ds)
        check(f"dataset {d.hf_path}", lambda p=d.hf_path: auth_check(p, repo_type="dataset") or "accessible")

    # --- weak/strong tokenizers must match (Eq. 9)
    from transformers import AutoTokenizer

    for fam in FAMILIES:
        m = _load("model", fam)
        students = [m.strong.name, *TABLE2_STUDENTS[fam]]

        def vocab(fam=fam, m=m, students=students):
            w = AutoTokenizer.from_pretrained(m.weak.name).get_vocab()
            for s in students:
                if AutoTokenizer.from_pretrained(s).get_vocab() != w:
                    raise ValueError(f"{m.weak.name} and {s} vocabularies differ")
            return f"{m.weak.name} matches {', '.join(students)}"

        check(f"vocab {fam}", vocab)

    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed.")
    if failed:
        sys.exit(1)
