"""Shared setup for the stage scripts."""

from __future__ import annotations

import logging
import os

from datasets import Dataset
from omegaconf import DictConfig, OmegaConf

from utils import dist, hub
from utils.config import register_resolvers
from utils.io import read_jsonl
from utils.seed import set_seed

register_resolvers()
log = logging.getLogger("mspo")

SUCCESS = "_SUCCESS"


def setup(cfg: DictConfig, stage: str) -> None:
    dist.init()
    set_seed(cfg.seed + dist.rank())
    logging.getLogger().setLevel(logging.INFO if dist.is_main() else logging.WARNING)
    if dist.is_main():
        log.info("[%s] config:\n%s", stage, OmegaConf.to_yaml(cfg, resolve=True))


def _exists(path: str) -> bool:
    return os.path.exists(path) if path.endswith((".jsonl", ".json")) else os.path.exists(os.path.join(path, SUCCESS))


def is_done(path: str, cfg: DictConfig, also: tuple[str, ...] = ()) -> bool:
    """True if a stage output exists locally or was restored from the Hub (then skip).

    `also`: extra files of the same stage restored together with `path`."""
    if cfg.overwrite:
        return False
    if not _exists(path) and hub.enabled(cfg):
        if dist.is_main():
            restored = hub.restore_file(cfg, path) if path.endswith((".jsonl", ".json")) else hub.restore_folder(cfg, path)
            if restored:
                for f in also:
                    hub.restore_file(cfg, f)
        dist.barrier()
    if _exists(path):
        if hub.enabled(cfg) and dist.is_main() and not hub.is_pushed(cfg, path):
            log.info("%s is missing on the Hub; pushing it now.", path)
            if path.endswith((".jsonl", ".json")):
                hub.push_files(cfg, [path, *also], "late push")
            else:
                hub.push_folder(cfg, path, "late push")
        dist.barrier()
        log.info("Output %s exists, skipping (set overwrite=true to recompute).", path)
        return True
    return False


def resume_checkpoint(cfg: DictConfig, out_dir: str) -> str | None:
    """Latest mid-training checkpoint (local, or the Hub's last-checkpoint/) to resume from."""
    from transformers.trainer_utils import get_last_checkpoint

    if dist.is_main():
        hub.restore_checkpoint(cfg, out_dir)
    dist.barrier()
    local = get_last_checkpoint(out_dir) if os.path.isdir(out_dir) else None
    hub_ckpt = os.path.join(out_dir, "last-checkpoint")
    ckpt = local or (hub_ckpt if os.path.isdir(hub_ckpt) else None)
    if ckpt:
        log.info("Resuming training from %s", ckpt)
    return ckpt


def mark_done(path: str) -> None:
    if dist.is_main():
        os.makedirs(path, exist_ok=True)
        open(os.path.join(path, SUCCESS), "w").close()


def require(path: str, hint: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run `{hint}` first.")
    return path


def processed_file(cfg, split: str) -> str:
    return require(os.path.join(cfg.paths.processed_dir, f"{split}.jsonl"), "main.py stage=prepare_data")


def annotated_file(cfg) -> str:
    return require(os.path.join(cfg.paths.annotated_dir, "unlabeled.jsonl"), "main.py stage=annotate")


def preference_records(cfg) -> list[dict]:
    """D_unlabeled with chosen/rejected given by the method's label source (human or weak teacher)."""
    if cfg.method.label_source == "human":
        return read_jsonl(processed_file(cfg, "unlabeled"))
    return read_jsonl(annotated_file(cfg))


def validation_set(cfg, columns=("prompt", "chosen", "rejected")) -> Dataset:
    """Held-out 1% of the unlabeled split (human labels), the eval set for eval_loss / best model."""
    return Dataset.from_list([{k: r[k] for k in columns} for r in read_jsonl(processed_file(cfg, "validation"))])


def run_name(cfg, stage: str) -> str:
    return f"{cfg.dataset.name}-{stage}"
