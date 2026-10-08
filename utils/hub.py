"""Upload checkpoints to the Hugging Face Hub."""

from __future__ import annotations

import logging

from omegaconf import DictConfig

from utils import dist

log = logging.getLogger("mspo")


def push_folder(cfg: DictConfig, folder: str, name: str, commit_message: str) -> str | None:
    """Upload `folder` to <hub.user>/<hub.prefix>-<name> (rank 0 only). Requires HF_TOKEN."""
    if not cfg.hub.push or not dist.is_main():
        return None
    if not cfg.hub.user:
        raise ValueError("hub.push=true requires hub.user (or the HF_USERNAME env var).")
    from huggingface_hub import HfApi

    repo_id = f"{cfg.hub.user}/{cfg.hub.prefix}-{name}"[:96]
    api = HfApi()
    api.create_repo(repo_id, private=cfg.hub.private, exist_ok=True)
    ignore = ["checkpoint-*", "*.pt", "runs/*"]
    try:
        api.upload_folder(repo_id=repo_id, folder_path=folder, commit_message=commit_message, ignore_patterns=ignore)
    except ValueError as e:
        # The model card PEFT writes has `base_model: <path>`; the Hub rejects it when the base
        # model is a local directory. Upload the weights without the card.
        if "metadata in README.md" not in str(e):
            raise
        log.warning("Hub rejected the README.md model card (%s); uploading without it.", e)
        api.upload_folder(
            repo_id=repo_id, folder_path=folder, commit_message=commit_message, ignore_patterns=ignore + ["README.md"]
        )
    log.info("Pushed %s to https://huggingface.co/%s", folder, repo_id)
    return repo_id
