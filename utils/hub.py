"""Hugging Face Hub sync (hub.push=true): push every stage output and restore it on a fresh machine.

* checkpoint folders -> private model repo  <user>/<prefix>-<path under checkpoints/ with / -> ->
  (plus `last-checkpoint/` pushed by the Trainer during training, for resuming)
* data/ and outputs/ files -> private dataset repo <user>/<prefix>-artifacts, same relative paths
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

from omegaconf import DictConfig
from transformers import TrainerCallback

from utils import dist

log = logging.getLogger("mspo")

SUCCESS = "_SUCCESS"
IGNORE = ["checkpoint-*", "last-checkpoint/*", "*.pt", "runs/*"]


def enabled(cfg: DictConfig) -> bool:
    if not cfg.hub.push:
        return False
    if not cfg.hub.user:
        raise ValueError("hub.push=true requires hub.user (or the HF_USERNAME env var).")
    return True


def model_repo(cfg: DictConfig, folder: str) -> str:
    rel = os.path.relpath(folder, cfg.paths.checkpoints_dir).replace(os.sep, "-")
    return f"{cfg.hub.user}/{cfg.hub.prefix}-{rel}"[:96]


def artifacts_repo(cfg: DictConfig) -> str:
    return f"{cfg.hub.user}/{cfg.hub.prefix}-artifacts"


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def _retry(fn, what: str, attempts: int = 8):
    """Retry Hub calls on rate limits (429), server errors and network errors, ~15 min in total."""
    import requests
    from huggingface_hub.errors import HfHubHTTPError

    for i in range(attempts):
        try:
            return fn()
        except (HfHubHTTPError, requests.exceptions.RequestException) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and status < 500 and status != 429 or i == attempts - 1:
                raise
            wait = min(30 * 2**i, 300)
            log.warning("%s failed (%s); retry %d/%d in %ds", what, status or type(e).__name__, i + 1, attempts - 1, wait)
            time.sleep(wait)


def is_pushed(cfg: DictConfig, path: str) -> bool:
    """Whether a finished folder (its _SUCCESS) or a data/outputs file is already on the Hub."""
    from huggingface_hub.utils import RepositoryNotFoundError

    api = _api()
    try:
        if path.endswith((".jsonl", ".json")):
            return api.file_exists(artifacts_repo(cfg), os.path.relpath(path, cfg.paths.root_dir), repo_type="dataset")
        return api.file_exists(model_repo(cfg, path), SUCCESS)
    except RepositoryNotFoundError:
        return False


def push_folder(cfg: DictConfig, folder: str, commit_message: str) -> str | None:
    """Upload a finished checkpoint folder (rank 0)."""
    if not enabled(cfg) or not dist.is_main():
        return None
    api, repo_id = _api(), model_repo(cfg, folder)
    _retry(lambda: api.create_repo(repo_id, private=cfg.hub.private, exist_ok=True), f"create {repo_id}")

    def upload(ignore):
        return _retry(lambda: api.upload_folder(repo_id=repo_id, folder_path=folder, commit_message=commit_message,
                                                ignore_patterns=ignore), f"upload {repo_id}")

    try:
        upload(IGNORE)
    except ValueError as e:
        # Hub rejects PEFT's card when base_model is a local path
        if "metadata in README.md" not in str(e):
            raise
        log.warning("Hub rejected the README.md model card (%s); uploading without it.", e)
        upload(IGNORE + ["README.md"])
    if any(f.startswith("last-checkpoint/") for f in _retry(lambda: api.list_repo_files(repo_id), "list")):
        _retry(lambda: api.delete_folder("last-checkpoint", repo_id=repo_id, commit_message="training finished"),
               f"cleanup {repo_id}")
    log.info("Pushed %s to https://huggingface.co/%s", folder, repo_id)
    return repo_id


def push_files(cfg: DictConfig, paths: list[str], commit_message: str) -> None:
    """Upload data/ or outputs/ files to the artifacts dataset repo (rank 0)."""
    if not enabled(cfg) or not dist.is_main():
        return
    from huggingface_hub import CommitOperationAdd

    api, repo_id = _api(), artifacts_repo(cfg)
    _retry(lambda: api.create_repo(repo_id, repo_type="dataset", private=cfg.hub.private, exist_ok=True),
           f"create {repo_id}")
    ops = [
        CommitOperationAdd(path_in_repo=os.path.relpath(p, cfg.paths.root_dir), path_or_fileobj=p)
        for p in paths
        if os.path.exists(p)
    ]
    if ops:
        _retry(lambda: api.create_commit(repo_id=repo_id, repo_type="dataset", operations=ops,
                                         commit_message=commit_message), f"upload to {repo_id}")
        log.info("Pushed %d file(s) to https://huggingface.co/datasets/%s", len(ops), repo_id)


def restore_file(cfg: DictConfig, path: str) -> bool:
    """Download one data/ or outputs/ file from the artifacts repo if it exists there."""
    if not enabled(cfg):
        return False
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    try:
        hf_hub_download(artifacts_repo(cfg), os.path.relpath(path, cfg.paths.root_dir), repo_type="dataset",
                        local_dir=cfg.paths.root_dir)
    except (EntryNotFoundError, RepositoryNotFoundError):
        return False
    log.info("Restored %s from the Hub", path)
    return True


def restore_pattern(cfg: DictConfig, pattern: str) -> None:
    """Download every artifacts-repo file matching `pattern` (e.g. all result JSONs)."""
    if not enabled(cfg):
        return
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError

    try:
        snapshot_download(artifacts_repo(cfg), repo_type="dataset", allow_patterns=[pattern], local_dir=cfg.paths.root_dir)
    except RepositoryNotFoundError:
        pass


def restore_folder(cfg: DictConfig, folder: str) -> bool:
    """Download a finished checkpoint folder (one whose repo holds _SUCCESS)."""
    if not enabled(cfg):
        return False
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError

    repo_id = model_repo(cfg, folder)
    try:
        if SUCCESS not in _api().list_repo_files(repo_id):
            return False
    except RepositoryNotFoundError:
        return False
    snapshot_download(repo_id, local_dir=folder, ignore_patterns=IGNORE)
    log.info("Restored %s from https://huggingface.co/%s", folder, repo_id)
    return True


def restore_checkpoint(cfg: DictConfig, folder: str) -> str | None:
    """Latest mid-training checkpoint: local checkpoint-* first, else the Hub's last-checkpoint/."""
    from transformers.trainer_utils import get_last_checkpoint

    local = get_last_checkpoint(folder) if os.path.isdir(folder) else None
    if local or not enabled(cfg):
        return local
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError

    repo_id = model_repo(cfg, folder)
    try:
        if not any(f.startswith("last-checkpoint/") for f in _api().list_repo_files(repo_id)):
            return None
    except RepositoryNotFoundError:
        return None
    snapshot_download(repo_id, local_dir=folder, allow_patterns=["last-checkpoint/*"])
    log.info("Resuming from https://huggingface.co/%s (last-checkpoint)", repo_id)
    return os.path.join(folder, "last-checkpoint")


class CheckpointUploader(TrainerCallback):
    """After every save, upload checkpoint-<step> to `last-checkpoint/` of the model repo in a
    background thread (one at a time, with retries; skips the model card; logs failures)."""

    def __init__(self, cfg: DictConfig, output_dir: str):
        self.cfg = cfg
        self.repo_id = model_repo(cfg, output_dir)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.job = None

    def _wait(self):
        if self.job is None:
            return
        try:
            self.job.result()
        except Exception as e:  # noqa: BLE001 - never kill training over an upload
            log.error("Checkpoint upload to %s failed: %s", self.repo_id, e)
        self.job = None

    def _upload(self, folder: str, step: int):
        api = _api()
        _retry(lambda: api.create_repo(self.repo_id, private=self.cfg.hub.private, exist_ok=True), "create repo")
        _retry(lambda: api.upload_folder(
            repo_id=self.repo_id, folder_path=folder, path_in_repo="last-checkpoint",
            ignore_patterns=["README.md"], delete_patterns="*", commit_message=f"checkpoint step {step}",
        ), f"checkpoint step {step}")
        log.info("Uploaded checkpoint step %d to https://huggingface.co/%s (last-checkpoint)", step, self.repo_id)

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self._wait()
        folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(folder):
            self.job = self.pool.submit(self._upload, folder, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._wait()


def callbacks(cfg: DictConfig, output_dir: str) -> list:
    return [CheckpointUploader(cfg, output_dir)] if enabled(cfg) else []
