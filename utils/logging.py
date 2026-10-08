"""W&B: one run per stage, grouped by experiment, flat `run.*` config for filtering;
final numbers under `results/*`."""

from __future__ import annotations

import logging
import os

from omegaconf import DictConfig

from utils import dist
from utils.config import to_container

log = logging.getLogger("mspo")

_METHOD_STAGES = ("strong_po", "evaluate")
_LABEL_STAGES = ("strong_sft", "strong_po", "evaluate")


def run_metadata(cfg: DictConfig, job_type: str) -> dict:
    """Flat, human-readable description of a run (logged as config `run.*`)."""
    if job_type == "results":  # summary over all experiments
        return {"stage": job_type}
    meta = {
        "stage": job_type,
        "dataset": cfg.dataset.name,
        "family": cfg.model.family,
        "weak_model": cfg.model.weak.name,
        "strong_model": cfg.model.strong.name,
        "pair": f"{cfg.model.weak.short} -> {cfg.model.strong.short}",
        "seed": cfg.seed,
    }
    if job_type in _LABEL_STAGES:
        meta["label_source"] = cfg.method.label_source
    if job_type in _METHOD_STAGES:
        meta["method"] = cfg.method.name
        meta["loss"] = cfg.loss.name
        meta["experiment"] = f"{cfg.method.name}_{cfg.loss.name}{cfg.method.run_suffix}"
        if cfg.method.weight == "ms":
            meta["ms_variant"] = cfg.ms.variant
            meta["ms_gamma"] = cfg.ms.gamma
    if job_type == "ms_weights":
        meta["ms_gamma"] = cfg.ms.gamma
    return meta


def setup_wandb(cfg: DictConfig, run_name: str, job_type: str, group: str | None = None):
    """Start a W&B run on rank 0 (the HF Trainer reuses it). Returns the run or None."""
    if not cfg.logger.enabled:
        os.environ["WANDB_DISABLED"] = "true"
        return None
    os.environ.setdefault("WANDB_PROJECT", cfg.logger.project)
    if not dist.is_main():
        return None
    import wandb

    meta = run_metadata(cfg, job_type)
    tags = [job_type] + [meta[k] for k in ("dataset", "family", "label_source", "method", "loss", "ms_variant") if k in meta]
    config = to_container(cfg)
    config["run"] = meta
    return wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        group=group or cfg.logger.group,
        tags=sorted(set(map(str, tags))),
        name=run_name,
        job_type=job_type,
        config=config,
        reinit=True,
    )


def report_to(cfg: DictConfig) -> list[str]:
    return ["wandb"] if cfg.logger.enabled else []


def _run():
    if not dist.is_main():
        return None
    try:
        import wandb
    except ImportError:
        return None
    return wandb.run


def wandb_log(metrics: dict, summary: bool = True) -> None:
    run = _run()
    if run is None:
        return
    run.log(metrics)
    if summary:
        for k, v in metrics.items():
            run.summary[k] = v


def log_table(key: str, columns: list[str], rows: list[list]) -> None:
    run = _run()
    if run is None:
        return
    import wandb

    run.log({key: wandb.Table(columns=columns, data=rows)})


def log_artifact(name: str, artifact_type: str, files: list[str], metadata: dict | None = None) -> None:
    """Upload result files (JSON / JSONL / CSV / Markdown) as a versioned W&B artifact."""
    run = _run()
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
    for f in files:
        if os.path.exists(f):
            artifact.add_file(f)
    run.log_artifact(artifact)


def finish_wandb() -> None:
    run = _run()
    if run is not None:
        run.finish()
