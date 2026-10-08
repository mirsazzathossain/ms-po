"""Pipeline stages; each module exposes run(cfg)."""

STAGES = (
    "preflight",
    "prepare_data",
    "train_weak",
    "annotate",
    "train_strong_sft",
    "compute_ms_weights",
    "train_strong_po",
    "evaluate",
    "collect_results",
)
