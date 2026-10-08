"""Pipeline stages. Each module exposes `run(cfg)`; `main.py` dispatches on `stage=<name>`.

Order: prepare_data -> train_weak -> annotate -> train_strong_sft -> compute_ms_weights
       -> train_strong_po -> evaluate -> collect_results
"""

STAGES = (
    "prepare_data",
    "train_weak",
    "annotate",
    "train_strong_sft",
    "compute_ms_weights",
    "train_strong_po",
    "evaluate",
    "collect_results",
)
