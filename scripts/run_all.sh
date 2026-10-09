#!/usr/bin/env bash
# All results: preflight -> Table 1 -> Table 2 -> Table 3. Re-run to resume.
#   NUM_GPUS=8 bash scripts/run_all.sh [hydra overrides]
# Options: RUN_ABLATION=1 (Appendix B variants), MODELS, DATASETS
set -euo pipefail
HERE="$(dirname "$0")"
source "${HERE}/common.sh"

python main.py stage=preflight "$@"

bash "${HERE}/table1.sh" "$@"
bash "${HERE}/table2.sh" "$@"
bash "${HERE}/table3.sh" "$@"

if [[ "${RUN_ABLATION:-0}" == "1" ]]; then
  for model in ${MODELS:-opt qwen2_5 qwen3}; do
    for dataset in ${DATASETS:-hh_helpful tldr ufb}; do
      DATASET=${dataset} MODEL=${model} bash "${HERE}/ablation.sh" "$@"
    done
  done
fi

python main.py stage=collect_results "$@"
