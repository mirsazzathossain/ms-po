#!/usr/bin/env bash
# Table 2: student sizes with DPO. Options: MODELS, DATASETS, METHODS
set -euo pipefail
HERE="$(dirname "$0")"
declare -A STRONG=(
  [opt]="facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b"
  [qwen2_5]="Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B"
)
for model in ${MODELS:-opt qwen2_5}; do
  for strong in ${STRONG[$model]}; do
    for dataset in ${DATASETS:-hh_rlhf tldr ufb}; do
      DATASET=${dataset} MODEL=${model} LOSSES=dpo \
        bash "${HERE}/run_pipeline.sh" model.strong.name="${strong}" "$@"
    done
  done
done
