#!/usr/bin/env bash
# Table 3: SimPO. Options: MODELS, DATASETS, METHODS
set -euo pipefail
HERE="$(dirname "$0")"
for model in ${MODELS:-opt qwen2_5 qwen3}; do
  for dataset in ${DATASETS:-hh_helpful tldr ufb}; do
    DATASET=${dataset} MODEL=${model} LOSSES=simpo bash "${HERE}/run_pipeline.sh" "$@"
  done
done
