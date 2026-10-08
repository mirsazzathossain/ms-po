#!/usr/bin/env bash
# Table 3: reference-free SimPO for all three model pairs and datasets.
set -euo pipefail
HERE="$(dirname "$0")"
for model in ${MODELS:-opt qwen2_5 qwen3}; do
  for dataset in ${DATASETS:-hh_rlhf tldr ufb}; do
    DATASET=${dataset} MODEL=${model} LOSSES=simpo bash "${HERE}/run_pipeline.sh" "$@"
  done
done
