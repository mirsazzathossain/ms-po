#!/usr/bin/env bash
# Table 1: 3 model pairs x {HH-RLHF, TL;DR, UFB} x {DPO, IPO, rDPO} x 4 methods.
# Options: MODELS, DATASETS, LOSSES, METHODS
set -euo pipefail
HERE="$(dirname "$0")"
for model in ${MODELS:-opt qwen2_5 qwen3}; do
  for dataset in ${DATASETS:-hh_rlhf tldr ufb}; do
    DATASET=${dataset} MODEL=${model} LOSSES="${LOSSES:-dpo ipo rdpo}" \
      bash "${HERE}/run_pipeline.sh" "$@"
  done
done
