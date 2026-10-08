#!/usr/bin/env bash
# Table 1: {OPT-125M->OPT-6.7B, Qwen2.5-0.5B->7B, Qwen3-0.6B->8B} x {HH-RLHF, TL;DR, UFB}
#          x {DPO, IPO, rDPO} x {Human, WS-PO, CW-PO, MS-PO}.
# Restrict with e.g. MODELS="opt" DATASETS="hh_rlhf" LOSSES="dpo".
set -euo pipefail
HERE="$(dirname "$0")"
for model in ${MODELS:-opt qwen2_5 qwen3}; do
  for dataset in ${DATASETS:-hh_rlhf tldr ufb}; do
    DATASET=${dataset} MODEL=${model} LOSSES="${LOSSES:-dpo ipo rdpo}" \
      bash "${HERE}/run_pipeline.sh" "$@"
  done
done
