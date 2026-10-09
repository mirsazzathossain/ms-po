#!/usr/bin/env bash
# Appendix B: C_MS variants for MS-PO (needs run_pipeline.sh done for DATASET/MODEL).
# Options: DATASET, MODEL, LOSS (dpo), VARIANTS (direct weak marginal bounded), GAMMAS (1.0)
source "$(dirname "$0")/common.sh"
DATASET=${DATASET:-hh_helpful}; MODEL=${MODEL:-opt}
BASE=(dataset="${DATASET}" model="${MODEL}" method=ms_po loss="${LOSS:-dpo}" "$@")
for variant in ${VARIANTS:-direct weak marginal bounded}; do
  for gamma in ${GAMMAS:-1.0}; do
    O=(ms.variant="${variant}" ms.gamma="${gamma}")
    launch train_strong_po "${BASE[@]}" "${O[@]}"
    launch evaluate "${BASE[@]}" "${O[@]}"
  done
done
python main.py stage=collect_results "${BASE[@]}"
