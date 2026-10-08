#!/usr/bin/env bash
# Appendix B: C_MS weighting variants (+ optional gamma sweep) for MS-PO.
# Requires run_pipeline.sh to have finished for the chosen dataset/model.
#   DATASET=hh_rlhf MODEL=opt VARIANTS="direct weak marginal bounded" GAMMAS="0.5 1.0 2.0" bash scripts/ablation.sh
source "$(dirname "$0")/common.sh"
DATASET=${DATASET:-hh_rlhf}; MODEL=${MODEL:-opt}
BASE=(dataset="${DATASET}" model="${MODEL}" method=ms_po loss="${LOSS:-dpo}" "$@")
for variant in ${VARIANTS:-direct weak marginal bounded}; do
  for gamma in ${GAMMAS:-1.0}; do
    O=(ms.variant="${variant}" ms.gamma="${gamma}")
    launch train_strong_po "${BASE[@]}" "${O[@]}"
    launch evaluate "${BASE[@]}" "${O[@]}"
  done
done
python main.py stage=collect_results "${BASE[@]}"
