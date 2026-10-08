#!/usr/bin/env bash
# All stages for one dataset / model pair. Re-run to resume.
#   DATASET=hh_rlhf MODEL=opt bash scripts/run_pipeline.sh [hydra overrides]
# Options: DATASET, MODEL, LOSSES (dpo ipo rdpo), METHODS (human ws_po cw_po ms_po)
source "$(dirname "$0")/common.sh"

DATASET=${DATASET:-hh_rlhf}
MODEL=${MODEL:-opt}
LOSSES=${LOSSES:-"dpo ipo rdpo"}
METHODS=${METHODS:-"human ws_po cw_po ms_po"}

BASE=(dataset="${DATASET}" model="${MODEL}" "$@")
STRONG=$(python - "${BASE[@]}" <<'PY'
import sys
from hydra import compose, initialize_config_dir
from utils.config import CONFIG_PATH
with initialize_config_dir(config_dir=CONFIG_PATH, version_base="1.3"):
    print(compose("config", overrides=sys.argv[1:]).model.strong.name)
PY
)
read -r -a BIG <<< "$(big_model_overrides "${STRONG}")"
echo "=== ${DATASET} | ${MODEL} | strong=${STRONG} | losses: ${LOSSES} | methods: ${METHODS}"

python main.py stage=prepare_data "${BASE[@]}"
launch train_weak "${BASE[@]}"
launch annotate "${BASE[@]}"

# SFT on human labels (Human) and on weak labels (WS-PO / CW-PO / MS-PO)
if [[ " ${METHODS} " == *" human "* ]]; then
  launch train_strong_sft "${BASE[@]}" "${BIG[@]}" method=human
fi
launch train_strong_sft "${BASE[@]}" "${BIG[@]}" method=ms_po
if [[ " ${METHODS} " == *" ms_po "* ]]; then
  launch compute_ms_weights "${BASE[@]}"
fi

for loss in ${LOSSES}; do
  for method in ${METHODS}; do
    launch train_strong_po "${BASE[@]}" "${BIG[@]}" method="${method}" loss="${loss}"
    launch evaluate "${BASE[@]}" "${BIG[@]}" method="${method}" loss="${loss}"
  done
done
python main.py stage=collect_results "${BASE[@]}"
