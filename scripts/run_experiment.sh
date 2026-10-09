#!/usr/bin/env bash
# One experiment (one cell of a table): only the stages it needs. Re-run to resume.
#   METHOD=ms_po LOSS=dpo bash scripts/run_experiment.sh [hydra overrides]
# Options: DATASET (hh_helpful), MODEL (opt), METHOD (ms_po | cw_po | ws_po | human), LOSS (dpo | ipo | rdpo | simpo),
#          PER_DEVICE / GRAD_ACCUM (strong SFT + PO batch; keep PER_DEVICE x GRAD_ACCUM x NUM_GPUS = 64),
#          GRAD_CKPT (true | false), NUM_GPUS
source "$(dirname "$0")/common.sh"

DATASET=${DATASET:-hh_helpful}
MODEL=${MODEL:-opt}
METHOD=${METHOD:-ms_po}
LOSS=${LOSS:-dpo}

STRONG=$(python - dataset="${DATASET}" model="${MODEL}" "$@" <<'PY'
import sys
from hydra import compose, initialize_config_dir
from utils.config import CONFIG_PATH
with initialize_config_dir(config_dir=CONFIG_PATH, version_base="1.3"):
    print(compose("config", overrides=sys.argv[1:]).model.strong.name)
PY
)
read -r -a BATCH <<< "$(big_model_overrides "${STRONG}")"
for stage in strong_sft strong_po; do
  if [[ -n "${PER_DEVICE:-}" ]]; then
    BATCH+=(train.${stage}.per_device_train_batch_size=${PER_DEVICE} train.${stage}.per_device_eval_batch_size=${PER_DEVICE})
  fi
  if [[ -n "${GRAD_ACCUM:-}" ]]; then
    BATCH+=(train.${stage}.gradient_accumulation_steps=${GRAD_ACCUM})
  fi
  if [[ -n "${GRAD_CKPT:-}" ]]; then
    BATCH+=(train.${stage}.gradient_checkpointing=${GRAD_CKPT})
  fi
done
BASE=(dataset="${DATASET}" model="${MODEL}" "${BATCH[@]}" "$@")
RUN=(method="${METHOD}" loss="${LOSS}")
echo "=== ${DATASET} | ${MODEL} (${STRONG}) | ${METHOD} | ${LOSS} | ${BATCH[*]:-paper batch settings}"

python main.py stage=preflight "${BASE[@]}"
python main.py stage=prepare_data "${BASE[@]}"
if [[ "${METHOD}" != "human" ]]; then
  launch train_weak "${BASE[@]}"
  launch annotate "${BASE[@]}"
fi
launch train_strong_sft "${BASE[@]}" "${RUN[@]}"
if [[ "${METHOD}" == "ms_po" ]]; then
  launch compute_ms_weights "${BASE[@]}"
fi
launch train_strong_po "${BASE[@]}" "${RUN[@]}"
launch evaluate "${BASE[@]}" "${RUN[@]}"
python main.py stage=collect_results "${BASE[@]}"
