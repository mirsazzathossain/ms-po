#!/usr/bin/env bash
# Shared helpers for all run scripts.
#   NUM_GPUS : GPUs per node (default: all visible GPUs). >1 launches DDP via torchrun.
#   MASTER_PORT : torchrun rendezvous port (default 29500).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export MSPO_ROOT="${MSPO_ROOT:-${ROOT_DIR}}"
export TOKENIZERS_PARALLELISM=false
[[ -f "${ROOT_DIR}/.env" ]] && set -a && source "${ROOT_DIR}/.env" && set +a

if [[ -z "${NUM_GPUS:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
  else
    NUM_GPUS=0
  fi
fi

# launch <stage> [hydra overrides...]  -> main.py stage=<stage> on all GPUs (torchrun if NUM_GPUS > 1)
launch() {
  local stage=$1; shift
  echo ">>> [$(date '+%F %T')] stage=${stage} $* (GPUs: ${NUM_GPUS})"
  if (( NUM_GPUS > 1 )); then
    torchrun --standalone --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT:-29500}" \
      main.py stage="${stage}" "$@"
  else
    python main.py stage="${stage}" "$@"
  fi
}

# Paper (App. C.3): per-device batch 16, reduced to 4 for models exceeding 7B parameters
# (Qwen2.5-7B = 7.6B, Qwen3-8B). Usage: big_model_overrides <hf model id> -> hydra overrides
big_model_overrides() {
  case "$1" in
    *-7B|*-8B)
      echo "train.strong_sft.per_device_train_batch_size=4 train.strong_sft.per_device_eval_batch_size=4 \
train.strong_po.per_device_train_batch_size=4 train.strong_po.per_device_eval_batch_size=4" ;;
    *) echo "" ;;
  esac
}
