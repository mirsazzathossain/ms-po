#!/usr/bin/env bash
# Reproduce every result of the paper, in dependency order:
#   preflight -> Table 1 (DPO/IPO/rDPO) -> Table 2 (student sizes) -> Table 3 (SimPO)
# Table 2 and 3 reuse Table 1's weak teachers, annotations and SFT students.
# Finished stages are skipped, so the script can simply be re-run after an interruption.
#
#   NUM_GPUS=8 bash scripts/run_all.sh [extra hydra overrides]
#   RUN_ABLATION=1 ...   also runs the Appendix-B C_MS variants (ablation.sh) per dataset / model
set -euo pipefail
HERE="$(dirname "$0")"
source "${HERE}/common.sh"

python main.py stage=preflight "$@"

bash "${HERE}/table1.sh" "$@"
bash "${HERE}/table2.sh" "$@"
bash "${HERE}/table3.sh" "$@"

if [[ "${RUN_ABLATION:-0}" == "1" ]]; then
  for model in ${MODELS:-opt qwen2_5 qwen3}; do
    for dataset in ${DATASETS:-hh_rlhf tldr ufb}; do
      DATASET=${dataset} MODEL=${model} bash "${HERE}/ablation.sh" "$@"
    done
  done
fi

python main.py stage=collect_results "$@"
