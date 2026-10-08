#!/usr/bin/env bash
# Small end-to-end test of every runner script with tiny random GPT-2 models and 48 samples per
# split. Checks the plumbing only (all datasets, stages, methods, losses, scripts); the GRA numbers
# are meaningless. Everything goes to .smoke/ so real data and checkpoints are untouched.
#
#   bash scripts/smoke_test.sh                          # W&B and Hub off
#   bash scripts/smoke_test.sh logger=wandb hub.push=true
#
# Covers: run_pipeline.sh (all 4 methods, DPO), table1.sh (TL;DR + UFB loaders, rDPO),
# table2.sh (student-size loop), table3.sh (SimPO), ablation.sh (C_MS variants, IPO),
# and collect_results.
source "$(dirname "$0")/common.sh"
export MSPO_ROOT="${ROOT_DIR}/.smoke"

# Tiny random GPT-2 weak/strong policies + reward model sharing the GPT-2 tokenizer.
python - <<'PY'
import os
from transformers import AutoTokenizer, GPT2Config, GPT2ForSequenceClassification, GPT2LMHeadModel
root = os.path.join(os.environ["MSPO_ROOT"], "tiny_models")
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
for name, n_embd, cls in (("weak", 32, GPT2LMHeadModel), ("strong", 64, GPT2LMHeadModel),
                          ("reward", 32, GPT2ForSequenceClassification)):
    out = os.path.join(root, name)
    if os.path.exists(out):
        continue
    # 2048 positions: TL;DR prompts can exceed 1024 tokens before generation.
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=n_embd, n_positions=2048, num_labels=1,
                     pad_token_id=tok.eos_token_id)
    cls(cfg).save_pretrained(out)
    tok.save_pretrained(out)
PY

TINY="${MSPO_ROOT}/tiny_models"
SMALL=(
  model.weak.name="${TINY}/weak" model.strong.name="${TINY}/strong" "model.lora_target_modules=[c_attn]"
  logger=none debug_max_samples=48 infer_batch_size=8
  eval.num_samples=8 eval.max_new_tokens=16 eval.generation_batch_size=8 eval.reward_batch_size=8
  dataset.reward_model=deberta eval.reward_models.deberta="${TINY}/reward"
  train.weak_sft.num_train_epochs=1 train.weak_po.num_train_epochs=1
  train.strong_sft.num_train_epochs=1 train.strong_po.num_train_epochs=1
  "$@"
)

echo "=== 1/5 run_pipeline.sh: HH-Helpful, all methods, DPO"
DATASET=hh_helpful MODEL=opt LOSSES=dpo bash scripts/run_pipeline.sh "${SMALL[@]}"

echo "=== 2/5 table1.sh: TL;DR + UFB loaders, rDPO"
MODELS=opt DATASETS="tldr ufb" LOSSES=rdpo METHODS="human ms_po" bash scripts/table1.sh "${SMALL[@]}"

echo "=== 3/5 table2.sh: student-size loop (sizes collapse to the tiny student)"
MODELS=opt DATASETS=hh_helpful METHODS="cw_po" bash scripts/table2.sh "${SMALL[@]}"

echo "=== 4/5 table3.sh: SimPO"
MODELS=opt DATASETS=hh_helpful METHODS="ms_po" bash scripts/table3.sh "${SMALL[@]}"

echo "=== 5/5 ablation.sh: C_MS variants with IPO"
DATASET=hh_helpful MODEL=opt LOSS=ipo VARIANTS="direct weak marginal bounded" GAMMAS="1.0 0.5" \
  bash scripts/ablation.sh "${SMALL[@]}"

python main.py stage=collect_results "${SMALL[@]}"
n=$(ls "${MSPO_ROOT}"/outputs/*/*/results/*.json | wc -l)
echo "Smoke test passed: ${n} evaluated runs."
