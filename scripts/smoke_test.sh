#!/usr/bin/env bash
# End-to-end CPU/GPU smoke test with tiny random models and a few hundred HH-RLHF pairs.
# Checks the plumbing only; numbers are meaningless.
source "$(dirname "$0")/common.sh"
export MSPO_ROOT="${ROOT_DIR}/.smoke"
NUM_GPUS=${NUM_GPUS:-0}

SMALL=(
  dataset=hh_helpful model=tiny logger=none precision=fp32 model_dtype=fp32 infer_batch_size=8
  eval.num_samples=8 eval.max_new_tokens=16 eval.reward_batch_size=4
  dataset.reward_model=deberta eval.reward_models.deberta=${MSPO_ROOT}/tiny_models/reward
  train.weak_sft.num_train_epochs=1 train.weak_po.num_train_epochs=1
  train.strong_sft.num_train_epochs=1 train.strong_po.num_train_epochs=1
  train.strong_sft.optim=adamw_torch train.strong_po.optim=adamw_torch
  train.weak_sft.per_device_train_batch_size=8 train.weak_po.per_device_train_batch_size=8
  train.strong_sft.per_device_train_batch_size=4 train.strong_po.per_device_train_batch_size=4
  train.strong_sft.gradient_accumulation_steps=1 train.strong_po.gradient_accumulation_steps=1
  train.strong_sft.warmup_steps=0 train.strong_po.warmup_steps=0
)
SMOKE_N=${SMOKE_N:-200}

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
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=n_embd, n_positions=1024, num_labels=1,
                     pad_token_id=tok.eos_token_id)
    cls(cfg).save_pretrained(out)
    tok.save_pretrained(out)
PY

python main.py stage=prepare_data "${SMALL[@]}"
# Shrink splits for speed.
for f in labeled unlabeled validation test; do
  p="${MSPO_ROOT}/data/processed/hh_helpful/${f}.jsonl"
  head -n "${SMOKE_N}" "$p" > "$p.tmp" && mv "$p.tmp" "$p"
done

launch train_weak "${SMALL[@]}"
launch annotate "${SMALL[@]}"
launch train_strong_sft "${SMALL[@]}" method=human
launch train_strong_sft "${SMALL[@]}" method=ms_po
launch compute_ms_weights "${SMALL[@]}"
for method in human ws_po cw_po ms_po; do
  launch train_strong_po "${SMALL[@]}" method=${method} loss=dpo
  launch evaluate "${SMALL[@]}" method=${method} loss=dpo
done
for loss in ipo rdpo simpo; do
  launch train_strong_po "${SMALL[@]}" method=ms_po loss=${loss}
done
launch train_strong_po "${SMALL[@]}" method=ms_po loss=dpo ms.variant=weak
python main.py stage=collect_results "${SMALL[@]}"
echo "Smoke test passed."
