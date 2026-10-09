# MS-PO: Multi-Granular Student-Aware Preference Optimization

Reference implementation of **MS-PO** for weak-to-strong alignment. A weak teacher, trained on 30%
of the human labels, pseudo-labels the remaining 70%. A stronger student is then aligned on those
labels, and each pair is weighted by how closely teacher and student agree, measured with
token-level forward KL.

The code covers every setting in the paper:

- **Methods:** Human, WS-PO, CW-PO, MS-PO
- **Losses:** DPO, IPO, rDPO, SimPO
- **Datasets:** HH-RLHF (`helpful-base`, as confirmed by the author; Harmless and the combination are also available), TL;DR, UltraFeedback-Binarized
- **Model pairs:** OPT-125M→OPT-{1.3B,2.7B,6.7B}, Qwen2.5-0.5B→Qwen2.5-{1.5B,3B,7B}, Qwen3-0.6B→Qwen3-8B
- **Metric:** Gold Reward Accuracy (GRA)

Stack: Hydra (configs), W&B (logging), the Hugging Face Hub (weights), Docker, and multi-GPU DDP through `torchrun`.

## Layout

```
ms-po/
├── main.py                 # single entry point: python main.py stage=<stage> ...
├── configs/                # Hydra configs
│   ├── config.yaml         #   root: seed, precision, MS-PO (gamma / variant)
│   ├── dataset/            #   hh_helpful (= paper "HH-RLHF"), hh_harmless, hh_rlhf (both), tldr, ufb
│   ├── model/              #   opt, qwen2_5, qwen3 (weak/strong pairs)
│   ├── method/             #   human, ws_po, cw_po, ms_po
│   ├── loss/               #   dpo, ipo, rdpo, simpo
│   ├── train/              #   weak_sft, weak_po, strong_sft, strong_po (Tables 4-6)
│   ├── eval/ logger/ hub/ paths/
├── dataset/                # dataset code: HH-RLHF / TL;DR / UFB loaders, length filters, splits, weak labels
├── models/                 # model code: loading, LoRA config, adapter lineage, batched teacher/student scoring
├── pipeline/               # one module per stage, each exposing run(cfg)
├── utils/                  # confidence scores (losses.py), TRL trainers (trainer.py), evaluation,
│                           #   W&B (logging.py), HF Hub (hub.py), distributed helpers, io
├── scripts/                # run_experiment.sh, run_pipeline.sh, table1-3.sh, run_all.sh, ablation.sh, smoke_test.sh, colab_setup.sh, common.sh
├── data/                   # datasets (generated; see data/README.md)
├── checkpoints/            # trained weights (generated)
├── outputs/                # generations, GRA results, Hydra logs (generated)
├── Dockerfile  docker-compose.yml  requirements.txt  .env.example
```

`resources/` (paper PDF + original reference code) is kept locally and is git-ignored.

## Sources

The implementation follows `resources/ms_po_ours/` (the reference code) and the paper. Where
the two disagree, hyperparameters come from the paper (Tables 4-6, App. C.3) and implementation
details come from the reference code.

| Component | Source | Code |
|---|---|---|
| HH-RLHF parsing, length filter, 30/70 split, 1% validation | `utils/data_processing_hh_rlhf.py` | `dataset/hh_rlhf.py`, `dataset/registry.py` |
| TL;DR parsing + 1024-word filter | `datasets/tl_dr_dataset.py` | `dataset/tldr.py` |
| UFB | paper Sec. 5.1 (no reference script; uses the same HH filter) | `dataset/ufb.py` |
| TRL DPOTrainer, ref_model=None + LoRA(q_proj, v_proj), DPOConfig fields | `train.py`, `utils/mange_config.py` | `utils/trainer.py`, `pipeline/train_strong_po.py` |
| Per-sample confidence × loss inside `dpo_loss` | `utils/CW_PO.py` | `utils/trainer.py::CWDPOTrainer` |
| S(x,y), C_align, C_weak (mean token prob), C_MS = 2(σ(ΔC_align) − ½) | `utils/Compute_MS_PO.py`, Eq. 5, 8-11 | `utils/losses.py`, `models/scoring.py` |
| Weak variant C_weak · C⁺_align · C⁻_align | `utils/MS_PO_1.py`, Eq. 25 | `utils/losses.py::ms_confidence` |
| Marginal / bounded / normalized variants | App. B, Eq. 26-28 | `utils/losses.py::ms_confidence` |
| Weak teacher SFT+DPO, pseudo-labels y⁺ = argmax r_w | Eq. 2-4, App. C.2 | `pipeline/train_weak.py`, `dataset/weak_labels.py` |
| Strong SFT on (x, chosen), human or weak labels | Sec. 5.1 | `pipeline/train_strong_sft.py` |
| SimPO | Table 3, via TRL CPOTrainer (`loss_type="simpo"`, `cpo_alpha=0`) | `utils/trainer.py::CWCPOTrainer` |
| GRA with Skywork-Reward-V2-Llama-3.1-8B / OA-DeBERTa, T=0.95, 512 new tokens | Sec. 5.2, App. C.3 | `utils/evaluation.py`, `pipeline/evaluate.py` |

## Pipeline

| # | Stage (`python main.py stage=...`) | Output |
|---|---|---|
| – | `preflight`: GPUs, disk, W&B / HF login, access to every model, dataset and gold RM (gated Skywork included), weak/strong vocabulary match | pass/fail report |
| 0 | `prepare_data`: parse, length filter, split 30/70, 1% of the 70% held out for validation | `data/processed/<ds>/` |
| 1 | `train_weak`: full fine-tune of the weak model, SFT then DPO on D_labeled | `checkpoints/<ds>/weak/<weak>/{sft,dpo}` |
| 2 | `annotate`: weak labels + C_weak on D_unlabeled | `data/annotated/<ds>/<weak>/unlabeled.jsonl` |
| 3 | `train_strong_sft`: LoRA SFT of the student on (x, chosen), with human or weak labels (TRL SFTTrainer) | `checkpoints/<ds>/strong/<strong>/sft_{human,weak}` |
| 4 | `compute_ms_weights`: S(x,y⁺), S(x,y⁻) between the weak DPO teacher and the SFT student | `data/annotated/<ds>/<weak>/ms_weights_<strong>.jsonl` |
| 5 | `train_strong_po`: weighted preference optimisation (LoRA; reference model = SFT) | `checkpoints/<ds>/strong/<strong>/<method>_<loss>` |
| 6 | `evaluate`: sample (T=0.95, 512 tokens) from the SFT and aligned models, then compute gold RM GRA | `outputs/<ds>/<strong>/results/*.json` |
| 7 | `collect_results`: build Table-1-style summaries (also logged to W&B) | `outputs/results.{md,csv}` |

A stage that has already finished is skipped. Pass `overwrite=true` to run it again.

## Setup

```bash
cp .env.example .env        # WANDB_API_KEY, WANDB_ENTITY, HF_TOKEN, HF_USERNAME
```

**Docker** (needs the NVIDIA Container Toolkit):

```bash
docker compose build
docker compose run --rm ms-po bash scripts/run_pipeline.sh
```

**Plain Python:** install `torch==2.11.0` for your CUDA version, then:

```bash
git clone https://github.com/mirsazzathossain/ms-po && cd ms-po
pip install -r requirements.txt
bash scripts/smoke_test.sh                                  # plumbing check with tiny models (W&B off)
bash scripts/smoke_test.sh logger=wandb hub.push=true       # same, also exercising W&B and the Hub
DATASET=hh_helpful MODEL=opt LOSSES=dpo bash scripts/run_pipeline.sh
```

The smoke test builds tiny random GPT-2 models, truncates every split to 48 samples
(`debug_max_samples=48`) and drives every runner script with them: `run_pipeline.sh` (all 4
methods, DPO), `table1.sh` (TL;DR and UFB loaders, rDPO), `table2.sh`, `table3.sh` (SimPO) and
`ablation.sh` (C_MS variants, IPO). Outputs go to `.smoke/`. It checks that everything runs; the
GRA numbers are meaningless (usually 0.0, since the samples tie). It takes about 15-20 minutes on
an A100, mostly dataset preparation and process start-up.

**Google Colab:** Colab ships newer transformers / huggingface_hub than this repo pins, so
`scripts/colab_setup.sh` builds a venv on top of Colab's torch (the tested setup, A100):

```bash
!git clone https://github.com/mirsazzathossain/ms-po.git /content/ms-po
!bash /content/ms-po/scripts/colab_setup.sh
# .env with WANDB_API_KEY, WANDB_ENTITY, HF_TOKEN, HF_USERNAME (e.g. from Colab secrets)
!source /content/env.sh && METHOD=ms_po LOSS=dpo bash scripts/run_experiment.sh model_dtype=bf16 precision=bf16
```

No Drive needed: with `hub.push=true` (default) every stage output goes to private Hugging Face
repos (models: `<user>/mspo-<checkpoint path>`, data/results: dataset `<user>/mspo-artifacts`),
and training checkpoints are uploaded to `last-checkpoint/` at every save. After a disconnect, set
up again and rerun the same command: finished stages are restored from the Hub and training
resumes from its last checkpoint.

## Running experiments

**Full reproduction** (all tables, in dependency order, resumable):

```bash
cp .env.example .env               # WANDB_API_KEY, WANDB_ENTITY, HF_TOKEN, HF_USERNAME
python main.py stage=preflight     # ~2 min; fix any FAIL before spending GPU hours
NUM_GPUS=8 bash scripts/run_all.sh
```

`run_all.sh` runs preflight, then Table 1, Table 2 and Table 3 (Tables 2-3 reuse Table 1's weak
teachers, annotations and SFT students), then `collect_results`. `RUN_ABLATION=1` adds the
Appendix-B variants. Re-run the same command after an interruption; finished stages are skipped.
Rough cost: ~5,000 A100-hours for everything (Table 1 is ~3,300).

**One experiment** (one table cell, only the stages it needs, resumable):

```bash
METHOD=ms_po LOSS=dpo bash scripts/run_experiment.sh            # defaults: DATASET=hh_helpful MODEL=opt
# single 80 GB GPU, faster but same effective batch (32 x 2 = 64):
PER_DEVICE=32 GRAD_ACCUM=2 GRAD_CKPT=false METHOD=ms_po LOSS=dpo bash scripts/run_experiment.sh model_dtype=bf16 precision=bf16
```

**Pieces:**

```bash
# One dataset / model pair: all 4 methods × losses
DATASET=ufb MODEL=qwen2_5 LOSSES="dpo ipo rdpo" bash scripts/run_pipeline.sh

bash scripts/table1.sh      # 3 model pairs × {HH-RLHF, TL;DR, UFB} × {DPO, IPO, rDPO} × 4 methods
bash scripts/table2.sh      # student-size sweep, DPO
bash scripts/table3.sh      # SimPO
DATASET=hh_helpful MODEL=opt bash scripts/ablation.sh   # App. B variants (+ GAMMAS="0.5 1 2")

# A single stage, with any Hydra override
python main.py stage=train_strong_po dataset=tldr model=qwen3 method=ms_po loss=ipo ms.gamma=0.5
```

Common overrides: `logger=none` turns off W&B, `hub.push=false` keeps everything local, `eval.num_samples=500`
evaluates on a subset, and `model_dtype=bf16` loads models in bf16 (the reference code loads fp32)
to save memory on 7B/8B students.

### Multi-GPU

Every stage supports multiple GPUs. Training runs as DDP through the HF Trainer. Annotation,
KL weights, generation and reward scoring split the data across ranks and merge the shards in
order afterwards.

```bash
NUM_GPUS=4 bash scripts/run_pipeline.sh                     # scripts launch torchrun automatically
torchrun --nproc_per_node=4 main.py stage=train_strong_po dataset=hh_helpful model=opt method=ms_po loss=dpo
```

Batch sizes are per device. The effective batch is `per_device × grad_accum × NUM_GPUS`. To keep
the paper's effective batch of 64 on N GPUs, lower `train.strong_po.gradient_accumulation_steps`
(and the matching SFT setting). Following App. C.3, the scripts drop the per-device batch to 4 for
students larger than 7B parameters (Qwen2.5-7B, Qwen3-8B).

### Hardware

| GPU memory | What fits |
|---|---|
| 8-12 GB (fp16 only, e.g. RTX 2080) | smoke test; weak teachers; 1.3B-3B students with `model_dtype=fp16`; TL;DR (DeBERTa RM) evaluation |
| 40 GB (A100) | everything, with `model_dtype=bf16` for 7B/8B students and the Skywork 8B RM |
| 80 GB | everything with the reference code's fp32 model loading |

With `model_dtype=fp16`, run `train_weak` without it: the weak teachers are fully fine-tuned and
need fp32 weights.

## Logging and weights

- **W&B** (project `ms-po`): every stage is its own run.
  - **Runs:** grouped by `<dataset>-<weak>-to-<strong>`, with `job_type` = stage and tags for
    dataset, family, stage, label source, method and loss. Each run's config has a flat `run.*`
    block (`run.dataset`, `run.pair`, `run.method`, `run.loss`, `run.ms_gamma`, ...), so the runs
    table can be grouped or pivoted directly.
  - **Training runs:** TRL's metrics (loss, rewards, accuracies, margins, log-probs), train and eval.
  - **Annotation / MS-weight runs:** weak-label accuracy vs. human labels, and C_weak / S / C_MS statistics.
  - **Evaluation runs:** `results/gra`, `results/reward_aligned_mean` and `results/reward_sft_mean`,
    plus a full `results/per_prompt` table (prompt, both responses, both rewards, win flag) and an
    `evaluation` artifact with the result JSON, generations and reward scores.
  - **Summary run:** `main.py stage=collect_results` (run automatically by the scripts) logs
    `results-summary` in group `results`. It holds one table per (model pair, loss) in the
    layout of Table 1, a `results/all` table and a `results-summary` artifact with the CSV and
    Markdown.
- **HF Hub** (`hub.push=true`): weak models are uploaded as full checkpoints, strong models as
  LoRA adapters plus `lineage.json`. Each PO repo also includes its SFT adapter, so the repo alone
  is enough to rebuild the model with `models.loading.load_merged(<dir>)`.

## Values the paper and reference code do not give

These values had to be filled in. Each one is a config key, so you can change it:

| Setting | Value used | Key |
|---|---|---|
| rDPO noise rate ε (TRL's default of 0 makes rDPO identical to DPO) | **0.1 (placeholder)** | `loss.label_smoothing` (`loss=rdpo`) |
| SimPO β, γ | TRL CPOConfig defaults: 0.1, 0.5 | `loss.beta`, `loss.simpo_gamma` (`loss=simpo`) |
| Weak-teacher scheduler, warmup, weight decay | HF defaults (linear, 0, 0) | `train.weak_*` |
| UFB preprocessing | same as HH-RLHF | `dataset/ufb.py` |
| TL;DR GRA prompts (its test split has ~86k comparisons) | fixed-seed random 2,000 | `dataset.eval_num_samples` (`dataset=tldr`) |
| Qwen3 checkpoints | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-8B`, the names in the paper | `model.*.name` |

Values taken from the reference code because the paper does not give them: β_w = 0.1
(`opt_125M.yaml`), γ = 1.0 (`compute_ms_po_scores`), max length 512 (1024 for TL;DR), fp16
mixed precision with fp32 weights, eval/save every 500 steps keeping the best checkpoint by
eval_loss, and logging every 10 steps.

Changes made so the reference code runs:

- **TL;DR:** validation rows with no Reddit post (`info["post"] is None`, CNN/DM articles) are
  dropped.
- **Vocabulary size:** for Qwen2.5-0.5B vs. 7B, whose embedding sizes differ, the KL uses the
  shared vocabulary prefix.
- **TRL API:** `CW_PO.py`'s overrides are adapted to the TRL 0.21 `dpo_loss` and
  `concatenated_forward` signatures.
- **SimPO:** TRL's CPOTrainer requires `max_prompt_length < max_length`, so it uses half of
  `max_length` (CPOConfig's own default ratio).
- **Hub upload:** if the Hub rejects PEFT's auto-generated model card (its `base_model` is a local
  path), the checkpoint is uploaded without `README.md`.

## Tested environment

The full pipeline (smoke test with W&B and Hub upload) has been run end to end on Colab with an
A100-SXM4 40 GB, Python 3.13, torch 2.11.0+cu130, and the exact package versions in
`requirements.txt`. The Docker image uses the matching `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime`
base, which needs NVIDIA driver >= 580; on older drivers switch to the `cuda12.8` tag.
