# MS-PO: Multi-Granular Student-Aware Preference Optimization

Reference implementation of **MS-PO** for weak-to-strong alignment. A weak teacher, trained on 30%
of the human labels, pseudo-labels the remaining 70%. A stronger student is then aligned on those
labels, and each pair is weighted by how closely teacher and student agree, measured with
token-level forward KL.

The code covers every setting in the paper:

- **Methods:** Human, WS-PO, CW-PO, MS-PO
- **Losses:** DPO, IPO, rDPO, SimPO
- **Datasets:** HH-RLHF (combined, Helpful, Harmless), TL;DR, UltraFeedback-Binarized
- **Model pairs:** OPT-125M→OPT-{1.3B,2.7B,6.7B}, Qwen2.5-0.5B→Qwen2.5-{1.5B,3B,7B}, Qwen3-0.6B→Qwen3-8B
- **Metric:** Gold Reward Accuracy (GRA)

Stack: Hydra (configs), W&B (logging), the Hugging Face Hub (weights), Docker, and multi-GPU DDP through `torchrun`.

## Layout

```
ms-po/
├── main.py                 # single entry point: python main.py stage=<stage> ...
├── configs/                # Hydra configs
│   ├── config.yaml         #   root: seed, precision, MS-PO (gamma / variant)
│   ├── dataset/            #   hh_rlhf, hh_helpful, hh_harmless, tldr, ufb
│   ├── model/              #   opt, qwen2_5, qwen3 (weak/strong pairs), tiny (smoke test)
│   ├── method/             #   human, ws_po, cw_po, ms_po
│   ├── loss/               #   dpo, ipo, rdpo, simpo
│   ├── train/              #   weak_sft, weak_po, strong_sft, strong_po (Tables 4-6)
│   ├── eval/ logger/ hub/ paths/
├── dataset/                # dataset code: loaders, length filter + 30/70 split, tokenisation, weak labels
├── models/                 # model code: loading, LoRA, adapter lineage, batched scoring
├── pipeline/               # one module per stage, each exposing run(cfg)
├── utils/                  # losses + weights, trainer, evaluation, distributed, W&B, HF Hub, io
├── scripts/                # shell runners (pipeline, Tables 1-3, ablations, smoke test)
├── data/                   # datasets (generated; see data/README.md)
├── checkpoints/            # trained weights (generated)
├── outputs/                # generations, GRA results, Hydra logs (generated)
├── Dockerfile  docker-compose.yml  requirements.txt  .env.example
```

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
| 0 | `prepare_data`: parse, length filter, split 30/70, 1% of the 70% held out for validation | `data/processed/<ds>/` |
| 1 | `train_weak`: full fine-tune of the weak model, SFT then DPO on D_labeled | `checkpoints/<ds>/weak/<weak>/{sft,dpo}` |
| 2 | `annotate`: weak labels + C_weak on D_unlabeled | `data/annotated/<ds>/<weak>/unlabeled.jsonl` |
| 3 | `train_strong_sft`: LoRA SFT of the student on (x, chosen), with human or weak labels (TRL SFTTrainer) | `checkpoints/<ds>/strong/<strong>/sft_{human,weak}` |
| 4 | `compute_ms_weights`: S(x,y⁺), S(x,y⁻) between the weak DPO teacher and the SFT student | `data/annotated/<ds>/<weak>/ms_weights_<strong>.jsonl` |
| 5 | `train_strong_po`: weighted preference optimisation (LoRA; reference model = SFT) | `checkpoints/<ds>/strong/<strong>/<method>_<loss>` |
| 6 | `evaluate`: sample (T=0.95, 512 tokens) from the SFT and aligned models, then compute gold RM GRA | `outputs/<ds>/<strong>/results/*.json` |
| 7 | `collect_results`: build a Table-1-style summary | `outputs/results.{md,csv}` |

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

**Plain Python / Google Colab:**

```bash
git clone <repo> ms-po && cd ms-po
pip install -r requirements.txt
wandb login            # or: export WANDB_API_KEY=...
huggingface-cli login  # needed for hub.push=true and the gated Llama-based Skywork RM
bash scripts/smoke_test.sh                                  # ~10 min plumbing check, tiny models
DATASET=hh_rlhf MODEL=opt LOSSES=dpo bash scripts/run_pipeline.sh
```

On Colab, keep `data/`, `checkpoints/` and `outputs/` on Google Drive so a disconnect does not
lose finished stages. Point `MSPO_ROOT` at a Drive folder, or symlink the three folders into it.
Because finished stages are skipped, you can rerun the same command after a disconnect to resume.

## Running experiments

```bash
# One dataset / model pair: all 4 methods × losses
DATASET=ufb MODEL=qwen2_5 LOSSES="dpo ipo rdpo" bash scripts/run_pipeline.sh

bash scripts/table1.sh      # 3 model pairs × {HH-RLHF, TL;DR, UFB} × {DPO, IPO, rDPO} × 4 methods
bash scripts/table2.sh      # student-size sweep, DPO
bash scripts/table3.sh      # SimPO
DATASET=hh_rlhf MODEL=opt bash scripts/ablation.sh   # App. B variants (+ GAMMAS="0.5 1 2")

# A single stage, with any Hydra override
python main.py stage=train_strong_po dataset=tldr model=qwen3 method=ms_po loss=ipo ms.gamma=0.5
```

Common overrides: `logger=none` turns off W&B, `hub.push=true` uploads weights, `eval.num_samples=500`
evaluates on a subset, and `model_dtype=bf16` loads models in bf16 (the reference code loads fp32)
to save memory on 7B/8B students.

### Multi-GPU

Every stage supports multiple GPUs. Training runs as DDP through the HF Trainer. Annotation,
KL weights, generation and reward scoring split the data across ranks and merge the shards in
order afterwards.

```bash
NUM_GPUS=4 bash scripts/run_pipeline.sh                     # scripts launch torchrun automatically
torchrun --nproc_per_node=4 main.py stage=train_strong_po dataset=hh_rlhf model=opt method=ms_po loss=dpo
```

Batch sizes are per device. The effective batch is `per_device × grad_accum × NUM_GPUS`. To keep
the paper's effective batch of 64 on N GPUs, lower `train.strong_po.gradient_accumulation_steps`
(and the matching SFT setting). The scripts follow the paper and drop the per-device batch to 4 for
7B/8B students.

## Logging and weights

- **W&B:** project `ms-po`, with runs grouped by `<dataset>-<weak>-to-<strong>`. Training logs
  include TRL's DPO metrics: loss, rewards, accuracies, margins and log-probs. Annotation and
  MS-weight runs log weak-label accuracy and C_weak/C_MS statistics.
  Evaluation logs GRA and a table of sample generations.
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
