"""TRL trainers and configs.

Configs mirror resources/ms_po_ours/utils/mange_config.py (DPOConfig fields, eval/save every
`eval_steps`, best model by eval_loss, fp16 on CUDA, remove_unused_columns=False) with the
hyperparameters of Tables 4-6.

Sample weighting follows resources/ms_po_ours/utils/CW_PO.py: the batch's `confidence` column is
captured in `concatenated_forward` and multiplies the per-sample loss before TRL's mean, i.e.
L = E[C(x, y+, y-) * l(pi_s; x, y+, y-)] (Eq. 6 / Eq. 12). C = 1 for Human / WS-PO,
C_weak for CW-PO and C_MS for MS-PO. All trainers are HF Trainers, so `torchrun` gives DDP.
"""

from __future__ import annotations

import torch
from trl import CPOConfig, CPOTrainer, DPOConfig, DPOTrainer, SFTConfig
from trl.trainer.dpo_trainer import DataCollatorForPreference

from utils.logging import report_to


def _common_args(cfg, stage_cfg, output_dir: str, run_name: str, max_length: int) -> dict:
    fp16 = cfg.precision == "fp16" and torch.cuda.is_available()
    bf16 = cfg.precision == "bf16" and torch.cuda.is_available()
    return dict(
        output_dir=output_dir,
        run_name=run_name,
        # Training
        num_train_epochs=stage_cfg.num_train_epochs,
        learning_rate=stage_cfg.learning_rate,
        lr_scheduler_type=stage_cfg.lr_scheduler_type,
        warmup_steps=stage_cfg.warmup_steps,
        weight_decay=stage_cfg.weight_decay,
        optim=stage_cfg.optim,
        # Batch
        per_device_train_batch_size=stage_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=stage_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=stage_cfg.gradient_accumulation_steps,
        # Sequence
        max_length=max_length,
        # Logging
        logging_steps=stage_cfg.logging_steps,
        logging_first_step=True,
        # Evaluation
        eval_strategy="steps",
        eval_steps=stage_cfg.eval_steps,
        # Checkpointing
        save_strategy="steps",
        save_steps=stage_cfg.save_steps,
        save_total_limit=2,
        # Best model
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Precision
        fp16=fp16,
        bf16=bf16,
        # Reporting
        report_to=report_to(cfg),
        # Dataset
        remove_unused_columns=False,
        # Gradient checkpointing
        gradient_checkpointing=stage_cfg.gradient_checkpointing,
        seed=cfg.seed,
    )


def sft_config(cfg, stage_cfg, output_dir, run_name, max_length) -> SFTConfig:
    # prompt/completion data -> loss on completion tokens only (TRL default for this format)
    return SFTConfig(**_common_args(cfg, stage_cfg, output_dir, run_name, max_length))


def dpo_config(cfg, stage_cfg, output_dir, run_name, max_length, loss_cfg) -> DPOConfig:
    return DPOConfig(
        **_common_args(cfg, stage_cfg, output_dir, run_name, max_length),
        beta=loss_cfg.beta,
        loss_type=loss_cfg.trl_loss_type,
        label_smoothing=loss_cfg.get("label_smoothing", 0.0),
    )


def cpo_config(cfg, stage_cfg, output_dir, run_name, max_length, loss_cfg) -> CPOConfig:
    return CPOConfig(
        **_common_args(cfg, stage_cfg, output_dir, run_name, max_length),
        beta=loss_cfg.beta,
        loss_type=loss_cfg.trl_loss_type,
        simpo_gamma=loss_cfg.simpo_gamma,
        cpo_alpha=loss_cfg.cpo_alpha,
        # CPOTrainer requires max_prompt_length < max_length; keep CPOConfig's default ratio (512 / 1024).
        max_prompt_length=max_length // 2,
    )


class ConfidenceCollator(DataCollatorForPreference):
    """TRL's preference collator that also keeps the per-pair `confidence` column."""

    def torch_call(self, examples):
        output = super().torch_call(examples)
        if "confidence" in examples[0]:
            output["confidence"] = torch.tensor([float(e["confidence"]) for e in examples], dtype=torch.float32)
        return output


class CWDPOTrainer(DPOTrainer):
    """DPO / IPO / rDPO with sample-wise confidence weights (resources CW_PO.py)."""

    def dpo_loss(
        self,
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        loss_type="sigmoid",
        model_output=None,
    ):
        losses, chosen_rewards, rejected_rewards = super().dpo_loss(
            chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, loss_type, model_output
        )
        # Retrieve confidence scores attached to current batch during forward step
        if getattr(self, "_current_batch_confidence", None) is not None:
            confidence = self._current_batch_confidence.to(device=losses.device, dtype=losses.dtype)
            # Apply sample-wise confidence weights
            losses = confidence * losses
        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(self, model, batch, is_ref_model=False):
        """Intercept batch forward pass to extract confidence tensor."""
        self._current_batch_confidence = batch.get("confidence")
        return super().concatenated_forward(model, batch, is_ref_model=is_ref_model)


class CWCPOTrainer(CPOTrainer):
    """Reference-free SimPO (TRL CPOTrainer, loss_type='simpo', cpo_alpha=0) with confidence weights."""

    def cpo_loss(self, policy_chosen_logps, policy_rejected_logps):
        losses, chosen_rewards, rejected_rewards = super().cpo_loss(policy_chosen_logps, policy_rejected_logps)
        if getattr(self, "_current_batch_confidence", None) is not None:
            confidence = torch.as_tensor(self._current_batch_confidence, dtype=losses.dtype, device=losses.device)
            losses = confidence * losses
        return losses, chosen_rewards, rejected_rewards

    def concatenated_forward(self, model, batch):
        self._current_batch_confidence = batch.get("confidence")
        return super().concatenated_forward(model, batch)
