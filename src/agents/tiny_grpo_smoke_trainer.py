"""
Phase 2.1: Tiny GRPO smoke trainer.

Runs a minimal real optimizer.step on clean 20_g4 strategy groups using
quality-only reward_largek_mix_1000 advantages. Does NOT promote checkpoints.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.agents.actor_logprob_dryrun import _extract_lengths
from src.agents.grpo_advantage_mock import _get_batch, _get_non_tensor_list
from src.agents.grpo_loss_dryrun import GRPOLossDryRun
from src.agents.no_update_trainer_dryrun import NoUpdateTrainerDryRun
from src.agents.real_grpo_loss_dryrun import DEFAULT_CANDIDATE, EPS

CHECKPOINT_LABEL = "SMOKE_ONLY_DO_NOT_PROMOTE"
KL_EXPLODE_THRESHOLD = 0.2
MAX_GRAD_NORM = 1.0
TINY_TRAIN_WARNING = (
    "Phase 2.1 tiny GRPO smoke training only. "
    "Checkpoints are SMOKE_ONLY_DO_NOT_PROMOTE and must not be promoted."
)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class TinyGrpoSmokeTrainer:
    """Minimal real GRPO update for engineering stability validation."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        candidate_name: str = DEFAULT_CANDIDATE,
        train_batch_size: int = 20,
        rollout_n: int = 4,
        ppo_mini_batch_size: int = 20,
        micro_batch_size: int = 4,
        max_update_steps: int = 1,
        learning_rate: float = 1e-6,
        kl_coef: float = 0.01,
        cliprange: float = 0.2,
        loss_agg_mode: str = "token-mean",
        max_grad_norm: float = MAX_GRAD_NORM,
        kl_explode_threshold: float = KL_EXPLODE_THRESHOLD,
        dtype: str = "bfloat16",
        device: str = "cuda",
        cuda_device_index: Optional[int] = None,
        logprob_micro_batch_size: int = 2,
        seed: int = 42,
        eps: float = EPS,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.candidate_name = candidate_name
        self.train_batch_size = train_batch_size
        self.rollout_n = rollout_n
        self.ppo_mini_batch_size = ppo_mini_batch_size
        self.micro_batch_size = micro_batch_size
        self.max_update_steps = max_update_steps
        self.learning_rate = learning_rate
        self.kl_coef = kl_coef
        self.cliprange = cliprange
        self.loss_agg_mode = loss_agg_mode
        self.max_grad_norm = max_grad_norm
        self.kl_explode_threshold = kl_explode_threshold
        self.dtype = dtype
        self.device = device
        self.cuda_device_index = cuda_device_index
        self.logprob_micro_batch_size = logprob_micro_batch_size
        self.seed = seed
        self.eps = eps

        self._builder = NoUpdateTrainerDryRun(
            tokenizer_path=tokenizer_path,
            candidate_name=candidate_name,
            train_batch_size=train_batch_size,
            rollout_n=rollout_n,
            ppo_mini_batch_size=ppo_mini_batch_size,
            micro_batch_size=micro_batch_size,
            cliprange=cliprange,
            kl_coef=kl_coef,
            loss_agg_mode=loss_agg_mode,
            seed=seed,
            eps=eps,
        )
        self._loss_fn = GRPOLossDryRun(
            cliprange=cliprange,
            kl_coef=kl_coef,
            loss_agg_mode=loss_agg_mode,
            seed=seed,
        )

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._ref_log_probs_snapshot: Optional[torch.Tensor] = None

    def load_model(self) -> None:
        if self.dtype not in _DTYPE_MAP:
            raise ValueError(f"unsupported dtype: {self.dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        device_map: Any
        if self.device == "cuda":
            if self.cuda_device_index is not None:
                device_map = {"": f"cuda:{self.cuda_device_index}"}
            else:
                device_map = "auto"
        else:
            device_map = None

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=_DTYPE_MAP[self.dtype],
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.train()
        try:
            self.model.gradient_checkpointing_enable()
        except Exception:
            pass

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

    @property
    def model_device(self) -> torch.device:
        if self.model is None:
            raise RuntimeError("model not loaded")
        return next(self.model.parameters()).device

    def _slice_batch(self, batch: Dict[str, torch.Tensor], indices: List[int]) -> Dict[str, torch.Tensor]:
        idx = torch.tensor(indices, dtype=torch.long)
        return {k: v.index_select(0, idx) for k, v in batch.items() if isinstance(v, torch.Tensor)}

    def _forward_response_log_probs(
        self,
        data_proto: Any,
        indices: List[int],
        *,
        enable_grad: bool,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("model not loaded")

        batch = _get_batch(data_proto)
        sub = self._slice_batch(batch, indices)
        input_ids = sub["input_ids"].to(self.model_device)
        attention_mask = sub["attention_mask"].to(self.model_device)
        responses = sub["responses"]
        response_attention_mask = sub["response_attention_mask"]

        prompt_lengths, response_lengths = _extract_lengths(data_proto)
        prompt_lengths = prompt_lengths[indices]
        response_lengths = response_lengths[indices]

        micro_batch_size = len(indices)
        response_len = responses.shape[1]
        out_device = self.model_device if enable_grad else torch.device("cpu")
        out_log_probs = torch.zeros(
            (micro_batch_size, response_len),
            dtype=torch.float32,
            device=out_device,
        )

        context = torch.enable_grad() if enable_grad else torch.no_grad()
        with context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            log_probs_all = F.log_softmax(shift_logits.float(), dim=-1)
            token_log_probs = torch.gather(
                log_probs_all,
                dim=-1,
                index=shift_labels.unsqueeze(-1),
            ).squeeze(-1)

        for i in range(micro_batch_size):
            plen = int(prompt_lengths[i].item())
            rlen = int(response_lengths[i].item())
            slice_start = plen - 1
            slice_end = plen - 1 + rlen
            slice_lp = token_log_probs[i, slice_start:slice_end]
            if enable_grad:
                out_log_probs[i, :rlen] = slice_lp
            else:
                out_log_probs[i, :rlen] = slice_lp.detach().cpu()

        mask = response_attention_mask.float()
        if enable_grad:
            out_log_probs = out_log_probs * mask.to(out_device)
        else:
            out_log_probs = out_log_probs * mask.cpu()
        return out_log_probs

    def _compute_old_log_probs(self, data_proto: Any) -> torch.Tensor:
        batch = _get_batch(data_proto)
        num_records = batch["input_ids"].shape[0]
        response_len = batch["responses"].shape[1]
        old_log_probs = torch.zeros((num_records, response_len), dtype=torch.float32)

        self.model.eval()
        for start in range(0, num_records, self.logprob_micro_batch_size):
            end = min(start + self.logprob_micro_batch_size, num_records)
            indices = list(range(start, end))
            with torch.no_grad():
                chunk = self._forward_response_log_probs(data_proto, indices, enable_grad=False)
            old_log_probs[start:end] = chunk
        self.model.train()
        return old_log_probs

    def _compute_kl_diagnostics(
        self,
        data_proto: Any,
        response_mask: torch.Tensor,
    ) -> Dict[str, float]:
        """Eval-mode current actor vs frozen ref snapshot logprob diagnostics."""
        if self._ref_log_probs_snapshot is None:
            raise RuntimeError("ref logprob snapshot missing; call run() first")

        assert self.model is not None
        ref_log_probs = self._ref_log_probs_snapshot.float() * response_mask.cpu()

        was_training = self.model.training
        self.model.eval()
        try:
            actor_log_probs = self._compute_old_log_probs(data_proto)
        finally:
            if was_training:
                self.model.train()

        valid = response_mask.bool()
        gap = (actor_log_probs - ref_log_probs)[valid]
        if gap.numel() == 0:
            return {
                "signed_logprob_gap_mean": 0.0,
                "signed_logprob_gap_abs_mean": 0.0,
                "approx_kl_nonnegative": 0.0,
                "actor_logprob_mean": 0.0,
                "ref_logprob_mean": 0.0,
            }

        log_ratio = gap
        approx_kl_tokens = torch.exp(log_ratio) - 1.0 - log_ratio
        valid_actor = actor_log_probs[valid]
        valid_ref = ref_log_probs[valid]

        return {
            "signed_logprob_gap_mean": float(gap.mean().item()),
            "signed_logprob_gap_abs_mean": float(gap.abs().mean().item()),
            "approx_kl_nonnegative": float(approx_kl_tokens.mean().item()),
            "actor_logprob_mean": float(valid_actor.mean().item()),
            "ref_logprob_mean": float(valid_ref.mean().item()),
        }

    def _reward_stats(self, merged_records: List[Dict[str, Any]]) -> Dict[str, float]:
        grouped: Dict[str, List[float]] = {}
        for row in merged_records:
            grouped.setdefault(row["group_id"], []).append(float(row["reward"]))

        zero_std_groups = 0
        spread_groups = 0
        for rewards in grouped.values():
            if len(rewards) > 1 and pstdev(rewards) <= self.eps:
                zero_std_groups += 1
            if len(rewards) > 1 and (max(rewards) - min(rewards)) > self.eps:
                spread_groups += 1

        num_groups = len(grouped) or 1
        all_rewards = [float(r["reward"]) for r in merged_records]
        return {
            "mean_reward": float(mean(all_rewards)) if all_rewards else 0.0,
            "reward_std": float(pstdev(all_rewards)) if len(all_rewards) > 1 else 0.0,
            "zero_std_group_rate": zero_std_groups / num_groups,
            "retrieval_quality_spread_group_rate": spread_groups / num_groups,
        }

    def _compute_micro_policy_loss(
        self,
        *,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio) * response_mask

        unclipped = ratio * advantages
        clipped_ratio = torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
        clipped = clipped_ratio * advantages
        policy_loss_mat = -torch.min(unclipped, clipped) * response_mask
        policy_loss = self._loss_fn._aggregate_loss(policy_loss_mat, response_mask)

        token_kl = (log_probs - ref_log_probs) * response_mask
        log_ratio_ref = (log_probs - ref_log_probs) * response_mask
        valid_mask = response_mask.bool()
        valid_ratio = ratio[valid_mask]
        valid_kl = token_kl[valid_mask]
        valid_log_ratio_ref = log_ratio_ref[valid_mask]
        valid_actor_lp = log_probs[valid_mask]
        valid_ref_lp = ref_log_probs[valid_mask]

        # Non-negative KL surrogate: E[exp(r) - 1 - r] >= 0
        approx_kl_tokens = torch.exp(valid_log_ratio_ref) - 1.0 - valid_log_ratio_ref
        approx_kl_nonnegative = float(approx_kl_tokens.mean().item()) if approx_kl_tokens.numel() else 0.0
        signed_logprob_gap_mean = float(valid_kl.mean().item()) if valid_kl.numel() else 0.0
        signed_logprob_gap_abs_mean = (
            float(valid_kl.abs().mean().item()) if valid_kl.numel() else 0.0
        )
        actor_logprob_mean = float(valid_actor_lp.mean().item()) if valid_actor_lp.numel() else 0.0
        ref_logprob_mean = float(valid_ref_lp.mean().item()) if valid_ref_lp.numel() else 0.0

        clip_mask = ((ratio < 1.0 - self.cliprange) | (ratio > 1.0 + self.cliprange)) & valid_mask
        clipfrac = clip_mask.float().sum() / response_mask.sum().clamp_min(1.0)

        return {
            "policy_loss": policy_loss,
            "policy_loss_value": float(policy_loss.item()),
            "clipfrac": float(clipfrac.item()),
            "mean_valid_kl": signed_logprob_gap_mean,
            "signed_logprob_gap_mean": signed_logprob_gap_mean,
            "signed_logprob_gap_abs_mean": signed_logprob_gap_abs_mean,
            "approx_kl_nonnegative": approx_kl_nonnegative,
            "actor_logprob_mean": actor_logprob_mean,
            "ref_logprob_mean": ref_logprob_mean,
        }

    def _hard_stop(
        self,
        *,
        nan_detected: bool,
        approx_kl_nonnegative: float,
        grad_norm: float,
        policy_loss_value: float,
        check_kl: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        if nan_detected:
            return True, "NaN detected in loss, KL, or gradients"
        if not torch.isfinite(torch.tensor(policy_loss_value)):
            return True, "policy loss is not finite"
        if not torch.isfinite(torch.tensor(grad_norm)):
            return True, "grad_norm is not finite"
        if check_kl and approx_kl_nonnegative > self.kl_explode_threshold:
            return True, (
                f"approx_kl_nonnegative {approx_kl_nonnegative:.4f} "
                f"exceeds threshold {self.kl_explode_threshold}"
            )
        return False, None

    def _save_checkpoint(
        self,
        output_dir: Path,
        step: int,
        *,
        checkpoint_prefix: str = "smoke_step",
    ) -> Path:
        ckpt_dir = output_dir / "checkpoints" / f"{checkpoint_prefix}_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        assert self.model is not None
        assert self.tokenizer is not None
        self.model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
        (ckpt_dir / "SMOKE_ONLY_DO_NOT_PROMOTE").write_text(
            CHECKPOINT_LABEL + "\n",
            encoding="utf-8",
        )
        return ckpt_dir

    def _optimizer_step(
        self,
        data_proto: Any,
        step_idx: int,
    ) -> Dict[str, Any]:
        assert self.model is not None
        assert self.optimizer is not None

        batch = _get_batch(data_proto)
        num_records = batch["input_ids"].shape[0]
        response_mask = batch["response_attention_mask"].float()
        advantages = batch["advantages"].float() * response_mask
        old_log_probs = batch["old_log_probs"].float() * response_mask
        ref_log_probs = (
            self._ref_log_probs_snapshot.float().to(old_log_probs.device)
            if self._ref_log_probs_snapshot is not None
            else batch["ref_log_probs"].float()
        ) * response_mask

        self.optimizer.zero_grad(set_to_none=True)
        num_micro = num_records // self.micro_batch_size
        total_policy_loss = torch.tensor(0.0, device=self.model_device)
        last_loss_output: Dict[str, Any] = {}

        nan_detected = False
        oom_detected = False

        try:
            for micro_idx in range(num_micro):
                start = micro_idx * self.micro_batch_size
                end = start + self.micro_batch_size
                indices = list(range(start, end))

                log_probs = self._forward_response_log_probs(
                    data_proto,
                    indices,
                    enable_grad=True,
                )

                micro_old = old_log_probs[start:end].to(self.model_device)
                micro_ref = ref_log_probs[start:end].to(self.model_device)
                micro_adv = advantages[start:end].to(self.model_device)
                micro_mask = response_mask[start:end].to(self.model_device)

                loss_output = self._compute_micro_policy_loss(
                    log_probs=log_probs * micro_mask,
                    old_log_probs=micro_old,
                    ref_log_probs=micro_ref,
                    advantages=micro_adv,
                    response_mask=micro_mask,
                )
                policy_loss = loss_output["policy_loss"] / num_micro
                if not torch.isfinite(policy_loss):
                    nan_detected = True
                    break
                policy_loss.backward()
                total_policy_loss = total_policy_loss + policy_loss.detach()
                last_loss_output = loss_output

        except torch.cuda.OutOfMemoryError:
            oom_detected = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "step": step_idx,
                "optimizer_step_called": False,
                "nan_detected": False,
                "oom_detected": True,
                "policy_loss": None,
                "mean_kl": None,
                "clipfrac": None,
                "entropy": None,
                "grad_norm": None,
                "learning_rate": self.learning_rate,
                "hard_stop": True,
                "hard_stop_reason": "CUDA OOM during training step",
            }

        grad_norm = 0.0
        if not nan_detected and not oom_detected:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                ).item()
            )
            if not torch.isfinite(torch.tensor(grad_norm)):
                nan_detected = True

        policy_loss_value = float(total_policy_loss.item()) if torch.isfinite(total_policy_loss) else float("nan")

        pre_hard_stop, pre_reason = self._hard_stop(
            nan_detected=nan_detected,
            approx_kl_nonnegative=0.0,
            grad_norm=grad_norm,
            policy_loss_value=policy_loss_value,
            check_kl=False,
        )

        optimizer_step_called = False
        if not pre_hard_stop:
            self.optimizer.step()
            optimizer_step_called = True

        kl_diag = self._compute_kl_diagnostics(data_proto, response_mask)
        mean_kl = float(kl_diag.get("signed_logprob_gap_mean", 0.0))
        approx_kl = float(kl_diag.get("approx_kl_nonnegative", 0.0))

        hard_stop, hard_stop_reason = self._hard_stop(
            nan_detected=nan_detected,
            approx_kl_nonnegative=approx_kl,
            grad_norm=grad_norm,
            policy_loss_value=policy_loss_value,
            check_kl=True,
        )
        if pre_hard_stop:
            hard_stop = True
            hard_stop_reason = pre_reason

        return {
            "step": step_idx,
            "optimizer_step_called": optimizer_step_called,
            "nan_detected": nan_detected,
            "oom_detected": oom_detected,
            "policy_loss": policy_loss_value,
            "mean_kl": mean_kl,
            "signed_logprob_gap_mean": mean_kl,
            "signed_logprob_gap_abs_mean": float(
                kl_diag.get("signed_logprob_gap_abs_mean", abs(mean_kl))
            ),
            "approx_kl_nonnegative": approx_kl,
            "actor_logprob_mean": float(kl_diag.get("actor_logprob_mean", 0.0)),
            "ref_logprob_mean": float(kl_diag.get("ref_logprob_mean", 0.0)),
            "kl_loss": approx_kl * self.kl_coef,
            "clipfrac": float(last_loss_output.get("clipfrac", 0.0) or 0.0),
            "entropy": None,
            "grad_norm": grad_norm,
            "learning_rate": self.learning_rate,
            "loss_finite": torch.isfinite(torch.tensor(policy_loss_value)).item(),
            "grad_norm_finite": torch.isfinite(torch.tensor(grad_norm)).item(),
            "json_format_ok": True,
            "checkpoint_saved": False,
            "checkpoint_promoted": False,
            "hard_stop": hard_stop,
            "hard_stop_reason": hard_stop_reason,
        }

    def run(
        self,
        rollout_path: str | Path,
        shaped_reward_path: str | Path,
        output_dir: str | Path,
        *,
        max_prompt_length: int = 1024,
        max_response_length: int = 2048,
        max_total_length: int = 3072,
        metrics_filename: str = "tiny_train_metrics.jsonl",
        summary_filename: str = "tiny_train_summary.json",
        config_filename: str = "tiny_train_config.yaml",
        phase: str = "2.1",
        mode: str = "tiny_grpo_smoke_training",
        stability_monitor: Optional[Any] = None,
        save_steps: Optional[List[int]] = None,
        eval_steps: Optional[List[int]] = None,
        checkpoint_prefix: str = "smoke_step",
        step_hook: Optional[Any] = None,
    ) -> Dict[str, Any]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        metrics_path = out / metrics_filename
        failure_flags = {
            "nan_detected": False,
            "oom_detected": False,
            "kl_exploded": False,
            "grad_norm_finite": True,
            "loss_finite": True,
        }

        actual_update_steps = 0
        optimizer_step_called = False
        checkpoint_saved = False
        checkpoint_dir: Optional[Path] = None
        step_metrics: List[Dict[str, Any]] = []
        merged_records: List[Dict[str, Any]] = []
        saved_checkpoints: List[Dict[str, Any]] = []
        save_step_set = set(save_steps) if save_steps is not None else None
        eval_step_set = set(eval_steps) if eval_steps is not None else set()
        stop_snapshot_path: Optional[str] = None

        try:
            inputs = self._builder.load_inputs(str(rollout_path), str(shaped_reward_path))
            data_proto, used_real_dataproto, merged_records = self._builder.build_trainer_facing_dataproto(
                inputs,
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                max_total_length=max_total_length,
            )

            self.load_model()
            old_log_probs = self._compute_old_log_probs(data_proto)
            self._ref_log_probs_snapshot = old_log_probs.clone()
            batch = _get_batch(data_proto)
            batch["old_log_probs"] = old_log_probs
            batch["ref_log_probs"] = self._ref_log_probs_snapshot.clone()

            data_proto, adv_meta = self._builder.compute_verl_grpo_advantage_no_update(data_proto)
            reward_stats = self._reward_stats(merged_records)

            for step_idx in range(1, self.max_update_steps + 1):
                step_result = self._optimizer_step(data_proto, step_idx)
                step_result.update(reward_stats)

                if step_result.get("optimizer_step_called"):
                    actual_update_steps += 1
                    optimizer_step_called = True
                    should_save = save_step_set is None or step_idx in save_step_set
                    if should_save:
                        checkpoint_dir = self._save_checkpoint(
                            out, step_idx, checkpoint_prefix=checkpoint_prefix
                        )
                        checkpoint_saved = True
                        step_result["checkpoint_saved"] = True
                        step_result["checkpoint_path"] = str(checkpoint_dir)
                        saved_checkpoints.append(
                            {
                                "step": step_idx,
                                "path": str(checkpoint_dir),
                                "label": CHECKPOINT_LABEL,
                                "optimizer_step_called": True,
                            }
                        )
                    else:
                        step_result["checkpoint_saved"] = False
                    step_result["checkpoint_promoted"] = False

                if (
                    step_idx in eval_step_set
                    and not step_result.get("checkpoint_saved")
                    and step_result.get("optimizer_step_called")
                ):
                    eval_ckpt_dir = self._save_checkpoint(
                        out, step_idx, checkpoint_prefix=f"{checkpoint_prefix}_eval"
                    )
                    step_result["eval_snapshot_path"] = str(eval_ckpt_dir)

                if stability_monitor is not None:
                    stop, stop_reason = stability_monitor.should_stop(step_result)
                    step_result["monitor_should_stop"] = stop
                    step_result["monitor_stop_reason"] = stop_reason
                    if stop and not step_result.get("hard_stop"):
                        step_result["hard_stop"] = True
                        step_result["hard_stop_reason"] = stop_reason

                if (
                    step_result.get("hard_stop")
                    and step_result.get("optimizer_step_called")
                    and not step_result.get("checkpoint_path")
                    and not step_result.get("eval_snapshot_path")
                ):
                    stop_ckpt_dir = self._save_checkpoint(
                        out, step_idx, checkpoint_prefix=f"{checkpoint_prefix}_stop"
                    )
                    step_result["stop_snapshot_path"] = str(stop_ckpt_dir)
                    stop_snapshot_path = str(stop_ckpt_dir)

                if step_hook is not None:
                    step_hook(step_idx, step_result)

                step_metrics.append(step_result)
                with metrics_path.open("a", encoding="utf-8") as fout:
                    fout.write(json.dumps(step_result, ensure_ascii=False) + "\n")

                if step_result.get("nan_detected"):
                    failure_flags["nan_detected"] = True
                if step_result.get("oom_detected"):
                    failure_flags["oom_detected"] = True
                if step_result.get("approx_kl_nonnegative", 0.0) > self.kl_explode_threshold:
                    failure_flags["kl_exploded"] = True
                if step_result.get("grad_norm") is not None and not torch.isfinite(
                    torch.tensor(step_result["grad_norm"])
                ):
                    failure_flags["grad_norm_finite"] = False
                if step_result.get("policy_loss") is not None and not torch.isfinite(
                    torch.tensor(step_result["policy_loss"])
                ):
                    failure_flags["loss_finite"] = False

                if step_result.get("hard_stop"):
                    break

            spread_stats = self._builder._spread_stats_from_records(merged_records)
            training_smoke_passed = (
                optimizer_step_called
                and not failure_flags["nan_detected"]
                and not failure_flags["oom_detected"]
                and not failure_flags["kl_exploded"]
                and failure_flags["grad_norm_finite"]
                and failure_flags["loss_finite"]
            )

            summary = {
                "phase": phase,
                "mode": mode,
                "max_update_steps": self.max_update_steps,
                "actual_update_steps": actual_update_steps,
                "train_batch_size": self.train_batch_size,
                "rollout_n": self.rollout_n,
                "num_rollout_records": self.train_batch_size * self.rollout_n,
                "reward_candidate": self.candidate_name,
                "penalties_in_advantage": False,
                "diagnostic_oracle_reward_used": False,
                "learning_rate": self.learning_rate,
                "kl_coef": self.kl_coef,
                "cliprange": self.cliprange,
                "nan_detected": failure_flags["nan_detected"],
                "oom_detected": failure_flags["oom_detected"],
                "kl_exploded": failure_flags["kl_exploded"],
                "grad_norm_finite": failure_flags["grad_norm_finite"],
                "loss_finite": failure_flags["loss_finite"],
                "optimizer_step_called": optimizer_step_called,
                "checkpoint_saved": checkpoint_saved,
                "checkpoint_promoted": False,
                "checkpoint_label": CHECKPOINT_LABEL,
                "checkpoint_path": (
                    saved_checkpoints[-1]["path"] if saved_checkpoints else (str(checkpoint_dir) if checkpoint_dir else None)
                ),
                "used_real_dataproto": used_real_dataproto,
                "used_verl_compute_advantage": adv_meta.get("used_verl_compute_advantage", False),
                "training_smoke_passed": training_smoke_passed,
                "safe_for_larger_training": False,
                "next_recommendation": (
                    "Inspect tiny smoke logs; if stable, run 3-step smoke before any larger training."
                    if training_smoke_passed
                    else "Fix training stability (lower LR / higher kl_coef) before retrying."
                ),
                "zero_std_group_rate": spread_stats["zero_std_group_rate"],
                "retrieval_quality_spread_group_rate": spread_stats[
                    "retrieval_quality_spread_group_rate"
                ],
                "dryrun_warning": TINY_TRAIN_WARNING,
            }

            if step_metrics:
                last = step_metrics[-1]
                summary.update(
                    {
                        "policy_loss": last.get("policy_loss"),
                        "mean_kl": last.get("mean_kl"),
                        "signed_logprob_gap_mean": last.get("signed_logprob_gap_mean"),
                        "approx_kl_nonnegative": last.get("approx_kl_nonnegative"),
                        "clipfrac": last.get("clipfrac"),
                        "grad_norm": last.get("grad_norm"),
                        "mean_reward": last.get("mean_reward"),
                        "reward_std": last.get("reward_std"),
                    }
                )
                if last.get("hard_stop"):
                    summary["hard_stop"] = True
                    summary["hard_stop_reason"] = last.get("hard_stop_reason")
            if stop_snapshot_path:
                summary["stop_snapshot_path"] = stop_snapshot_path

            config = {
                "model_path": self.model_path,
                "tokenizer_path": self.tokenizer_path,
                "rollout_path": str(rollout_path),
                "shaped_reward_path": str(shaped_reward_path),
                **{k: summary[k] for k in summary if k not in {"checkpoint_path", "dryrun_warning"}},
                "max_prompt_length": max_prompt_length,
                "max_response_length": max_response_length,
                "max_total_length": max_total_length,
                "ppo_mini_batch_size": self.ppo_mini_batch_size,
                "micro_batch_size": self.micro_batch_size,
            }

            (out / config_filename).write_text(
                "\n".join(f"{k}: {v}" for k, v in config.items()) + "\n",
                encoding="utf-8",
            )
            (out / summary_filename).write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            manifest = {
                "checkpoint_label": CHECKPOINT_LABEL,
                "checkpoint_promoted": False,
                "save_steps": sorted(save_step_set) if save_step_set is not None else "all",
                "checkpoints": saved_checkpoints
                if saved_checkpoints
                else [
                    {
                        "step": m["step"],
                        "path": str(out / "checkpoints" / f"{checkpoint_prefix}_{m['step']}"),
                        "label": CHECKPOINT_LABEL,
                        "optimizer_step_called": m.get("optimizer_step_called", False),
                    }
                    for m in step_metrics
                    if m.get("optimizer_step_called") and m.get("checkpoint_saved")
                ],
            }
            manifest["checkpoint_prefix"] = checkpoint_prefix
            (out / "checkpoint_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            if not training_smoke_passed:
                self._write_failure_report(out, summary, step_metrics)

            return {
                "summary": summary,
                "step_metrics": step_metrics,
                "merged_records": merged_records,
                "checkpoint_dir": checkpoint_dir,
                "saved_checkpoints": saved_checkpoints,
                "adv_meta": adv_meta,
            }

        except Exception as exc:
            self._write_failure_report(
                out,
                {"error": str(exc), "traceback": traceback.format_exc()},
                step_metrics,
            )
            raise
        finally:
            self.model = None
            self.optimizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _write_failure_report(
        output_dir: Path,
        summary: Dict[str, Any],
        step_metrics: List[Dict[str, Any]],
    ) -> None:
        lines = [
            "# Phase 2.1 Tiny GRPO Smoke Training Failure Report",
            "",
            "## Summary",
            "",
            "```json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Step Metrics",
            "",
        ]
        for metric in step_metrics:
            lines.append(f"- step {metric.get('step')}: {metric}")
        (output_dir / "failure_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_tiny_grpo_smoke_report(
    summary: Dict[str, Any],
    clean_summary: Optional[Dict[str, Any]] = None,
    preflight_summary: Optional[Dict[str, Any]] = None,
    post_eval_summary: Optional[Dict[str, Any]] = None,
) -> str:
    lines = [
        "# Phase 2.1 Tiny GRPO Smoke Training Report",
        "",
        "## Mode",
        "",
        f"- phase: `{summary.get('phase')}`",
        f"- mode: `{summary.get('mode')}`",
        f"- reward_candidate: `{summary.get('reward_candidate')}`",
        f"- checkpoint_label: **{summary.get('checkpoint_label')}**",
        "",
    ]

    if clean_summary:
        lines.extend(
            [
                "## Clean Set",
                "",
                f"- num_groups: **{clean_summary.get('num_groups')}**",
                f"- excluded_group_ids: `{clean_summary.get('excluded_group_ids')}`",
                f"- num_replacements_added: **{clean_summary.get('num_replacements_added')}**",
                "",
            ]
        )

    if preflight_summary:
        lines.extend(
            [
                "## Preflight Gate",
                "",
                f"- v2_gate_passed: **{preflight_summary.get('v2_gate_passed')}**",
                f"- v2_retrieval_quality_spread_group_rate: **{preflight_summary.get('v2_retrieval_quality_spread_group_rate')}**",
                f"- v2_zero_std_group_rate: **{preflight_summary.get('v2_zero_std_group_rate')}**",
                f"- v2_strategy_collapse_count: **{preflight_summary.get('v2_strategy_collapse_count')}**",
                "",
            ]
        )

    lines.extend(
        [
            "## Training",
            "",
            f"- max_update_steps: **{summary.get('max_update_steps')}**",
            f"- actual_update_steps: **{summary.get('actual_update_steps')}**",
            f"- optimizer_step_called: **{summary.get('optimizer_step_called')}**",
            f"- training_smoke_passed: **{summary.get('training_smoke_passed')}**",
            f"- nan_detected: **{summary.get('nan_detected')}**",
            f"- oom_detected: **{summary.get('oom_detected')}**",
            f"- kl_exploded: **{summary.get('kl_exploded')}**",
            f"- policy_loss: **{summary.get('policy_loss')}**",
            f"- mean_kl: **{summary.get('mean_kl')}**",
            f"- clipfrac: **{summary.get('clipfrac')}**",
            f"- grad_norm: **{summary.get('grad_norm')}**",
            f"- checkpoint_saved: **{summary.get('checkpoint_saved')}**",
            f"- checkpoint_promoted: **{summary.get('checkpoint_promoted')}**",
            "",
        ]
    )

    if post_eval_summary:
        lines.extend(
            [
                "## Post-Train Eval",
                "",
                f"- parse_success_rate: **{post_eval_summary.get('parse_success_rate')}**",
                f"- finish_rate: **{post_eval_summary.get('finish_rate')}**",
                f"- invalid_action_rate: **{post_eval_summary.get('invalid_action_rate')}**",
                f"- mean_reward_largek_mix_1000: **{post_eval_summary.get('mean_reward_largek_mix_1000')}**",
                "",
            ]
        )

    lines.extend(
        [
            summary.get("dryrun_warning", TINY_TRAIN_WARNING),
            "",
            "## Next Steps",
            "",
            summary.get("next_recommendation", ""),
        ]
    )
    return "\n".join(lines) + "\n"
