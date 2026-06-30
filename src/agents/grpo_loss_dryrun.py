"""
Phase 1.16 GRPO/PPO-style Policy Loss Dry-Run.

Computes clipped policy loss from mock log_probs, advantages, and response masks.
Does NOT train or connect to VERL trainer.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from src.agents.grpo_advantage_mock import _get_batch

LOSS_DRYRUN_WARNING = (
    "GRPO loss was computed for validation only; "
    "no training or optimizer.step was performed."
)


class GRPOLossDryRun:
    """
    Phase 1.16 GRPO/PPO-style clipped policy loss dry-run.

    Validates loss/ratio/clip/mask data paths without training.
    """

    def __init__(
        self,
        cliprange: float = 0.2,
        kl_coef: float = 0.01,
        loss_agg_mode: str = "token-mean",
        synthetic_logprob_delta: float = 0.02,
        seed: int = 42,
    ):
        self.cliprange = cliprange
        self.kl_coef = kl_coef
        self.loss_agg_mode = loss_agg_mode
        self.synthetic_logprob_delta = synthetic_logprob_delta
        self.seed = seed

        valid_modes = {"token-mean", "seq-mean-token-sum", "seq-mean-token-mean"}
        if loss_agg_mode not in valid_modes:
            raise ValueError(f"unsupported loss_agg_mode: {loss_agg_mode}")

    def build_mock_logprob_inputs(self, data_proto: Any, adv_output: Dict[str, Any]) -> Dict[str, Any]:
        """Build controlled mock logprob tensors for loss dry-run."""
        batch = _get_batch(data_proto)
        advantages = adv_output["token_level_advantages"].float()
        response_mask = batch["response_attention_mask"].float()

        old_log_probs = torch.zeros_like(advantages)
        log_probs = old_log_probs + self.synthetic_logprob_delta
        ref_log_probs = old_log_probs.clone()

        old_log_probs = old_log_probs * response_mask
        log_probs = log_probs * response_mask
        ref_log_probs = ref_log_probs * response_mask
        advantages = advantages * response_mask

        return {
            "log_probs": log_probs,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "advantages": advantages,
            "response_mask": response_mask,
            "is_mock": True,
        }

    def _aggregate_loss(self, policy_loss_mat: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        if self.loss_agg_mode == "token-mean":
            return policy_loss_mat.sum() / response_mask.sum().clamp_min(1.0)
        if self.loss_agg_mode == "seq-mean-token-sum":
            return policy_loss_mat.sum(dim=1).mean()
        seq_token_count = response_mask.sum(dim=1).clamp_min(1.0)
        seq_loss = policy_loss_mat.sum(dim=1) / seq_token_count
        return seq_loss.mean()

    def compute_policy_loss(self, data_proto: Any, loss_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute PPO/GRPO-style clipped surrogate loss and KL diagnostics."""
        batch = _get_batch(data_proto)
        response_mask = loss_inputs["response_mask"]
        log_probs = loss_inputs["log_probs"]
        old_log_probs = loss_inputs["old_log_probs"]
        ref_log_probs = loss_inputs["ref_log_probs"]
        advantages = loss_inputs["advantages"]
        token_level_rewards = batch["token_level_rewards"].float()

        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio) * response_mask

        unclipped = ratio * advantages
        clipped_ratio = torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
        clipped = clipped_ratio * advantages

        policy_loss_mat = -torch.min(unclipped, clipped) * response_mask

        token_kl = (log_probs - ref_log_probs) * response_mask
        kl_penalty = self.kl_coef * token_kl
        kl_adjusted_token_rewards = token_level_rewards - kl_penalty

        policy_loss = self._aggregate_loss(policy_loss_mat, response_mask)

        valid_mask = response_mask.bool()
        valid_ratio = ratio[valid_mask]
        valid_kl = token_kl[valid_mask]
        valid_kl_penalty = kl_penalty[valid_mask]

        clip_mask = ((ratio < 1.0 - self.cliprange) | (ratio > 1.0 + self.cliprange)) & valid_mask
        clipfrac = clip_mask.float().sum() / response_mask.sum().clamp_min(1.0)

        return {
            "policy_loss": policy_loss,
            "policy_loss_value": float(policy_loss.item()),
            "policy_loss_mat": policy_loss_mat,
            "ratio": ratio,
            "clipped_ratio": clipped_ratio * response_mask,
            "log_ratio": log_ratio * response_mask,
            "token_kl": token_kl,
            "kl_penalty": kl_penalty,
            "kl_adjusted_token_rewards": kl_adjusted_token_rewards,
            "clipfrac": float(clipfrac.item()),
            "mean_valid_ratio": float(valid_ratio.mean().item()) if valid_ratio.numel() else 0.0,
            "min_valid_ratio": float(valid_ratio.min().item()) if valid_ratio.numel() else 0.0,
            "max_valid_ratio": float(valid_ratio.max().item()) if valid_ratio.numel() else 0.0,
            "mean_valid_kl": float(valid_kl.mean().item()) if valid_kl.numel() else 0.0,
            "mean_valid_kl_penalty": float(valid_kl_penalty.mean().item()) if valid_kl_penalty.numel() else 0.0,
            "loss_agg_mode": self.loss_agg_mode,
            "cliprange": self.cliprange,
            "kl_coef": self.kl_coef,
            "is_dryrun": True,
        }

    def check_loss_output(
        self,
        data_proto: Any,
        loss_inputs: Dict[str, Any],
        loss_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate loss finiteness, shapes, and padding zeros."""
        batch = _get_batch(data_proto)
        responses = batch["responses"]
        mask = batch["response_attention_mask"].bool()

        policy_loss = loss_output["policy_loss"]
        policy_loss_mat = loss_output["policy_loss_mat"]
        ratio = loss_output["ratio"]
        clipped_ratio = loss_output["clipped_ratio"]
        token_kl = loss_output["token_kl"]
        kl_penalty = loss_output["kl_penalty"]

        assert torch.isfinite(policy_loss).item()
        assert policy_loss_mat.shape == responses.shape
        assert ratio.shape == responses.shape
        assert clipped_ratio.shape == responses.shape
        assert token_kl.shape == responses.shape
        assert kl_penalty.shape == responses.shape

        ratio_finite = torch.isfinite(ratio[mask]).all().item()
        loss_finite = torch.isfinite(policy_loss_mat[mask]).all().item()
        kl_finite = torch.isfinite(token_kl[mask]).all().item()
        kl_penalty_finite = torch.isfinite(kl_penalty[mask]).all().item()

        padding_loss_zero = (policy_loss_mat[~mask] == 0).all().item()
        padding_ratio_zero = (ratio[~mask] == 0).all().item()
        padding_kl_zero = (kl_penalty[~mask] == 0).all().item()

        passed = all(
            [
                ratio_finite,
                loss_finite,
                kl_finite,
                kl_penalty_finite,
                padding_loss_zero,
                padding_ratio_zero,
                padding_kl_zero,
                torch.isfinite(policy_loss).item(),
            ]
        )

        return {
            "loss_check_passed": bool(passed),
            "policy_loss_finite": bool(torch.isfinite(policy_loss).item()),
            "ratio_finite": bool(ratio_finite),
            "kl_finite": bool(kl_finite),
            "padding_loss_zero": bool(padding_loss_zero),
            "padding_ratio_zero": bool(padding_ratio_zero),
            "padding_kl_zero": bool(padding_kl_zero),
            "policy_loss_mat_shape": list(policy_loss_mat.shape),
            "ratio_shape": list(ratio.shape),
            "token_kl_shape": list(token_kl.shape),
            "kl_penalty_shape": list(kl_penalty.shape),
            "clipfrac": loss_output["clipfrac"],
            "mean_valid_ratio": loss_output["mean_valid_ratio"],
            "mean_valid_kl": loss_output["mean_valid_kl"],
        }
