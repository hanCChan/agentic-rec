"""
Phase 1.14 Reference LogProb / KL Dry-Run.

Computes actor/current, old, and reference logprobs plus token-level KL diagnostics
under torch.no_grad(). Does NOT train or connect to VERL trainer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from src.agents.actor_logprob_dryrun import ActorLogProbDryRun
from src.agents.actor_logprob_mock import _get_batch

KL_DRYRUN_WARNING = (
    "KL tensors were computed for validation only; "
    "no training or optimizer.step was performed."
)


class ReferenceKLDryRun:
    """
    Phase 1.14 reference logprob / KL dry-run.

    Verifies actor/old/ref logprob relationships and KL masking.
    Does NOT train, does NOT call optimizer.step.
    """

    def __init__(
        self,
        actor_model_path: str,
        ref_model_path: Optional[str] = None,
        dtype: str = "bfloat16",
        device: str = "cuda",
        kl_coef: float = 0.01,
        shared_ref: bool = True,
        max_batch_size: int = 4,
    ):
        self.actor_model_path = actor_model_path
        self.ref_model_path = ref_model_path or actor_model_path
        self.dtype = dtype
        self.device = device
        self.kl_coef = kl_coef
        self.shared_ref = shared_ref
        self.max_batch_size = max_batch_size

        self._actor_runner = ActorLogProbDryRun(
            model_path=actor_model_path,
            dtype=dtype,
            device=device,
            max_batch_size=max_batch_size,
        )
        self._ref_runner: Optional[ActorLogProbDryRun] = None

    def load_actor(self) -> None:
        self._actor_runner.load_model()

    def load_ref(self) -> None:
        if self.shared_ref:
            return
        self._ref_runner = ActorLogProbDryRun(
            model_path=self.ref_model_path,
            dtype=self.dtype,
            device=self.device,
            max_batch_size=self.max_batch_size,
        )
        self._ref_runner.load_model()

    def compute_actor_log_probs(self, data_proto: Any) -> Dict[str, Any]:
        """Compute current actor logprobs; old_log_probs clone for dry-run baseline."""
        output = self._actor_runner.compute_response_log_probs(data_proto)
        actor_log_probs = output["real_old_log_probs"]
        old_log_probs = actor_log_probs.clone()
        return {
            "actor_log_probs": actor_log_probs,
            "old_log_probs": old_log_probs,
        }

    def compute_ref_log_probs(
        self,
        data_proto: Any,
        actor_output: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute reference logprobs; shared-ref clones actor output."""
        if self.shared_ref:
            if actor_output is None:
                raise ValueError("actor_output required when shared_ref=True")
            ref_log_probs = actor_output["actor_log_probs"].clone()
            return {"ref_log_probs": ref_log_probs, "shared_ref": True}

        if self._ref_runner is None:
            raise RuntimeError("ref model not loaded; call load_ref() first")
        output = self._ref_runner.compute_response_log_probs(data_proto)
        return {"ref_log_probs": output["real_old_log_probs"], "shared_ref": False}

    def compute_kl(
        self,
        data_proto: Any,
        actor_output: Dict[str, Any],
        ref_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute token KL, ratio diagnostics, and KL-adjusted token rewards."""
        batch = _get_batch(data_proto)
        response_attention_mask = batch["response_attention_mask"].float()
        token_level_rewards = batch["token_level_rewards"].float()

        actor_log_probs = actor_output["actor_log_probs"]
        old_log_probs = actor_output["old_log_probs"]
        ref_log_probs = ref_output["ref_log_probs"]

        token_kl = actor_log_probs - ref_log_probs
        masked_token_kl = token_kl * response_attention_mask
        kl_penalty = self.kl_coef * masked_token_kl
        kl_adjusted_token_rewards = token_level_rewards - kl_penalty

        logprob_delta = actor_log_probs - old_log_probs
        ratio = torch.exp(logprob_delta)
        masked_ratio = ratio * response_attention_mask

        return {
            "actor_log_probs": actor_log_probs,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "token_kl": token_kl,
            "masked_token_kl": masked_token_kl,
            "kl_penalty": kl_penalty,
            "kl_adjusted_token_rewards": kl_adjusted_token_rewards,
            "logprob_delta": logprob_delta,
            "ratio": ratio,
            "masked_ratio": masked_ratio,
            "kl_coef": self.kl_coef,
            "shared_ref": self.shared_ref,
            "is_dryrun": True,
        }

    def check_kl_output(self, data_proto: Any, kl_output: Dict[str, Any]) -> Dict[str, Any]:
        """Validate KL tensor shapes, finiteness, and padding zeros."""
        batch = _get_batch(data_proto)
        responses = batch["responses"]
        mask = batch["response_attention_mask"].bool()

        actor_log_probs = kl_output["actor_log_probs"]
        old_log_probs = kl_output["old_log_probs"]
        ref_log_probs = kl_output["ref_log_probs"]
        token_kl = kl_output["token_kl"]
        kl_penalty = kl_output["kl_penalty"]
        kl_adjusted_token_rewards = kl_output["kl_adjusted_token_rewards"]
        ratio = kl_output["ratio"]

        assert actor_log_probs.shape == responses.shape
        assert old_log_probs.shape == responses.shape
        assert ref_log_probs.shape == responses.shape
        assert token_kl.shape == responses.shape
        assert kl_penalty.shape == responses.shape
        assert kl_adjusted_token_rewards.shape == responses.shape
        assert ratio.shape == responses.shape

        finite_actor = torch.isfinite(actor_log_probs[mask]).all().item()
        finite_ref = torch.isfinite(ref_log_probs[mask]).all().item()
        finite_kl = torch.isfinite(token_kl[mask]).all().item()
        finite_ratio = torch.isfinite(ratio[mask]).all().item()

        padding_zero = (
            (actor_log_probs[~mask] == 0).all().item()
            and (ref_log_probs[~mask] == 0).all().item()
            and (kl_penalty[~mask] == 0).all().item()
        )

        valid_kl = token_kl[mask]
        valid_ratio = ratio[mask]
        valid_actor = actor_log_probs[mask]
        valid_ref = ref_log_probs[mask]

        stats = {
            "mean_valid_actor_logprob": float(valid_actor.mean().item()) if valid_actor.numel() else 0.0,
            "mean_valid_ref_logprob": float(valid_ref.mean().item()) if valid_ref.numel() else 0.0,
            "mean_valid_kl": float(valid_kl.mean().item()) if valid_kl.numel() else 0.0,
            "mean_valid_abs_kl": float(valid_kl.abs().mean().item()) if valid_kl.numel() else 0.0,
            "mean_valid_ratio": float(valid_ratio.mean().item()) if valid_ratio.numel() else 0.0,
            "min_valid_ratio": float(valid_ratio.min().item()) if valid_ratio.numel() else 0.0,
            "max_valid_ratio": float(valid_ratio.max().item()) if valid_ratio.numel() else 0.0,
        }

        passed = all([finite_actor, finite_ref, finite_kl, finite_ratio, padding_zero])

        return {
            "kl_check_passed": bool(passed),
            "actor_log_probs_shape": list(actor_log_probs.shape),
            "old_log_probs_shape": list(old_log_probs.shape),
            "ref_log_probs_shape": list(ref_log_probs.shape),
            "token_kl_shape": list(token_kl.shape),
            "ratio_shape": list(ratio.shape),
            "finite_actor_log_probs": bool(finite_actor),
            "finite_ref_log_probs": bool(finite_ref),
            "finite_token_kl": bool(finite_kl),
            "finite_ratio": bool(finite_ratio),
            "padding_zero_check": bool(padding_zero),
            **stats,
        }
