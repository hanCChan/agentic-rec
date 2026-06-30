"""
Phase 2.4: GRPO pilot monitor for 50-step controlled pilot training.

Extends GRPOStabilityMonitor with fresh-eval hard stops (format, reward collapse,
checkpoint save failure) while keeping approx_kl_nonnegative as the KL stop signal.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.agents.grpo_stability_monitor import GRPOStabilityMonitor, _is_finite

PILOT_CHECKPOINT_LABEL = "SMOKE_ONLY_DO_NOT_PROMOTE"


class GRPOPilotMonitor(GRPOStabilityMonitor):
    """
    Phase 2.4 pilot monitor for 50-step GRPO training.

    Training hard stops inherit from GRPOStabilityMonitor (approx_kl, grad_norm,
    NaN, json_format_ok). Adds eval-time stops for reward collapse and format
    degradation on fresh rollout eval.
    """

    def __init__(
        self,
        max_signed_logprob_gap_abs: float = 5.0,
        max_approx_kl: float = 0.2,
        max_grad_norm: float = 10.0,
        max_reward_drop_ratio: float = 0.30,
        min_parse_success_rate: float = 0.95,
        max_invalid_action_rate: float = 0.05,
    ):
        super().__init__(
            max_signed_logprob_gap_abs=max_signed_logprob_gap_abs,
            max_approx_kl=max_approx_kl,
            max_grad_norm=max_grad_norm,
        )
        self.max_reward_drop_ratio = max_reward_drop_ratio
        self.min_parse_success_rate = min_parse_success_rate
        self.max_invalid_action_rate = max_invalid_action_rate
        self.baseline_reward: Optional[float] = None

    def set_baseline_reward(self, reward: float) -> None:
        """Set step-0 fresh eval baseline for collapse detection."""
        self.baseline_reward = float(reward)

    def check_step_metrics(self, step_metrics: dict) -> dict:
        checks = super().check_step_metrics(step_metrics)

        if step_metrics.get("checkpoint_save_failed"):
            checks["should_stop"] = True
            checks["stop_reason"] = "checkpoint save failed"
            checks["step_passed"] = False

        return checks

    def should_stop_training(self, step_metrics: dict) -> Tuple[bool, Optional[str]]:
        """Alias for should_stop with checkpoint save failure."""
        if step_metrics.get("checkpoint_save_failed"):
            return True, "checkpoint save failed"
        return self.should_stop(step_metrics)

    def check_eval_summary(self, eval_summary: dict, *, step: int) -> dict:
        """Validate fresh rollout eval against pilot acceptance thresholds."""
        parse_rate = float(eval_summary.get("parse_success_rate", 0.0))
        invalid_raw = eval_summary.get("invalid_action_rate")
        invalid_rate = float(invalid_raw) if invalid_raw is not None else 1.0
        json_ok = eval_summary.get("json_format_ok", False) is True
        mean_reward = float(eval_summary.get("mean_reward_largek_mix_1000", 0.0))

        reward_drop_ratio = 0.0
        reward_collapse = False
        if self.baseline_reward is not None and self.baseline_reward > 0:
            reward_drop_ratio = max(0.0, (self.baseline_reward - mean_reward) / self.baseline_reward)
            reward_collapse = reward_drop_ratio > self.max_reward_drop_ratio

        parse_ok = parse_rate >= self.min_parse_success_rate
        invalid_ok = invalid_rate <= self.max_invalid_action_rate

        should_stop = False
        stop_reason: Optional[str] = None
        if not parse_ok:
            should_stop = True
            stop_reason = f"parse_success_rate {parse_rate:.4f} < {self.min_parse_success_rate}"
        elif not invalid_ok:
            should_stop = True
            stop_reason = f"invalid_action_rate {invalid_rate:.4f} > {self.max_invalid_action_rate}"
        elif not json_ok:
            should_stop = True
            stop_reason = "json_format_ok is false on fresh eval"
        elif reward_collapse:
            should_stop = True
            stop_reason = (
                f"mean_reward dropped {reward_drop_ratio:.1%} "
                f"(baseline={self.baseline_reward:.4f}, eval={mean_reward:.4f})"
            )

        return {
            "step": step,
            "parse_success_rate": parse_rate,
            "invalid_action_rate": invalid_rate,
            "json_format_ok": json_ok,
            "mean_reward_largek_mix_1000": mean_reward,
            "reward_drop_ratio": reward_drop_ratio,
            "parse_ok": parse_ok,
            "invalid_ok": invalid_ok,
            "reward_collapse": reward_collapse,
            "eval_passed": parse_ok and invalid_ok and json_ok and not reward_collapse,
            "should_stop": should_stop,
            "stop_reason": stop_reason,
        }

    def summarize_pilot(
        self,
        train_metrics: List[dict],
        eval_summaries: List[dict],
    ) -> dict:
        """Aggregate pilot-level summary across training and eval."""
        train_summary = self.summarize(train_metrics)
        eval_checks = [
            self.check_eval_summary(ev, step=int(ev.get("eval_step", ev.get("step", 0))))
            for ev in eval_summaries
        ]
        eval_passed_all = all(c.get("eval_passed", False) for c in eval_checks) if eval_checks else True
        eval_stop_any = any(c.get("should_stop", False) for c in eval_checks)

        return {
            **train_summary,
            "eval_checks": eval_checks,
            "eval_passed_all": eval_passed_all,
            "eval_early_stop": eval_stop_any,
            "baseline_reward": self.baseline_reward,
            "checkpoint_label": PILOT_CHECKPOINT_LABEL,
            "thresholds": {
                **train_summary.get("thresholds", {}),
                "max_reward_drop_ratio": self.max_reward_drop_ratio,
                "min_parse_success_rate": self.min_parse_success_rate,
                "max_invalid_action_rate": self.max_invalid_action_rate,
            },
        }
