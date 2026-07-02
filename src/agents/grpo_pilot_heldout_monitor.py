"""
Phase 2.5c: Extended GRPO pilot monitor with heldout and overfit detection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.agents.grpo_pilot_monitor import GRPOPilotMonitor, PILOT_CHECKPOINT_LABEL


class GRPOPilotHeldoutMonitor(GRPOPilotMonitor):
    """Phase 2.5c monitor with separate train/heldout baselines and overfit stops."""

    def __init__(
        self,
        max_signed_logprob_gap_abs: float = 5.0,
        max_approx_kl: float = 0.2,
        max_grad_norm: float = 10.0,
        max_train_reward_drop_ratio: float = 0.30,
        max_heldout_reward_drop_ratio: float = 0.20,
        min_parse_success_rate: float = 0.95,
        max_invalid_action_rate: float = 0.05,
    ):
        super().__init__(
            max_signed_logprob_gap_abs=max_signed_logprob_gap_abs,
            max_approx_kl=max_approx_kl,
            max_grad_norm=max_grad_norm,
            max_reward_drop_ratio=max_train_reward_drop_ratio,
            min_parse_success_rate=min_parse_success_rate,
            max_invalid_action_rate=max_invalid_action_rate,
        )
        self.max_heldout_reward_drop_ratio = max_heldout_reward_drop_ratio
        self.baseline_train_reward: Optional[float] = None
        self.baseline_heldout_reward: Optional[float] = None
        self.eval_history: List[Dict[str, Any]] = []

    def set_baselines(self, train_reward: float, heldout_reward: float) -> None:
        self.baseline_train_reward = float(train_reward)
        self.baseline_heldout_reward = float(heldout_reward)
        self.baseline_reward = float(train_reward)

    def check_eval_pair(
        self,
        train_eval: Dict[str, Any],
        heldout_eval: Dict[str, Any],
        *,
        step: int,
    ) -> Dict[str, Any]:
        train_check = self.check_eval_summary(train_eval, step=step)
        heldout_check = self._check_heldout_summary(heldout_eval, step=step)

        overfit_risk = False
        overfit_reason: Optional[str] = None
        heldout_drop_ratio = 0.0
        train_reward = float(train_eval.get("mean_reward_largek_mix_1000", 0.0))
        heldout_reward = float(heldout_eval.get("mean_reward_largek_mix_1000", 0.0))

        if (
            self.baseline_train_reward is not None
            and self.baseline_heldout_reward is not None
            and step > 0
        ):
            train_delta = train_reward - self.baseline_train_reward
            heldout_delta = heldout_reward - self.baseline_heldout_reward
            heldout_drop_ratio = max(
                0.0,
                (self.baseline_heldout_reward - heldout_reward)
                / self.baseline_heldout_reward,
            )
            if train_delta > 1e-4 and heldout_delta < -1e-4:
                overfit_risk = True
                overfit_reason = (
                    f"train reward up ({train_delta:+.4f}) but heldout reward down "
                    f"({heldout_delta:+.4f}, drop={heldout_drop_ratio:.1%}) at step {step}"
                )

        should_stop = False
        stop_reason: Optional[str] = None
        if train_check["should_stop"]:
            should_stop = True
            stop_reason = f"train: {train_check['stop_reason']}"
        elif heldout_check["should_stop"]:
            should_stop = True
            stop_reason = f"heldout: {heldout_check['stop_reason']}"
        elif overfit_risk and heldout_drop_ratio >= self.max_heldout_reward_drop_ratio * 0.5:
            should_stop = True
            stop_reason = overfit_reason

        record = {
            "step": step,
            "train_mean_reward_largek_mix_1000": train_reward,
            "heldout_mean_reward_largek_mix_1000": heldout_reward,
            "train_parse_success_rate": train_check["parse_success_rate"],
            "heldout_parse_success_rate": heldout_check["parse_success_rate"],
            "train_invalid_action_rate": train_check["invalid_action_rate"],
            "heldout_invalid_action_rate": heldout_check["invalid_action_rate"],
            "train_strategy_distribution": train_eval.get("strategy_distribution", {}),
            "heldout_strategy_distribution": heldout_eval.get("strategy_distribution", {}),
            "train_eval_passed": train_check["eval_passed"],
            "heldout_eval_passed": heldout_check["eval_passed"],
            "overfit_risk": overfit_risk,
            "overfit_reason": overfit_reason,
            "should_stop": should_stop,
            "stop_reason": stop_reason,
        }
        self.eval_history.append(record)
        return record

    def _check_heldout_summary(self, eval_summary: dict, *, step: int) -> dict:
        parse_rate = float(eval_summary.get("parse_success_rate", 0.0))
        invalid_raw = eval_summary.get("invalid_action_rate")
        invalid_rate = float(invalid_raw) if invalid_raw is not None else 1.0
        json_ok = eval_summary.get("json_format_ok", False) is True
        mean_reward = float(eval_summary.get("mean_reward_largek_mix_1000", 0.0))

        reward_drop_ratio = 0.0
        reward_collapse = False
        if self.baseline_heldout_reward is not None and self.baseline_heldout_reward > 0:
            reward_drop_ratio = max(
                0.0, (self.baseline_heldout_reward - mean_reward) / self.baseline_heldout_reward
            )
            reward_collapse = reward_drop_ratio > self.max_heldout_reward_drop_ratio

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
            stop_reason = "json_format_ok is false on heldout fresh eval"
        elif reward_collapse:
            should_stop = True
            stop_reason = (
                f"heldout mean_reward dropped {reward_drop_ratio:.1%} "
                f"(baseline={self.baseline_heldout_reward:.4f}, eval={mean_reward:.4f})"
            )

        return {
            "step": step,
            "parse_success_rate": parse_rate,
            "invalid_action_rate": invalid_rate,
            "json_format_ok": json_ok,
            "mean_reward_largek_mix_1000": mean_reward,
            "reward_drop_ratio": reward_drop_ratio,
            "eval_passed": parse_ok and invalid_ok and json_ok and not reward_collapse,
            "should_stop": should_stop,
            "stop_reason": stop_reason,
        }

    def summarize_pilot(
        self,
        train_metrics: List[dict],
        eval_summaries: List[dict],
    ) -> dict:
        base = super().summarize_pilot(train_metrics, eval_summaries)
        return {
            **base,
            "baseline_train_reward": self.baseline_train_reward,
            "baseline_heldout_reward": self.baseline_heldout_reward,
            "eval_history": self.eval_history,
            "overfit_stop_any": any(r.get("overfit_risk") and r.get("should_stop") for r in self.eval_history),
            "thresholds": {
                **base.get("thresholds", {}),
                "max_heldout_reward_drop_ratio": self.max_heldout_reward_drop_ratio,
            },
        }
