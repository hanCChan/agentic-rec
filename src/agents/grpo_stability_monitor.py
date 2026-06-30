"""
Phase 2.2: GRPO stability monitor for multi-step smoke training.

Tracks loss, KL diagnostics, grad norm, clipfrac, and checkpoint safety
across multiple optimizer steps. Uses non-negative KL approximations rather
than signed logprob gap alone.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class GRPOStabilityMonitor:
    """
    Phase 2.2 stability monitor for 3-step GRPO smoke training.

    Tracks loss, KL diagnostics, grad norm, clipfrac, JSON format stability,
    and checkpoint safety across multiple optimizer steps.
    """

    def __init__(
        self,
        max_signed_logprob_gap_abs: float = 5.0,
        max_approx_kl: float = 0.2,
        max_grad_norm: float = 10.0,
    ):
        self.max_signed_logprob_gap_abs = max_signed_logprob_gap_abs
        self.max_approx_kl = max_approx_kl
        self.max_grad_norm = max_grad_norm

    def check_step_metrics(self, step_metrics: dict) -> dict:
        """Validate a single step's metrics against stability thresholds."""
        policy_loss = step_metrics.get("policy_loss")
        grad_norm = step_metrics.get("grad_norm")
        approx_kl = float(step_metrics.get("approx_kl_nonnegative", 0.0) or 0.0)
        signed_gap = float(step_metrics.get("signed_logprob_gap_mean", 0.0) or 0.0)
        signed_gap_abs = float(
            step_metrics.get("signed_logprob_gap_abs_mean", abs(signed_gap)) or abs(signed_gap)
        )

        loss_finite = policy_loss is not None and _is_finite(policy_loss)
        grad_finite = grad_norm is not None and _is_finite(grad_norm)

        checks = {
            "step": step_metrics.get("step"),
            "nan_detected": bool(step_metrics.get("nan_detected", False)),
            "oom_detected": bool(step_metrics.get("oom_detected", False)),
            "loss_finite": loss_finite,
            "grad_norm_finite": grad_finite,
            "approx_kl_ok": approx_kl <= self.max_approx_kl,
            "grad_norm_ok": grad_finite and float(grad_norm) <= self.max_grad_norm,
            "signed_gap_warn": abs(signed_gap) > self.max_signed_logprob_gap_abs,
            "signed_gap_abs_ok": signed_gap_abs <= self.max_signed_logprob_gap_abs,
            "json_format_ok": step_metrics.get("json_format_ok", True) is not False,
            "optimizer_step_called": bool(step_metrics.get("optimizer_step_called", False)),
        }

        stop, reason = self.should_stop(step_metrics)
        checks["should_stop"] = stop
        checks["stop_reason"] = reason
        checks["step_passed"] = (
            not checks["nan_detected"]
            and not checks["oom_detected"]
            and checks["loss_finite"]
            and checks["grad_norm_finite"]
            and checks["approx_kl_ok"]
            and checks["grad_norm_ok"]
            and checks["json_format_ok"]
            and checks["optimizer_step_called"]
        )
        return checks

    def should_stop(self, step_metrics: dict) -> Tuple[bool, Optional[str]]:
        """Return whether training should halt after this step."""
        if step_metrics.get("nan_detected"):
            return True, "NaN detected"

        policy_loss = step_metrics.get("policy_loss")
        if policy_loss is not None and not _is_finite(policy_loss):
            return True, "policy loss is not finite"

        grad_norm = step_metrics.get("grad_norm")
        if grad_norm is not None and not _is_finite(grad_norm):
            return True, "grad_norm is not finite"

        if grad_norm is not None and _is_finite(grad_norm) and float(grad_norm) > self.max_grad_norm:
            return True, f"grad_norm {float(grad_norm):.4f} exceeds {self.max_grad_norm}"

        approx_kl = float(step_metrics.get("approx_kl_nonnegative", 0.0) or 0.0)
        if approx_kl > self.max_approx_kl:
            return True, f"approx_kl_nonnegative {approx_kl:.4f} exceeds {self.max_approx_kl}"

        if step_metrics.get("json_format_ok") is False:
            return True, "json_format_ok is false"

        if step_metrics.get("oom_detected"):
            return True, "CUDA OOM detected"

        return False, None

    def summarize(self, metrics: List[dict]) -> dict:
        """Aggregate stability summary across all completed steps."""
        if not metrics:
            return {
                "num_steps": 0,
                "all_steps_passed": False,
                "kl_exploded": False,
            }

        step_checks = [self.check_step_metrics(m) for m in metrics]
        approx_kls = [float(m.get("approx_kl_nonnegative", 0.0) or 0.0) for m in metrics]
        grad_norms = [
            float(m["grad_norm"])
            for m in metrics
            if m.get("grad_norm") is not None and _is_finite(m["grad_norm"])
        ]
        signed_gaps = [float(m.get("signed_logprob_gap_mean", 0.0) or 0.0) for m in metrics]
        signed_gaps_abs = [
            float(m.get("signed_logprob_gap_abs_mean", 0.0) or 0.0) for m in metrics
        ]
        clipfracs = [float(m.get("clipfrac", 0.0) or 0.0) for m in metrics]
        policy_losses = [
            float(m["policy_loss"])
            for m in metrics
            if m.get("policy_loss") is not None and _is_finite(m["policy_loss"])
        ]

        optimizer_steps = sum(1 for m in metrics if m.get("optimizer_step_called"))
        nan_any = any(m.get("nan_detected") for m in metrics)
        oom_any = any(m.get("oom_detected") for m in metrics)
        kl_exploded = any(k > self.max_approx_kl for k in approx_kls)
        loss_finite_all = all(
            m.get("policy_loss") is not None and _is_finite(m["policy_loss"]) for m in metrics
        )
        grad_finite_all = all(
            m.get("grad_norm") is not None and _is_finite(m["grad_norm"]) for m in metrics
        )

        max_approx_kl = max(approx_kls) if approx_kls else 0.0
        max_grad_norm = max(grad_norms) if grad_norms else 0.0
        max_abs_signed_gap = max(abs(g) for g in signed_gaps) if signed_gaps else 0.0
        max_signed_gap_abs_mean = max(signed_gaps_abs) if signed_gaps_abs else 0.0
        max_clipfrac = max(clipfracs) if clipfracs else 0.0

        warnings = []
        for m in metrics:
            gap = float(m.get("signed_logprob_gap_mean", 0.0) or 0.0)
            if abs(gap) > self.max_signed_logprob_gap_abs and (
                float(m.get("approx_kl_nonnegative", 0.0) or 0.0) <= self.max_approx_kl
            ):
                warnings.append(
                    f"step {m.get('step')}: large signed_logprob_gap_mean={gap:.4f} "
                    f"but approx_kl_nonnegative still ok"
                )

        all_steps_passed = all(c["step_passed"] for c in step_checks) and optimizer_steps == len(
            metrics
        )

        return {
            "num_steps": len(metrics),
            "optimizer_steps_called": optimizer_steps,
            "step_checks": step_checks,
            "nan_detected": nan_any,
            "oom_detected": oom_any,
            "kl_exploded": kl_exploded,
            "loss_finite_all_steps": loss_finite_all,
            "grad_norm_finite_all_steps": grad_finite_all,
            "max_approx_kl_nonnegative": max_approx_kl,
            "max_grad_norm": max_grad_norm,
            "max_abs_signed_logprob_gap": max_abs_signed_gap,
            "max_signed_logprob_gap_abs_mean": max_signed_gap_abs_mean,
            "max_clipfrac": max_clipfrac,
            "max_policy_loss": max(policy_losses) if policy_losses else None,
            "min_policy_loss": min(policy_losses) if policy_losses else None,
            "warnings": warnings,
            "all_steps_passed": all_steps_passed,
            "thresholds": {
                "max_approx_kl": self.max_approx_kl,
                "max_grad_norm": self.max_grad_norm,
                "max_signed_logprob_gap_abs": self.max_signed_logprob_gap_abs,
            },
        }


def _is_finite(value: Any) -> bool:
    try:
        import math

        v = float(value)
        return math.isfinite(v)
    except (TypeError, ValueError):
        return False
