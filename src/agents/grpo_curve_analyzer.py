"""
Phase 2.3: GRPO curve analyzer for multi-step smoke training.

Analyzes loss/KL/grad norm trends and classifies stability risks.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _series(metrics: List[Dict[str, Any]], key: str) -> List[float]:
    out: List[float] = []
    for row in metrics:
        val = row.get(key)
        if val is not None:
            try:
                out.append(float(val))
            except (TypeError, ValueError):
                pass
    return out


def _trend(values: List[float]) -> str:
    if len(values) < 2:
        return "flat"
    delta = values[-1] - values[0]
    span = max(abs(values[0]), abs(values[-1]), 1e-8)
    rel = abs(delta) / span
    if rel < 0.05:
        return "flat"
    return "increasing" if delta > 0 else "decreasing"


def _risk_class(
    *,
    max_approx_kl: float,
    max_grad_norm: float,
    approx_kl_trend: str,
    grad_trend: str,
    reward_drop: float,
    json_ok: bool,
    early_stop: bool,
) -> str:
    if not json_ok:
        return "format_risk"
    if early_stop:
        return "kl_risk" if max_approx_kl > 0.2 else "grad_risk"
    if max_approx_kl > 0.15 or (approx_kl_trend == "increasing" and max_approx_kl > 0.05):
        return "kl_risk"
    if max_grad_norm > 5.0 or (grad_trend == "increasing" and max_grad_norm > 1.0):
        return "grad_risk"
    if reward_drop > 0.2:
        return "reward_collapse"
    if approx_kl_trend == "increasing" or grad_trend == "increasing":
        return "mild_drift"
    return "stable"


class GRPOCurveAnalyzer:
    """
    Analyze 10-step GRPO smoke curves.

    Checks loss stability, KL trend, grad norm trend, clipfrac trend,
    JSON stability, and reward collapse.
    """

    def __init__(
        self,
        max_approx_kl: float = 0.2,
        max_grad_norm: float = 10.0,
        reward_drop_threshold: float = 0.2,
    ):
        self.max_approx_kl = max_approx_kl
        self.max_grad_norm = max_grad_norm
        self.reward_drop_threshold = reward_drop_threshold

    def load_metrics(self, metrics_path: str | Path) -> List[Dict[str, Any]]:
        return _load_jsonl(metrics_path)

    def analyze_trends(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        policy_loss = _series(metrics, "policy_loss")
        approx_kl = _series(metrics, "approx_kl_nonnegative")
        grad_norm = _series(metrics, "grad_norm")
        clipfrac = _series(metrics, "clipfrac")
        rewards = _series(metrics, "mean_reward")
        signed_gap = _series(metrics, "signed_logprob_gap_mean")

        early_stop = any(bool(m.get("hard_stop")) for m in metrics)
        json_ok = all(m.get("json_format_ok", True) is not False for m in metrics)

        reward_drop = 0.0
        if len(rewards) >= 2 and rewards[0] > 0:
            reward_drop = max(0.0, (rewards[0] - rewards[-1]) / rewards[0])

        approx_kl_trend = _trend(approx_kl)
        grad_trend = _trend(grad_norm)
        loss_trend = _trend(policy_loss)
        clipfrac_trend = _trend(clipfrac)
        reward_trend = _trend(rewards) if rewards else "flat"

        max_approx = max(approx_kl) if approx_kl else 0.0
        max_grad = max(grad_norm) if grad_norm else 0.0
        max_signed = max(abs(v) for v in signed_gap) if signed_gap else 0.0
        max_clip = max(clipfrac) if clipfrac else 0.0

        stability_class = _risk_class(
            max_approx_kl=max_approx,
            max_grad_norm=max_grad,
            approx_kl_trend=approx_kl_trend,
            grad_trend=grad_trend,
            reward_drop=reward_drop,
            json_ok=json_ok,
            early_stop=early_stop,
        )

        return {
            "num_steps": len(metrics),
            "early_stop_triggered": early_stop,
            "policy_loss_trend": loss_trend,
            "approx_kl_trend": approx_kl_trend,
            "grad_norm_trend": grad_trend,
            "clipfrac_trend": clipfrac_trend,
            "reward_trend": reward_trend,
            "json_stability": json_ok,
            "max_approx_kl_nonnegative": max_approx,
            "max_grad_norm": max_grad,
            "max_abs_signed_logprob_gap": max_signed,
            "max_clipfrac": max_clip,
            "mean_policy_loss": mean(policy_loss) if policy_loss else None,
            "mean_approx_kl_nonnegative": mean(approx_kl) if approx_kl else None,
            "mean_grad_norm": mean(grad_norm) if grad_norm else None,
            "reward_drop_ratio": reward_drop,
            "stability_class": stability_class,
            "series": {
                "policy_loss": policy_loss,
                "approx_kl_nonnegative": approx_kl,
                "grad_norm": grad_norm,
                "clipfrac": clipfrac,
                "mean_reward": rewards,
                "signed_logprob_gap_mean": signed_gap,
            },
        }

    def compare_lr_runs(self, run_a: Dict[str, Any], run_b: Dict[str, Any]) -> Dict[str, Any]:
        a_lr = run_a.get("learning_rate")
        b_lr = run_b.get("learning_rate")
        a_trend = run_a.get("curve_analysis", {})
        b_trend = run_b.get("curve_analysis", {})

        def _better(run: Dict[str, Any]) -> bool:
            cls = run.get("stability_class", run.get("curve_analysis", {}).get("stability_class"))
            return bool(run.get("ten_step_smoke_passed", False)) and cls in {
                "stable",
                "mild_drift",
            }

        winner = None
        if _better(run_a) and not _better(run_b):
            winner = a_lr
        elif _better(run_b) and not _better(run_a):
            winner = b_lr
        elif run_a.get("max_approx_kl_nonnegative", 999) < run_b.get("max_approx_kl_nonnegative", 999):
            winner = a_lr
        elif run_b.get("max_approx_kl_nonnegative", 999) < run_a.get("max_approx_kl_nonnegative", 999):
            winner = b_lr

        return {
            "run_a_learning_rate": a_lr,
            "run_b_learning_rate": b_lr,
            "run_a_stability_class": a_trend.get("stability_class"),
            "run_b_stability_class": b_trend.get("stability_class"),
            "run_a_max_approx_kl": run_a.get("max_approx_kl_nonnegative"),
            "run_b_max_approx_kl": run_b.get("max_approx_kl_nonnegative"),
            "run_a_max_grad_norm": run_a.get("max_grad_norm"),
            "run_b_max_grad_norm": run_b.get("max_grad_norm"),
            "run_a_ten_step_smoke_passed": run_a.get("ten_step_smoke_passed"),
            "run_b_ten_step_smoke_passed": run_b.get("ten_step_smoke_passed"),
            "recommended_learning_rate": winner,
            "both_stable": _better(run_a) and _better(run_b),
        }

    def recommend_next_step(self, trend_report: Dict[str, Any]) -> Dict[str, Any]:
        cls = trend_report.get("stability_class", "stable")
        early = trend_report.get("early_stop_triggered", False)

        if cls == "stable" and not early:
            return {
                "next_phase": "2.4",
                "action": "Write 50-step pilot GRPO training plan; do not launch yet.",
                "safe_for_larger_training": False,
            }
        if cls in {"mild_drift", "kl_risk"}:
            return {
                "next_phase": "2.3b",
                "action": "Try conservative LR=5e-7 or higher kl_coef before scaling steps.",
                "safe_for_larger_training": False,
            }
        if cls == "grad_risk":
            return {
                "next_phase": "2.3c",
                "action": "Training stability fix: lower LR, stronger grad clip, check advantage scale.",
                "safe_for_larger_training": False,
            }
        if cls == "format_risk":
            return {
                "next_phase": "2.3c",
                "action": "Format preservation fix: lower LR, stronger KL, optional JSON SFT mix.",
                "safe_for_larger_training": False,
            }
        if cls == "reward_collapse":
            return {
                "next_phase": "2.3c",
                "action": "Inspect reward path and advantage normalization before longer training.",
                "safe_for_larger_training": False,
            }
        return {
            "next_phase": "2.3c",
            "action": "Review curve_analysis and stability monitor logs.",
            "safe_for_larger_training": False,
        }
