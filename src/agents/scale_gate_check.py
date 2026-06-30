"""
Phase 1.19b: Scale gate check for large-K quality reward.

Aggregates strategy rollout, qrels/metric diagnostics, large-K reward dry-run,
and optional GRPO loss dry-run results across scales. Does NOT train.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_GATE_THRESHOLDS = {
    "retrieval_quality_spread_group_rate_min": 0.60,
    "penalty_only_spread_group_rate_max": 0.20,
    "zero_std_group_rate_max": 0.50,
    "diagnostic_only": False,
}

BASELINE_5_G4_PATHS = {
    "large_k_summary": Path("experiments/phase118f_large_k_reward_dryrun_5_g4/summary.json"),
    "large_k_comparison": Path(
        "experiments/phase118f_large_k_reward_dryrun_5_g4/large_k_candidate_comparison.json"
    ),
    "grpo_loss_summary": Path(
        "experiments/phase119_real_grpo_loss_dryrun_5_g4_largek1000/summary.json"
    ),
}


def _load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class ScaleGateCheck:
    """
    Phase 1.19b scale gate check.

    Aggregates results from strategy rollout, qrels/metric diagnostics,
    large-K reward dry-run, and optional GRPO loss dry-run.

    Does NOT train and does NOT modify rewards.
    """

    def __init__(self, gate_thresholds: Optional[Dict[str, Any]] = None):
        self.gate_thresholds = gate_thresholds or dict(DEFAULT_GATE_THRESHOLDS)

    def load_large_k_summary(self, summary_path: str | Path) -> Dict[str, Any]:
        return _load_json(summary_path)

    def load_large_k_comparison(self, comparison_path: str | Path) -> Dict[str, Any]:
        return _load_json(comparison_path)

    def _candidate_summary(
        self,
        comparison: Dict[str, Any],
        candidate_name: str,
    ) -> Dict[str, Any]:
        for item in comparison.get("candidates", []):
            if item.get("candidate_name") == candidate_name:
                return item
        raise KeyError(f"candidate `{candidate_name}` not found in comparison")

    def evaluate_scale_gate(
        self,
        large_k_summary: Dict[str, Any],
        comparison: Dict[str, Any],
        candidate_name: str = "reward_largek_mix_1000",
    ) -> Dict[str, Any]:
        candidate = self._candidate_summary(comparison, candidate_name)
        t = self.gate_thresholds

        gate_passed = (
            not candidate.get("diagnostic_only", True)
            and candidate.get("retrieval_quality_spread_group_rate", 0.0)
            >= t["retrieval_quality_spread_group_rate_min"]
            and candidate.get("penalty_only_spread_group_rate", 1.0)
            <= t["penalty_only_spread_group_rate_max"]
            and candidate.get("zero_std_group_rate", 1.0) <= t["zero_std_group_rate_max"]
        )

        return {
            "candidate_name": candidate_name,
            "zero_std_group_rate": candidate.get("zero_std_group_rate"),
            "retrieval_quality_spread_group_rate": candidate.get(
                "retrieval_quality_spread_group_rate"
            ),
            "penalty_only_spread_group_rate": candidate.get("penalty_only_spread_group_rate"),
            "mean_group_reward_std": candidate.get("mean_group_reward_std"),
            "mean_abs_sequence_advantage": candidate.get("mean_abs_sequence_advantage"),
            "gate_passed": gate_passed,
            "safe_candidate": candidate.get("safe_candidate", False),
            "comparison_gate": comparison.get("gate", {}),
            "summary_gate_passed": large_k_summary.get("gate_passed"),
            "recommended_candidate": large_k_summary.get("recommended_candidate"),
        }

    def count_failure_groups(self, qrels_group_spread_path: str | Path) -> Dict[str, int]:
        rows = _load_jsonl(qrels_group_spread_path)
        counts = {
            "bm25_retrieval_failure": 0,
            "strategy_query_too_similar": 0,
            "qrels_sparse_all_k_blind": 0,
            "small_k_blind_large_k_signal": 0,
            "metric_has_quality_signal": 0,
        }
        for row in rows:
            t = row.get("metric_blindness_type", "unknown")
            if t in counts:
                counts[t] += 1
        collapse = (
            counts["bm25_retrieval_failure"]
            + counts["strategy_query_too_similar"]
            + counts["qrels_sparse_all_k_blind"]
        )
        return {
            **counts,
            "collapse_group_count": collapse,
            "bm25_failure_group_count": counts["bm25_retrieval_failure"],
            "strategy_collapse_group_count": counts["strategy_query_too_similar"],
            "metric_blind_group_count": counts["qrels_sparse_all_k_blind"],
        }

    def build_scale_report(
        self,
        scale: int,
        group_size: int,
        rollout_summary: Dict[str, Any],
        large_k_summary: Dict[str, Any],
        comparison: Dict[str, Any],
        qrels_group_spread_path: str | Path,
        grpo_loss_summary: Optional[Dict[str, Any]] = None,
        loss_dryrun_skipped: bool = False,
        skip_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        gate_eval = self.evaluate_scale_gate(large_k_summary, comparison)
        failure_counts = self.count_failure_groups(qrels_group_spread_path)

        loss_dryrun_ran = grpo_loss_summary is not None and not loss_dryrun_skipped
        loss_check_passed = (
            grpo_loss_summary.get("loss_check_passed") if loss_dryrun_ran else None
        )

        if gate_eval["gate_passed"]:
            rec = (
                f"Gate passed at {scale}_g{group_size}. "
                "Continue checking larger scales before trainer dry-run."
            )
        else:
            rec = (
                f"Gate failed at {scale}_g{group_size}. "
                "Do not proceed to trainer dry-run; fix sample selection or strategy prompts."
            )

        return {
            "scale": f"{scale}_g{group_size}",
            "num_base_records": scale,
            "num_groups": large_k_summary.get("num_groups", scale),
            "group_size": group_size,
            "num_rollout_records": large_k_summary.get("num_rollout_records", scale * group_size),
            "strategy_rollout_passed": len(rollout_summary.get("failures", [])) == 0,
            "parse_success_rate": rollout_summary.get("parse_success_rate"),
            "invalid_action_rate": rollout_summary.get("invalid_action_rate"),
            "finish_rate": rollout_summary.get("finish_rate"),
            "candidate_name": gate_eval["candidate_name"],
            "zero_std_group_rate": gate_eval["zero_std_group_rate"],
            "retrieval_quality_spread_group_rate": gate_eval["retrieval_quality_spread_group_rate"],
            "penalty_only_spread_group_rate": gate_eval["penalty_only_spread_group_rate"],
            "mean_group_reward_std": gate_eval["mean_group_reward_std"],
            "mean_abs_sequence_advantage": gate_eval["mean_abs_sequence_advantage"],
            "gate_passed": gate_eval["gate_passed"],
            "safe_for_phase_120": False,
            "loss_dryrun_ran": loss_dryrun_ran,
            "loss_dryrun_skipped": loss_dryrun_skipped,
            "skip_reason": skip_reason,
            "loss_check_passed": loss_check_passed,
            **failure_counts,
            "recommendation": rec,
        }

    def load_baseline_5_g4(self, candidate_name: str = "reward_largek_mix_1000") -> Dict[str, Any]:
        summary_path = BASELINE_5_G4_PATHS["large_k_summary"]
        comparison_path = BASELINE_5_G4_PATHS["large_k_comparison"]
        if not summary_path.exists() or not comparison_path.exists():
            return {"scale": "5_g4", "available": False}

        summary = self.load_large_k_summary(summary_path)
        comparison = self.load_large_k_comparison(comparison_path)
        gate_eval = self.evaluate_scale_gate(summary, comparison, candidate_name)

        grpo_loss = None
        grpo_path = BASELINE_5_G4_PATHS["grpo_loss_summary"]
        if grpo_path.exists():
            grpo_loss = _load_json(grpo_path)

        return {
            "scale": "5_g4",
            "available": True,
            "num_groups": summary.get("num_groups", 5),
            "num_rollout_records": summary.get("num_rollout_records", 20),
            "zero_std_group_rate": gate_eval["zero_std_group_rate"],
            "retrieval_quality_spread_group_rate": gate_eval["retrieval_quality_spread_group_rate"],
            "penalty_only_spread_group_rate": gate_eval["penalty_only_spread_group_rate"],
            "gate_passed": gate_eval["gate_passed"],
            "loss_dryrun_ran": grpo_loss is not None,
            "loss_check_passed": grpo_loss.get("loss_check_passed") if grpo_loss else None,
        }

    def compare_scales(
        self,
        scale_reports: List[Dict[str, Any]],
        baseline_5_g4: Optional[Dict[str, Any]] = None,
        candidate_name: str = "reward_largek_mix_1000",
    ) -> Dict[str, Any]:
        baseline = baseline_5_g4 or self.load_baseline_5_g4(candidate_name)
        scale_results = [
            {
                "scale": r["scale"],
                "num_groups": r.get("num_groups"),
                "zero_std_group_rate": r.get("zero_std_group_rate"),
                "retrieval_quality_spread_group_rate": r.get(
                    "retrieval_quality_spread_group_rate"
                ),
                "penalty_only_spread_group_rate": r.get("penalty_only_spread_group_rate"),
                "gate_passed": r.get("gate_passed"),
                "loss_dryrun_ran": r.get("loss_dryrun_ran"),
                "loss_check_passed": r.get("loss_check_passed"),
                "completed": r.get("completed", True),
            }
            for r in scale_reports
        ]

        report_10 = next((r for r in scale_reports if r["scale"] == "10_g4"), None)
        report_20 = next((r for r in scale_reports if r["scale"] == "20_g4"), None)

        stable = False
        if report_10 and report_20 and report_10.get("completed", True) and report_20.get(
            "completed", True
        ):
            t = self.gate_thresholds
            stable = (
                report_10.get("gate_passed") is True
                and report_20.get("gate_passed") is True
                and report_20.get("retrieval_quality_spread_group_rate", 0.0)
                >= t["retrieval_quality_spread_group_rate_min"]
                and report_20.get("zero_std_group_rate", 1.0) <= t["zero_std_group_rate_max"]
                and report_20.get("penalty_only_spread_group_rate", 1.0)
                <= t["penalty_only_spread_group_rate_max"]
                and report_20.get("loss_check_passed") is True
            )

        return {
            "candidate_name": candidate_name,
            "scales": ["5_g4"] + [r["scale"] for r in scale_reports],
            "baseline_5_g4": baseline if baseline.get("available") else None,
            "scale_results": scale_results,
            "stable_gate_passed": stable,
            "safe_for_phase_120": stable,
        }

    def recommend_next_step(self, scale_comparison: Dict[str, Any]) -> Dict[str, Any]:
        if scale_comparison.get("stable_gate_passed"):
            return {
                "next_phase": "Phase 1.20: No-update VERL Trainer Dry-Run",
                "main_conclusion": (
                    "Large-K reward_largek_mix_1000 passed gate at 10_g4 and 20_g4 with "
                    "loss dry-run checks. Proceed to no-update trainer dry-run only."
                ),
                "do_not_train": True,
            }

        results = scale_comparison.get("scale_results", [])
        completed = [r for r in results if r.get("completed", True)]
        passed = [r for r in completed if r.get("gate_passed")]
        failed = [r for r in completed if not r.get("gate_passed")]

        if len(passed) >= 1 and len(failed) >= 1:
            conclusion = (
                "Large-K reward passed at some scales but not all. "
                "Continue Phase 1.18g/1.18h sample and prompt fixes before trainer dry-run."
            )
            next_phase = "Phase 1.18g / 1.18h, then re-run scale gate check"
        elif not passed:
            conclusion = (
                "Large-K reward failed scale gate check. "
                "Stop advancing trainer path; fix smoke set and strategy prompts."
            )
            next_phase = "Phase 1.18g / 1.18h"
        else:
            conclusion = (
                "Partial scale evidence only. Run remaining scales before Phase 1.20."
            )
            next_phase = "Complete Phase 1.19b remaining scales"

        incomplete = [r["scale"] for r in results if not r.get("completed", True)]
        if incomplete:
            conclusion += f" Incomplete scales: {', '.join(incomplete)}."

        return {
            "next_phase": next_phase,
            "main_conclusion": conclusion,
            "do_not_train": True,
        }


def build_scale_gate_comparison_md(
    comparison: Dict[str, Any],
    recommendation: Dict[str, Any],
) -> str:
    lines = [
        "# Phase 1.19b Scale Gate Comparison",
        "",
        f"**Candidate:** `{comparison['candidate_name']}`",
        f"**Stable gate passed:** {comparison['stable_gate_passed']}",
        f"**Safe for Phase 1.20:** {comparison['safe_for_phase_120']}",
        "",
        recommendation["main_conclusion"],
        "",
        "## Baseline 5_g4",
        "",
    ]
    baseline = comparison.get("baseline_5_g4")
    if baseline and baseline.get("available"):
        lines.append(
            f"- zero_std={baseline['zero_std_group_rate']:.2f}, "
            f"retrieval_spread={baseline['retrieval_quality_spread_group_rate']:.2f}, "
            f"gate_passed={baseline['gate_passed']}"
        )
    else:
        lines.append("- baseline not available")

    lines.extend(["", "## Scale Results", ""])
    lines.append(
        "| Scale | zero_std | retrieval_spread | penalty_only | gate | loss_check | completed |"
    )
    lines.append("|-------|----------|------------------|--------------|------|------------|-----------|")
    for row in comparison.get("scale_results", []):
        loss = row.get("loss_check_passed")
        loss_str = str(loss) if row.get("loss_dryrun_ran") else "skipped"
        lines.append(
            f"| {row['scale']} | {row.get('zero_std_group_rate', 0):.2f} | "
            f"{row.get('retrieval_quality_spread_group_rate', 0):.2f} | "
            f"{row.get('penalty_only_spread_group_rate', 0):.2f} | "
            f"{row.get('gate_passed')} | {loss_str} | {row.get('completed', True)} |"
        )
    return "\n".join(lines) + "\n"


def build_scale_gate_recommendations_md(
    scale_reports: List[Dict[str, Any]],
    comparison: Dict[str, Any],
    recommendation: Dict[str, Any],
) -> str:
    lines = [
        "# Phase 1.19b Scale Gate Recommendations",
        "",
        "## Main Conclusion",
        "",
        recommendation["main_conclusion"],
        "",
        f"**Next phase:** {recommendation['next_phase']}",
        "",
        "## Per-Scale Details",
        "",
    ]
    for report in scale_reports:
        lines.extend(
            [
                f"### {report['scale']}",
                "",
                f"- completed: **{report.get('completed', True)}**",
                f"- gate_passed: **{report.get('gate_passed')}**",
                f"- zero_std_group_rate: **{report.get('zero_std_group_rate', 0):.2f}**",
                f"- retrieval_quality_spread_group_rate: **"
                f"{report.get('retrieval_quality_spread_group_rate', 0):.2f}**",
                f"- penalty_only_spread_group_rate: **"
                f"{report.get('penalty_only_spread_group_rate', 0):.2f}**",
                f"- bm25_failure_group_count: **{report.get('bm25_failure_group_count', 0)}**",
                f"- strategy_collapse_group_count: **"
                f"{report.get('strategy_collapse_group_count', 0)}**",
                f"- loss_dryrun_ran: **{report.get('loss_dryrun_ran')}**",
                f"- loss_check_passed: **{report.get('loss_check_passed')}**",
                "",
            ]
        )
        if report.get("skip_reason"):
            lines.append(f"- skip_reason: {report['skip_reason']}")
            lines.append("")
    lines.append(f"**Stable gate passed:** {comparison['stable_gate_passed']}")
    return "\n".join(lines) + "\n"
