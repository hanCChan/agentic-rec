"""
Phase 1.18f: Large-K reward candidate dry-run.

Evaluates global large-K retrieval-quality reward candidates using existing
strategy-controlled rollout records and Phase 1.18e metric-by-K diagnostics.

Does NOT train or modify the official environment reward.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

from src.agents.reward_sensitivity_diagnostics import EPS, decompose_trajectory_reward
from src.agents.strategy_reward_decomposition import classify_group_spread_source

CANDIDATE_SPECS: List[Dict[str, Any]] = [
    {
        "candidate_name": "reward_current",
        "field": "reward_current",
        "diagnostic_only": False,
        "uses_penalty": True,
        "description": "Current total_reward from Phase 1.18d rollout (baseline).",
    },
    {
        "candidate_name": "reward_ndcg10",
        "field": "reward_ndcg10",
        "diagnostic_only": False,
        "uses_penalty": False,
        "description": "Quality-only NDCG@10 (reproduces small-K sparsity).",
    },
    {
        "candidate_name": "reward_ndcg100",
        "field": "reward_ndcg100",
        "diagnostic_only": False,
        "uses_penalty": False,
        "description": "Quality-only NDCG@100.",
    },
    {
        "candidate_name": "reward_ndcg1000",
        "field": "reward_ndcg1000",
        "diagnostic_only": False,
        "uses_penalty": False,
        "description": "Quality-only NDCG@1000 for small-K blind groups.",
    },
    {
        "candidate_name": "reward_largek_mix_100",
        "field": "reward_largek_mix_100",
        "diagnostic_only": False,
        "uses_penalty": False,
        "description": "Global K=100: NDCG@100 + 0.2*Recall@100 + 0.1*MRR@100.",
    },
    {
        "candidate_name": "reward_largek_mix_1000",
        "field": "reward_largek_mix_1000",
        "diagnostic_only": False,
        "uses_penalty": False,
        "description": "Global K=1000: NDCG@1000 + 0.2*Recall@1000 + 0.1*MRR@1000.",
    },
    {
        "candidate_name": "reward_best_global_k",
        "field": "reward_best_global_k",
        "diagnostic_only": True,
        "uses_penalty": False,
        "description": "Diagnostic only: max(largek_mix_100, largek_mix_1000) per record.",
    },
    {
        "candidate_name": "reward_per_group_best_k",
        "field": "reward_per_group_best_k",
        "diagnostic_only": True,
        "uses_penalty": False,
        "description": "Diagnostic only: per-group oracle K maximizing group spread.",
    },
]

GATE_THRESHOLDS = {
    "min_retrieval_quality_spread_rate": 0.6,
    "max_penalty_only_spread_rate": 0.2,
    "max_zero_std_group_rate": 0.5,
}

ORACLE_K_LIST = [10, 100, 1000]


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _spread(values: List[float]) -> float:
    if not values:
        return 0.0
    return max(values) - min(values)


def _std(values: List[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def compute_group_advantages(rewards: List[float], eps: float = EPS) -> List[float]:
    if not rewards:
        return []
    mu = mean(rewards)
    std = _std(rewards)
    return [(r - mu) / (std + eps) for r in rewards]


def mix_reward_at_k(metrics: Dict[str, Any], k: int) -> float:
    return (
        1.0 * float(metrics.get(f"ndcg@{k}", 0.0))
        + 0.2 * float(metrics.get(f"recall@{k}", 0.0))
        + 0.1 * float(metrics.get(f"mrr@{k}", 0.0))
    )


def ndcg_only_at_k(metrics: Dict[str, Any], k: int) -> float:
    return float(metrics.get(f"ndcg@{k}", 0.0))


class LargeKRewardDryRun:
    """
    Phase 1.18f large-K reward candidate dry-run.

    Evaluates global large-K retrieval-quality reward candidates using existing
    strategy-controlled rollout records and metric-by-K diagnostics.

    Does NOT train and does NOT modify the official environment reward.
    """

    def __init__(self, eps: float = EPS):
        self.eps = eps
        self.candidate_specs = CANDIDATE_SPECS

    def load_inputs(
        self,
        rollout_path: str | Path,
        metric_by_k_path: str | Path,
        group_metric_spread_path: str | Path,
        decomposition_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        return {
            "rollout_records": _load_jsonl(rollout_path),
            "metric_by_k": _load_jsonl(metric_by_k_path),
            "group_metric_spread": _load_jsonl(group_metric_spread_path),
            "decomposition_reports": _load_jsonl(decomposition_path) if decomposition_path else [],
        }

    def _metric_lookup(self, metric_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return {row["sample_id"]: row for row in metric_rows}

    def compute_large_k_candidates(
        self,
        records: List[Dict[str, Any]],
        metrics: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        lookup = self._metric_lookup(metrics)
        shaped: List[Dict[str, Any]] = []

        for record in records:
            sample_id = record.get("sample_id")
            traj = record["trajectory"]
            metric_row = lookup.get(sample_id, {})
            breakdown = decompose_trajectory_reward(traj)

            mix_100 = mix_reward_at_k(metric_row, 100)
            mix_1000 = mix_reward_at_k(metric_row, 1000)
            quality_only = breakdown["final_ndcg_component"] + breakdown["step_delta_component"]

            row: Dict[str, Any] = {
                "group_id": record.get("group_id"),
                "group_index": record.get("group_index"),
                "sample_id": sample_id,
                "strategy_name": record.get("strategy_name"),
                "original_query": traj.get("user_query"),
                "final_query": record["extra_info"].get("final_query") or traj.get("final_query"),
                "total_reward": float(record["reward"]),
                "penalty_component": -breakdown["total_penalty"],
                "quality_only_ndcg10": quality_only,
                "ndcg@10": float(metric_row.get("ndcg@10", traj.get("final_ndcg_at_10", 0.0))),
                "ndcg@100": float(metric_row.get("ndcg@100", 0.0)),
                "ndcg@1000": float(metric_row.get("ndcg@1000", 0.0)),
                "recall@100": float(metric_row.get("recall@100", 0.0)),
                "recall@1000": float(metric_row.get("recall@1000", 0.0)),
                "mrr@100": float(metric_row.get("mrr@100", 0.0)),
                "mrr@1000": float(metric_row.get("mrr@1000", 0.0)),
                "reward_current": float(record["reward"]),
                "reward_ndcg10": ndcg_only_at_k(metric_row, 10),
                "reward_ndcg100": ndcg_only_at_k(metric_row, 100),
                "reward_ndcg1000": ndcg_only_at_k(metric_row, 1000),
                "reward_largek_mix_100": mix_100,
                "reward_largek_mix_1000": mix_1000,
                "reward_best_global_k": max(mix_100, mix_1000),
                "reward_per_group_best_k": 0.0,
            }
            shaped.append(row)

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in shaped:
            grouped.setdefault(row["group_id"], []).append(row)

        for group_id, members in grouped.items():
            best_k, _ = self._select_best_k_for_group(members, lookup)
            for member in members:
                metric_row = lookup.get(member["sample_id"], {})
                member["reward_per_group_best_k"] = mix_reward_at_k(metric_row, best_k)
                member["per_group_oracle_k"] = best_k

        return shaped

    def _select_best_k_for_group(
        self,
        members: List[Dict[str, Any]],
        lookup: Dict[str, Dict[str, Any]],
    ) -> Tuple[int, float]:
        best_k = 100
        best_spread = -1.0
        for k in ORACLE_K_LIST:
            rewards = [mix_reward_at_k(lookup.get(m["sample_id"], {}), k) for m in members]
            spread = _spread(rewards)
            if spread > best_spread:
                best_spread = spread
                best_k = k
        return best_k, best_spread

    def analyze_candidate_groups(
        self,
        shaped_records: List[Dict[str, Any]],
        candidate: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        field = candidate["field"]
        name = candidate["candidate_name"]
        uses_penalty = candidate.get("uses_penalty", False)

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in shaped_records:
            grouped.setdefault(row["group_id"], []).append(row)

        reports: List[Dict[str, Any]] = []
        for group_id in sorted(grouped.keys()):
            members = sorted(grouped[group_id], key=lambda r: r.get("group_index", 0))
            rewards = [float(m[field]) for m in members]
            penalties = [float(m["penalty_component"]) for m in members]
            quality_values = [
                float(m["reward_ndcg10"]) if not uses_penalty else float(m["quality_only_ndcg10"])
                for m in members
            ]

            reward_std = _std(rewards)
            reward_spread = _spread(rewards)
            penalty_spread = _spread(penalties)
            quality_spread = _spread(quality_values)
            ndcg_spread = _spread([float(m["ndcg@10"]) for m in members])

            advantages = compute_group_advantages(rewards, self.eps)
            mean_abs_advantage = mean(abs(a) for a in advantages) if advantages else 0.0

            if uses_penalty:
                spread_source = classify_group_spread_source(
                    total_spread=reward_spread,
                    quality_spread=quality_spread,
                    penalty_spread=penalty_spread,
                    ndcg_spread=ndcg_spread,
                    eps=self.eps,
                )
            elif reward_spread > self.eps:
                spread_source = "retrieval_quality_spread"
            else:
                spread_source = "no_spread"

            reports.append(
                {
                    "candidate_name": name,
                    "diagnostic_only": candidate["diagnostic_only"],
                    "group_id": group_id,
                    "group_size": len(members),
                    "candidate_rewards": rewards,
                    "candidate_reward_std": reward_std,
                    "candidate_reward_spread": reward_spread,
                    "penalty_values": penalties,
                    "penalty_spread": penalty_spread,
                    "retrieval_quality_values": quality_values,
                    "retrieval_quality_spread": quality_spread,
                    "spread_source": spread_source,
                    "candidate_sequence_advantages": advantages,
                    "mean_abs_sequence_advantage": mean_abs_advantage,
                    "zero_std_reward": reward_std <= self.eps,
                }
            )
        return reports

    def _summarize_candidate(
        self,
        candidate: Dict[str, Any],
        group_reports: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        num_groups = len(group_reports)
        zero_std_count = sum(1 for r in group_reports if r["zero_std_reward"])
        retrieval_count = sum(
            1 for r in group_reports if r["spread_source"] == "retrieval_quality_spread"
        )
        penalty_count = sum(
            1 for r in group_reports if r["spread_source"] == "penalty_only_spread"
        )

        with_spread = [r["group_id"] for r in group_reports if r["candidate_reward_spread"] > self.eps]
        collapsed = [r["group_id"] for r in group_reports if r["zero_std_reward"]]

        summary = {
            "candidate_name": candidate["candidate_name"],
            "diagnostic_only": candidate["diagnostic_only"],
            "description": candidate["description"],
            "num_groups": num_groups,
            "zero_std_group_count": zero_std_count,
            "zero_std_group_rate": zero_std_count / num_groups if num_groups else 0.0,
            "mean_group_reward_std": mean(r["candidate_reward_std"] for r in group_reports)
            if group_reports
            else 0.0,
            "mean_reward_spread": mean(r["candidate_reward_spread"] for r in group_reports)
            if group_reports
            else 0.0,
            "mean_abs_sequence_advantage": mean(r["mean_abs_sequence_advantage"] for r in group_reports)
            if group_reports
            else 0.0,
            "retrieval_quality_spread_group_count": retrieval_count,
            "retrieval_quality_spread_group_rate": retrieval_count / num_groups if num_groups else 0.0,
            "penalty_only_spread_group_count": penalty_count,
            "penalty_only_spread_group_rate": penalty_count / num_groups if num_groups else 0.0,
            "groups_with_spread": with_spread,
            "groups_still_collapsed": collapsed,
        }
        summary["safe_candidate"] = self._is_safe_candidate(summary)
        return summary

    def _is_safe_candidate(self, summary: Dict[str, Any]) -> bool:
        if summary["diagnostic_only"]:
            return False
        return (
            summary["retrieval_quality_spread_group_rate"]
            >= GATE_THRESHOLDS["min_retrieval_quality_spread_rate"]
            and summary["penalty_only_spread_group_rate"]
            <= GATE_THRESHOLDS["max_penalty_only_spread_rate"]
            and summary["zero_std_group_rate"] <= GATE_THRESHOLDS["max_zero_std_group_rate"]
        )

    def compare_large_k_candidates(
        self,
        candidate_reports: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        summaries: List[Dict[str, Any]] = []
        spec_by_name = {s["candidate_name"]: s for s in self.candidate_specs}

        for name, reports in candidate_reports.items():
            summaries.append(self._summarize_candidate(spec_by_name[name], reports))

        non_diagnostic = [s for s in summaries if not s["diagnostic_only"]]
        safe = [s for s in non_diagnostic if s["safe_candidate"]]

        if safe:
            best_non_diagnostic = min(
                safe,
                key=lambda s: (
                    s["zero_std_group_rate"],
                    -s["retrieval_quality_spread_group_rate"],
                    -s["mean_abs_sequence_advantage"],
                ),
            )
        elif non_diagnostic:
            best_non_diagnostic = min(
                non_diagnostic,
                key=lambda s: (
                    s["zero_std_group_rate"],
                    -s["retrieval_quality_spread_group_rate"],
                    -s["mean_abs_sequence_advantage"],
                ),
            )
        else:
            best_non_diagnostic = None

        baseline_ndcg10 = next((s for s in summaries if s["candidate_name"] == "reward_ndcg10"), None)

        return {
            "candidates": summaries,
            "baseline_zero_std_group_rate": baseline_ndcg10["zero_std_group_rate"]
            if baseline_ndcg10
            else None,
            "best_non_diagnostic_candidate": best_non_diagnostic["candidate_name"]
            if best_non_diagnostic
            else None,
            "best_non_diagnostic_zero_std_group_rate": best_non_diagnostic["zero_std_group_rate"]
            if best_non_diagnostic
            else None,
            "safe_candidates": [s["candidate_name"] for s in safe],
        }

    def evaluate_gate(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        summaries = {s["candidate_name"]: s for s in comparison["candidates"]}
        passing = [
            s
            for s in comparison["candidates"]
            if not s["diagnostic_only"] and s["safe_candidate"]
        ]
        diagnostic_passing = [
            s
            for s in comparison["candidates"]
            if s["diagnostic_only"] and s["retrieval_quality_spread_group_rate"]
            >= GATE_THRESHOLDS["min_retrieval_quality_spread_rate"]
            and s["zero_std_group_rate"] <= GATE_THRESHOLDS["max_zero_std_group_rate"]
        ]

        if passing:
            best = max(
                passing,
                key=lambda s: (
                    s["retrieval_quality_spread_group_rate"],
                    -s["zero_std_group_rate"],
                    s["mean_abs_sequence_advantage"],
                ),
            )
            return {
                "gate_passed": True,
                "safe_for_phase_119": True,
                "recommended_candidate": best["candidate_name"],
                "reason": (
                    f"Deployable global candidate `{best['candidate_name']}` passes gate: "
                    f"retrieval_spread={best['retrieval_quality_spread_group_rate']:.2f}, "
                    f"penalty_only={best['penalty_only_spread_group_rate']:.2f}, "
                    f"zero_std={best['zero_std_group_rate']:.2f}."
                ),
            }

        if diagnostic_passing and not passing:
            return {
                "gate_passed": False,
                "safe_for_phase_119": False,
                "recommended_candidate": "none",
                "reason": (
                    "Only diagnostic oracle candidates passed; no deployable global reward "
                    "candidate passed."
                ),
            }

        best_nd = comparison.get("best_non_diagnostic_candidate")
        best_summary = summaries.get(best_nd) if best_nd else None
        detail = ""
        if best_summary:
            detail = (
                f" Best non-diagnostic `{best_nd}`: retrieval_spread="
                f"{best_summary['retrieval_quality_spread_group_rate']:.2f}, "
                f"zero_std={best_summary['zero_std_group_rate']:.2f}."
            )

        return {
            "gate_passed": False,
            "safe_for_phase_119": False,
            "recommended_candidate": "none",
            "reason": (
                "No deployable global large-K candidate passed gate."
                + detail
            ),
        }

    def run(
        self,
        rollout_path: str | Path,
        metric_by_k_path: str | Path,
        group_metric_spread_path: str | Path,
        decomposition_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        inputs = self.load_inputs(
            rollout_path,
            metric_by_k_path,
            group_metric_spread_path,
            decomposition_path,
        )
        shaped_records = self.compute_large_k_candidates(
            inputs["rollout_records"],
            inputs["metric_by_k"],
        )

        candidate_reports: Dict[str, List[Dict[str, Any]]] = {}
        all_reports: List[Dict[str, Any]] = []
        for spec in self.candidate_specs:
            reports = self.analyze_candidate_groups(shaped_records, spec)
            candidate_reports[spec["candidate_name"]] = reports
            all_reports.extend(reports)

        comparison = self.compare_large_k_candidates(candidate_reports)
        gate = self.evaluate_gate(comparison)

        return {
            "shaped_records": shaped_records,
            "candidate_group_reports": all_reports,
            "comparison": comparison,
            "gate": gate,
            "num_groups": len({r["group_id"] for r in shaped_records}),
            "num_rollout_records": len(shaped_records),
        }


def build_large_k_candidate_comparison_md(
    comparison: Dict[str, Any],
    gate: Dict[str, Any],
) -> str:
    lines = [
        "# Phase 1.18f Large-K Reward Candidate Comparison",
        "",
        f"**Gate passed:** {gate['gate_passed']}",
        f"**Safe for Phase 1.19:** {gate['safe_for_phase_119']}",
        f"**Recommended candidate:** `{gate['recommended_candidate']}`",
        "",
        gate["reason"],
        "",
        "| Candidate | diagnostic | zero_std | retrieval_spread | penalty_only | safe |",
        "|-----------|------------|----------|------------------|--------------|------|",
    ]
    for s in comparison["candidates"]:
        lines.append(
            f"| `{s['candidate_name']}` | {s['diagnostic_only']} | "
            f"{s['zero_std_group_rate']:.2f} | "
            f"{s['retrieval_quality_spread_group_rate']:.2f} | "
            f"{s['penalty_only_spread_group_rate']:.2f} | {s['safe_candidate']} |"
        )
    lines.extend(
        [
            "",
            f"**Best non-diagnostic:** `{comparison.get('best_non_diagnostic_candidate')}` "
            f"(zero_std={comparison.get('best_non_diagnostic_zero_std_group_rate')})",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_large_k_reward_recommendations(result: Dict[str, Any]) -> str:
    comparison = result["comparison"]
    gate = result["gate"]
    summaries = {s["candidate_name"]: s for s in comparison["candidates"]}

    lines = [
        "# Phase 1.18f Large-K Reward Recommendations",
        "",
        "## Gate Result",
        "",
        f"- gate_passed: **{gate['gate_passed']}**",
        f"- safe_for_phase_119: **{gate['safe_for_phase_119']}**",
        f"- recommended_candidate: **`{gate['recommended_candidate']}`**",
        "",
        gate["reason"],
        "",
        "## Candidate Summary",
        "",
    ]

    for name in [
        "reward_current",
        "reward_ndcg10",
        "reward_ndcg100",
        "reward_ndcg1000",
        "reward_largek_mix_100",
        "reward_largek_mix_1000",
        "reward_best_global_k",
        "reward_per_group_best_k",
    ]:
        s = summaries.get(name)
        if not s:
            continue
        lines.extend(
            [
                f"### `{name}`",
                "",
                f"- diagnostic_only: **{s['diagnostic_only']}**",
                f"- zero_std_group_rate: **{s['zero_std_group_rate']:.2f}**",
                f"- retrieval_quality_spread_group_rate: **{s['retrieval_quality_spread_group_rate']:.2f}**",
                f"- penalty_only_spread_group_rate: **{s['penalty_only_spread_group_rate']:.2f}**",
                f"- mean_abs_sequence_advantage: **{s['mean_abs_sequence_advantage']:.4f}**",
                f"- safe_candidate: **{s['safe_candidate']}**",
                f"- groups_with_spread: {s['groups_with_spread']}",
                f"- groups_still_collapsed: {s['groups_still_collapsed']}",
                "",
            ]
        )

    lines.extend(["## Decision", ""])
    if gate["gate_passed"]:
        lines.extend(
            [
                f"Proceed to **Phase 1.19: Real GRPO Loss Dry-Run** with "
                f"`{gate['recommended_candidate']}` as quality-only advantage.",
                "",
                "- Penalties must NOT enter GRPO advantage.",
                "- Penalties remain diagnostics or minimal auxiliary terms only.",
            ]
        )
    elif summaries.get("reward_best_global_k", {}).get("safe_candidate") or (
        summaries.get("reward_per_group_best_k", {}).get("retrieval_quality_spread_group_rate", 0)
        >= GATE_THRESHOLDS["min_retrieval_quality_spread_rate"]
    ):
        lines.extend(
            [
                "Only diagnostic oracle candidates show spread. Do **not** proceed to Phase 1.19.",
                "",
                "Next: **Phase 1.18g — Smoke Set Expansion / Query Selection**.",
            ]
        )
    else:
        lines.extend(
            [
                "No deployable global large-K candidate passed gate.",
                "",
                "Next steps depend on failure mode:",
                "- qrels / BM25 failure → Phase 1.18g (smoke set expansion)",
                "- strategy query collapse → Phase 1.18h (Strategy Prompt V2)",
            ]
        )

    return "\n".join(lines) + "\n"
