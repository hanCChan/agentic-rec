"""
Phase 1.18c reward shaping proposal dry-run.

Evaluates candidate reward formulas offline using existing multi-sample rollout
records and diagnostic metrics. Does NOT train or modify official reward.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

from src.agents.reward_sensitivity_diagnostics import EPS, decompose_trajectory_reward
from src.reward.process_reward import RewardConfig

CANDIDATE_SPECS: List[Dict[str, Any]] = [
    {
        "candidate_name": "reward_current",
        "field": "shaped_reward_current",
        "diagnostic_only": False,
        "description": "Current total_reward from Phase 1.17 rollout (baseline).",
    },
    {
        "candidate_name": "reward_quality_only",
        "field": "shaped_reward_quality_only",
        "diagnostic_only": False,
        "description": "Pure retrieval quality: NDCG@10 + 0.2*Recall@50 + 0.1*MRR@50.",
    },
    {
        "candidate_name": "reward_quality_best_step",
        "field": "shaped_reward_quality_best_step",
        "diagnostic_only": False,
        "description": "Retrieval quality plus 0.5 * best_step_delta_ndcg.",
    },
    {
        "candidate_name": "reward_penalty_decoupled",
        "field": "shaped_reward_penalty_decoupled",
        "diagnostic_only": False,
        "description": "Retrieval quality only; penalties tracked separately for GRPO.",
    },
    {
        "candidate_name": "reward_hit_depth",
        "field": "shaped_reward_hit_depth",
        "diagnostic_only": False,
        "description": "Soft rank reward: 1/log2(first_relevant_rank+1) on final query topK.",
    },
    {
        "candidate_name": "reward_overlap_diagnostic",
        "field": "shaped_reward_overlap_diagnostic",
        "diagnostic_only": True,
        "description": "Diagnostic only: retrieval quality + diversity term from top50 overlap.",
    },
]


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


def _set_jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def first_relevant_rank(retrieved: List[str], targets: List[str]) -> Optional[int]:
    target_set = set(targets)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in target_set:
            return rank
    return None


def compute_group_advantages(rewards: List[float], eps: float = EPS) -> List[float]:
    if not rewards:
        return []
    mu = mean(rewards)
    std = _std(rewards)
    return [(r - mu) / (std + eps) for r in rewards]


def retrieval_quality_score(metrics: Dict[str, Any]) -> float:
    return (
        1.0 * float(metrics.get("ndcg@10", 0.0))
        + 0.2 * float(metrics.get("recall@50", 0.0))
        + 0.1 * float(metrics.get("mrr@50", 0.0))
    )


class RewardShapingDryRun:
    """
    Phase 1.18c reward shaping proposal dry-run.

    This class evaluates candidate reward formulas offline using existing
    multi-sample rollout records and diagnostic metrics.

    It does NOT train and does NOT modify the official environment reward.
    """

    def __init__(self, eps: float = EPS):
        self.eps = eps
        self.candidate_specs = CANDIDATE_SPECS

    def load_inputs(
        self,
        rollout_path: str | Path,
        group_sensitivity_path: str | Path,
        query_metrics_path: str | Path,
    ) -> Dict[str, Any]:
        return {
            "rollout_records": _load_jsonl(rollout_path),
            "group_sensitivity": _load_jsonl(group_sensitivity_path),
            "query_metrics": _load_jsonl(query_metrics_path),
        }

    def _build_query_metric_lookup(self, query_metrics: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
        lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for row in query_metrics:
            key = (row["sample_id"], row["query_type"])
            lookup[key] = row
        return lookup

    def compute_candidate_rewards(
        self,
        records: List[Dict[str, Any]],
        query_metrics: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        lookup = self._build_query_metric_lookup(query_metrics)
        shaped: List[Dict[str, Any]] = []

        for record in records:
            sample_id = record.get("sample_id")
            traj = record["trajectory"]
            target_items = traj.get("target_items", [])
            decomposition = decompose_trajectory_reward(traj)

            final_item = lookup.get((sample_id, "final_query"), {})
            original_item = lookup.get((sample_id, "original_query"), {})
            final_metrics = final_item.get("metrics", {})
            original_metrics = original_item.get("metrics", {})

            final_retrieved = final_item.get("retrieved_doc_ids") or []
            original_retrieved = original_item.get("retrieved_doc_ids") or []

            quality = retrieval_quality_score(final_metrics)
            original_ndcg10 = float(original_metrics.get("ndcg@10", 0.0))

            step_ndcg_values = [
                float(s.get("ndcg_at_10") or 0.0)
                for s in traj.get("steps", [])
                if (s.get("action") or {}).get("tool") == "bm25_search"
            ]
            if step_ndcg_values:
                best_step_delta = max(step_ndcg_values) - original_ndcg10
                best_step_delta_available = True
            else:
                best_ndcg = float(traj.get("best_ndcg_at_10", 0.0))
                if best_ndcg or original_ndcg10:
                    best_step_delta = best_ndcg - original_ndcg10
                    best_step_delta_available = True
                else:
                    best_step_delta = 0.0
                    best_step_delta_available = False

            rank = first_relevant_rank(final_retrieved, target_items) if final_retrieved else None
            reward_hit_depth = 1.0 / math.log2(rank + 1) if rank is not None else 0.0

            top50_overlap = _set_jaccard(final_retrieved[:50], original_retrieved[:50])
            overlap_diversity = 1.0 - top50_overlap

            retrieval_component = decomposition["final_ndcg_component"] + decomposition["step_delta_component"]
            penalty_component = -decomposition["total_penalty"]

            row: Dict[str, Any] = {
                "group_id": record.get("group_id"),
                "group_index": record.get("group_index"),
                "sample_id": sample_id,
                "original_query": traj.get("user_query"),
                "final_query": record["extra_info"].get("final_query") or traj.get("final_query"),
                "total_reward": float(record["reward"]),
                "retrieval_quality_component": retrieval_component,
                "penalty_component": penalty_component,
                "search_cost_penalty": -decomposition["search_cost_penalty"],
                "repeat_penalty": -decomposition["repeat_penalty"],
                "invalid_penalty": -decomposition["invalid_penalty"],
                "no_final_penalty": -decomposition["no_final_penalty"],
                "retrieval_quality_score": quality,
                "best_step_delta_ndcg": best_step_delta,
                "best_step_delta_available": best_step_delta_available,
                "first_relevant_rank": rank,
                "top50_overlap_with_original_query": top50_overlap,
                "shaped_reward_current": float(record["reward"]),
                "shaped_reward_quality_only": quality,
                "shaped_reward_quality_best_step": quality + 0.5 * best_step_delta,
                "shaped_reward_penalty_decoupled": quality,
                "shaped_reward_hit_depth": reward_hit_depth,
                "shaped_reward_overlap_diagnostic": quality + 0.05 * overlap_diversity,
            }
            shaped.append(row)
        return shaped

    def analyze_candidate_by_group(self, shaped_records: List[Dict[str, Any]], candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
        field = candidate["field"]
        name = candidate["candidate_name"]
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in shaped_records:
            grouped.setdefault(row["group_id"], []).append(row)

        reports: List[Dict[str, Any]] = []
        for group_id in sorted(grouped.keys()):
            members = sorted(grouped[group_id], key=lambda r: r.get("group_index", 0))
            rewards = [float(m[field]) for m in members]
            qualities = [float(m["retrieval_quality_score"]) for m in members]
            penalties = [float(m["penalty_component"]) for m in members]

            reward_std = _std(rewards)
            quality_std = _std(qualities)
            penalty_std = _std(penalties)
            reward_spread = _spread(rewards)
            quality_spread = _spread(qualities)
            penalty_spread = _spread(penalties)

            advantages = compute_group_advantages(rewards, self.eps)
            mean_abs_advantage = mean(abs(a) for a in advantages) if advantages else 0.0

            if reward_spread > self.eps and quality_spread > self.eps:
                spread_source = "retrieval_quality"
            elif reward_spread > self.eps and penalty_spread > self.eps:
                spread_source = "penalty_only"
            elif reward_spread > self.eps:
                spread_source = "mixed"
            else:
                spread_source = "no_spread"

            reports.append(
                {
                    "candidate_name": name,
                    "group_id": group_id,
                    "group_size": len(members),
                    "candidate_rewards": rewards,
                    "candidate_reward_std": reward_std,
                    "candidate_reward_spread": reward_spread,
                    "retrieval_quality_values": qualities,
                    "retrieval_quality_spread": quality_spread,
                    "penalty_values": penalties,
                    "penalty_spread": penalty_spread,
                    "main_spread_source": spread_source,
                    "candidate_sequence_advantages": advantages,
                    "mean_abs_sequence_advantage": mean_abs_advantage,
                    "zero_std_reward": reward_std <= self.eps,
                }
            )
        return reports

    def _summarize_candidate(self, candidate: Dict[str, Any], group_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        num_groups = len(group_reports)
        zero_std_count = sum(1 for r in group_reports if r["zero_std_reward"])
        retrieval_spread_groups = sum(1 for r in group_reports if r["main_spread_source"] == "retrieval_quality")
        penalty_only_groups = sum(1 for r in group_reports if r["main_spread_source"] == "penalty_only")

        baseline_zero = None
        improved = []
        collapsed = []
        for r in group_reports:
            if r["candidate_reward_spread"] > self.eps and r["main_spread_source"] == "retrieval_quality":
                improved.append(r["group_id"])
            if r["zero_std_reward"]:
                collapsed.append(r["group_id"])

        return {
            "candidate_name": candidate["candidate_name"],
            "diagnostic_only": candidate["diagnostic_only"],
            "description": candidate["description"],
            "num_groups": num_groups,
            "zero_std_group_count": zero_std_count,
            "zero_std_group_rate": zero_std_count / num_groups if num_groups else 0.0,
            "mean_group_reward_std": mean(r["candidate_reward_std"] for r in group_reports) if group_reports else 0.0,
            "mean_reward_spread": mean(r["candidate_reward_spread"] for r in group_reports) if group_reports else 0.0,
            "mean_abs_sequence_advantage": mean(r["mean_abs_sequence_advantage"] for r in group_reports)
            if group_reports
            else 0.0,
            "retrieval_quality_spread_group_count": retrieval_spread_groups,
            "retrieval_quality_spread_rate": retrieval_spread_groups / num_groups if num_groups else 0.0,
            "penalty_only_spread_group_count": penalty_only_groups,
            "penalty_only_spread_rate": penalty_only_groups / num_groups if num_groups else 0.0,
            "groups_with_improved_spread": improved,
            "groups_still_collapsed": collapsed,
        }

    def compare_candidates(self, candidate_group_reports: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        summaries: List[Dict[str, Any]] = []
        spec_by_name = {s["candidate_name"]: s for s in self.candidate_specs}

        for name, reports in candidate_group_reports.items():
            summaries.append(self._summarize_candidate(spec_by_name[name], reports))

        non_diagnostic = [s for s in summaries if not s["diagnostic_only"]]
        safe_candidates = [
            s
            for s in non_diagnostic
            if s["zero_std_group_rate"] <= 0.5
            and s["penalty_only_spread_rate"] <= 0.2
            and s["retrieval_quality_spread_rate"] > 0.0
        ]

        best_variance = min(non_diagnostic, key=lambda s: s["zero_std_group_rate"]) if non_diagnostic else None
        best_safe = min(safe_candidates, key=lambda s: s["zero_std_group_rate"]) if safe_candidates else None

        baseline = next((s for s in summaries if s["candidate_name"] == "reward_current"), None)
        quality_only = next((s for s in summaries if s["candidate_name"] == "reward_quality_only"), None)
        best_step = next((s for s in summaries if s["candidate_name"] == "reward_quality_best_step"), None)
        overlap = next((s for s in summaries if s["candidate_name"] == "reward_overlap_diagnostic"), None)

        main_conclusion = self._derive_main_conclusion(baseline, quality_only, best_step, overlap, non_diagnostic)

        return {
            "candidates": summaries,
            "baseline_zero_std_group_rate": baseline["zero_std_group_rate"] if baseline else None,
            "best_candidate_by_reward_variance": best_variance["candidate_name"] if best_variance else None,
            "best_candidate_zero_std_group_rate": best_variance["zero_std_group_rate"] if best_variance else None,
            "best_candidate_safe_for_training": best_safe["candidate_name"] if best_safe else "none",
            "main_conclusion": main_conclusion,
        }

    def _derive_main_conclusion(
        self,
        baseline: Optional[Dict[str, Any]],
        quality_only: Optional[Dict[str, Any]],
        best_step: Optional[Dict[str, Any]],
        overlap: Optional[Dict[str, Any]],
        non_diagnostic: List[Dict[str, Any]],
    ) -> str:
        if quality_only and quality_only["zero_std_group_rate"] >= 0.8:
            msg = "Pure retrieval-quality candidates still show high zero-std rates on this smoke set. "
        else:
            msg = ""

        if best_step and baseline and best_step["zero_std_group_rate"] < baseline["zero_std_group_rate"]:
            msg += (
                f"best_step_delta candidate reduces zero_std from {baseline['zero_std_group_rate']:.2f} "
                f"to {best_step['zero_std_group_rate']:.2f}, but still not training-ready. "
            )

        if overlap and baseline and overlap["zero_std_group_rate"] + 0.15 < baseline["zero_std_group_rate"]:
            if not any(s["retrieval_quality_spread_rate"] > 0 for s in non_diagnostic):
                msg += (
                    "Overlap diagnostic improves variance without retrieval-quality spread — "
                    "qrels/metrics may be blind; do not use overlap as training reward. "
                )

        if all(s["zero_std_group_rate"] >= 0.6 for s in non_diagnostic):
            msg += "Do not train yet. Current smoke set has insufficient retrieval-quality reward variance."
        elif not msg:
            msg = "Some candidates reduce zero_std; proceed to Phase 1.19 loss dry-run with caution."

        return msg.strip()

    def recommend_reward_formula(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        summaries = {s["candidate_name"]: s for s in comparison["candidates"]}
        baseline = summaries.get("reward_current")
        quality_only = summaries.get("reward_quality_only")
        best_step = summaries.get("reward_quality_best_step")
        overlap = summaries.get("reward_overlap_diagnostic")

        recommendation = {
            "safe_for_training_candidate": comparison["best_candidate_safe_for_training"],
            "best_candidate_by_reward_variance": comparison["best_candidate_by_reward_variance"],
            "main_recommendation": comparison["main_conclusion"],
            "next_phase": "Phase 1.18d: Rollout Diversity Prompt / Search Strategy Fix",
            "rationale": [],
        }

        if comparison["best_candidate_safe_for_training"] != "none":
            recommendation["next_phase"] = "Phase 1.19: Real GRPO Loss Dry-Run with Candidate Reward"
            recommendation["rationale"].append("A non-diagnostic candidate shows acceptable group variance without penalty-only spread.")
            return recommendation

        if quality_only and quality_only["zero_std_group_rate"] >= 0.8:
            recommendation["rationale"].append(
                "Adding Recall@50/MRR@50 alone does not create group spread on this smoke set."
            )

        if best_step and baseline and best_step["zero_std_group_rate"] < baseline["zero_std_group_rate"]:
            recommendation["rationale"].append(
                "best_step_delta improves spread slightly; consider retrieval_quality + best_step_delta in future dry-runs."
            )
            recommendation["preferred_future_formula"] = (
                "retrieval_quality + 0.5 * best_step_delta_ndcg (penalties tracked separately)"
            )

        if overlap and baseline and overlap["zero_std_group_rate"] < baseline["zero_std_group_rate"]:
            if quality_only and quality_only["zero_std_group_rate"] >= 0.8:
                recommendation["next_phase"] = "Phase 1.18e: Qrels / Metric Blindness Analysis"
                recommendation["rationale"].append(
                    "Overlap diagnostic reduces zero_std but quality metrics do not — investigate qrels/metric blindness."
                )

        if all(s["zero_std_group_rate"] >= 0.6 for s in comparison["candidates"] if not s["diagnostic_only"]):
            recommendation["rationale"].append(
                "All non-diagnostic candidates keep zero_std >= 0.6; improve rollout diversity before reward changes."
            )

        return recommendation

    def run(
        self,
        rollout_path: str | Path,
        group_sensitivity_path: str | Path,
        query_metrics_path: str | Path,
    ) -> Dict[str, Any]:
        inputs = self.load_inputs(rollout_path, group_sensitivity_path, query_metrics_path)
        shaped_records = self.compute_candidate_rewards(inputs["rollout_records"], inputs["query_metrics"])

        candidate_group_reports: Dict[str, List[Dict[str, Any]]] = {}
        all_group_reports: List[Dict[str, Any]] = []
        for spec in self.candidate_specs:
            reports = self.analyze_candidate_by_group(shaped_records, spec)
            candidate_group_reports[spec["candidate_name"]] = reports
            all_group_reports.extend(reports)

        comparison = self.compare_candidates(candidate_group_reports)
        recommendation = self.recommend_reward_formula(comparison)

        return {
            "shaped_records": shaped_records,
            "candidate_group_reports": all_group_reports,
            "comparison": comparison,
            "recommendation": recommendation,
            "num_groups": len({r["group_id"] for r in shaped_records}),
            "num_rollout_records": len(shaped_records),
        }


def build_candidate_comparison_md(comparison: Dict[str, Any]) -> str:
    lines = [
        "# Phase 1.18c Candidate Reward Comparison",
        "",
        f"**Main conclusion:** {comparison['main_conclusion']}",
        "",
        "| Candidate | zero_std_rate | mean_std | mean_abs_adv | retrieval_spread_rate | penalty_only_rate | diagnostic_only |",
        "|-----------|---------------|----------|--------------|----------------------|-------------------|-----------------|",
    ]
    for s in comparison["candidates"]:
        lines.append(
            f"| `{s['candidate_name']}` | {s['zero_std_group_rate']:.2f} | "
            f"{s['mean_group_reward_std']:.4f} | {s['mean_abs_sequence_advantage']:.4f} | "
            f"{s['retrieval_quality_spread_rate']:.2f} | {s['penalty_only_spread_rate']:.2f} | "
            f"{s['diagnostic_only']} |"
        )
    lines.extend(
        [
            "",
            f"**Best by variance:** `{comparison.get('best_candidate_by_reward_variance')}` "
            f"(zero_std={comparison.get('best_candidate_zero_std_group_rate')})",
            "",
            f"**Safe for training:** `{comparison.get('best_candidate_safe_for_training')}`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def build_reward_shaping_recommendations(result: Dict[str, Any]) -> str:
    comparison = result["comparison"]
    rec = result["recommendation"]
    lines = [
        "# Phase 1.18c Reward Shaping Recommendations",
        "",
        "## Main Finding",
        "",
        comparison["main_conclusion"],
        "",
        "## Current Reward Problem",
        "",
        "Phase 1.17/1.18b showed that total_reward spread comes mainly from search penalties, not NDCG@10. "
        "GRPO would learn to avoid penalties rather than improve retrieval.",
        "",
        "## Candidate Results",
        "",
    ]
    for s in comparison["candidates"]:
        lines.extend(
            [
                f"### `{s['candidate_name']}`",
                "",
                f"- zero_std_group_rate: **{s['zero_std_group_rate']:.2f}**",
                f"- mean_group_reward_std: **{s['mean_group_reward_std']:.4f}**",
                f"- retrieval_quality_spread_rate: **{s['retrieval_quality_spread_rate']:.2f}**",
                f"- penalty_only_spread_rate: **{s['penalty_only_spread_rate']:.2f}**",
                f"- diagnostic_only: **{s['diagnostic_only']}**",
                "",
            ]
        )

    lines.extend(
        [
            "## Recommendation",
            "",
            f"- **Safe for training:** `{rec['safe_for_training_candidate']}`",
            f"- **Best variance candidate:** `{rec['best_candidate_by_reward_variance']}`",
            f"- **Next phase:** {rec['next_phase']}",
            "",
        ]
    )
    if rec.get("preferred_future_formula"):
        lines.extend(["**Preferred future formula (dry-run only):**", "", f"`{rec['preferred_future_formula']}`", ""])
    for item in rec.get("rationale", []):
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
