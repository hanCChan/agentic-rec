"""
Phase 1.18h strategy collapse diagnostics.

Compares V1 and V2 strategy-controlled rollouts for query collapse and reward spread.
Does NOT train or modify reward.
"""

from __future__ import annotations

import re
from statistics import mean, pstdev
from typing import Any, Callable, Dict, List, Optional

from src.agents.rollout_diagnostics import token_jaccard

EPS = 1e-6


def _spread(values: List[float]) -> float:
    if not values:
        return 0.0
    return max(values) - min(values)


def _std(values: List[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _trajectory_fingerprint(trajectory: Dict[str, Any]) -> str:
    parts: List[str] = []
    for step in trajectory.get("steps", []):
        action = step.get("action") or {}
        tool = action.get("tool", "")
        if tool == "bm25_search":
            parts.append(f"search:{action.get('query', '')}")
        elif tool == "final_answer":
            parts.append(f"final:{action.get('final_query', '')}")
    return "|".join(parts)


def _pairwise_average(values: List[str], metric_fn: Callable[[str, str], float]) -> float:
    if len(values) < 2:
        return 1.0 if values else 0.0
    scores = [metric_fn(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values))]
    return sum(scores) / len(scores) if scores else 0.0


class StrategyCollapseDiagnostics:
    """
    Phase 1.18h strategy collapse diagnostics.

    Compares V1 and V2 strategy-controlled rollouts and determines whether V2
    reduces same-query collapse and improves quality reward spread.

    Does NOT train and does NOT modify reward.
    """

    def __init__(self, jaccard_threshold: float = 0.90):
        self.jaccard_threshold = jaccard_threshold

    def normalize_query(self, query: str) -> str:
        text = (query or "").lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s#-]", "", text)
        return text

    def token_jaccard(self, a: str, b: str) -> float:
        return token_jaccard(a, b)

    def _final_queries(self, records: List[Dict[str, Any]]) -> List[str]:
        out: List[str] = []
        for record in records:
            q = (
                record.get("extra_info", {}).get("final_query")
                or record.get("trajectory", {}).get("final_query")
                or ""
            )
            out.append(str(q))
        return out

    def detect_group_collapse(
        self,
        records: List[Dict[str, Any]],
        *,
        metric_spread: float = 0.0,
    ) -> Dict[str, Any]:
        final_queries = self._final_queries(records)
        fingerprints = [_trajectory_fingerprint(r.get("trajectory", {})) for r in records]

        unique_final_query_count = len(set(self.normalize_query(q) for q in final_queries if q))
        unique_trajectory_count = len(set(fingerprints))
        avg_pairwise_final_query_jaccard = _pairwise_average(final_queries, self.token_jaccard)

        same_final_query_collapse = unique_final_query_count <= 1
        high_similarity_collapse = (
            avg_pairwise_final_query_jaccard >= self.jaccard_threshold and metric_spread <= EPS
        )
        trajectory_collapse = unique_trajectory_count <= 1

        collapsed = same_final_query_collapse or high_similarity_collapse or trajectory_collapse
        collapse_types: List[str] = []
        if same_final_query_collapse:
            collapse_types.append("same_final_query_collapse")
        if high_similarity_collapse:
            collapse_types.append("high_similarity_collapse")
        if trajectory_collapse:
            collapse_types.append("trajectory_collapse")

        return {
            "unique_final_query_count": unique_final_query_count,
            "unique_trajectory_count": unique_trajectory_count,
            "avg_pairwise_final_query_jaccard": avg_pairwise_final_query_jaccard,
            "same_final_query_collapse": same_final_query_collapse,
            "high_similarity_collapse": high_similarity_collapse,
            "trajectory_collapse": trajectory_collapse,
            "strategy_collapse": collapsed,
            "collapse_types": collapse_types,
            "final_queries": final_queries,
        }

    def summarize_strategy_effect(
        self,
        records: List[Dict[str, Any]],
        *,
        candidate_name: str = "reward_largek_mix_1000",
        shaped_by_sample: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            grouped.setdefault(record["group_id"], []).append(record)

        group_summaries: List[Dict[str, Any]] = []
        collapse_count = 0

        for group_id in sorted(grouped.keys()):
            members = sorted(grouped[group_id], key=lambda r: r.get("group_index", 0))
            rewards: List[float] = []
            for member in members:
                sid = member.get("sample_id")
                if shaped_by_sample and sid in shaped_by_sample:
                    rewards.append(float(shaped_by_sample[sid]))
                else:
                    rewards.append(float(member.get("reward", 0.0)))

            reward_spread = _spread(rewards)
            zero_std = _std(rewards) <= EPS
            collapse = self.detect_group_collapse(members, metric_spread=reward_spread)
            if collapse["strategy_collapse"]:
                collapse_count += 1

            original_query = members[0].get("extra_info", {}).get("original_query") or members[0][
                "trajectory"
            ].get("user_query", "")

            group_summaries.append(
                {
                    "group_id": group_id,
                    "original_query": original_query,
                    "unique_final_query_count": collapse["unique_final_query_count"],
                    "avg_pairwise_final_query_jaccard": collapse["avg_pairwise_final_query_jaccard"],
                    f"{candidate_name}_spread": reward_spread,
                    "zero_std": zero_std,
                    "strategy_collapse": collapse["strategy_collapse"],
                    "collapse_types": collapse["collapse_types"],
                    "strategy_names": [m.get("strategy_name") for m in members],
                    "final_queries": collapse["final_queries"],
                }
            )

        num_groups = len(group_summaries)
        return {
            "group_summaries": group_summaries,
            "num_groups": num_groups,
            "strategy_collapse_count": collapse_count,
            "zero_std_group_count": sum(1 for g in group_summaries if g["zero_std"]),
            "zero_std_group_rate": sum(1 for g in group_summaries if g["zero_std"]) / num_groups
            if num_groups
            else 0.0,
        }

    def compare_v1_v2(
        self,
        old_records: List[Dict[str, Any]],
        new_records: List[Dict[str, Any]],
        *,
        candidate_name: str = "reward_largek_mix_1000",
        v1_shaped: Optional[Dict[str, float]] = None,
        v2_shaped: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        old_by_group: Dict[str, List[Dict[str, Any]]] = {}
        new_by_group: Dict[str, List[Dict[str, Any]]] = {}
        for record in old_records:
            old_by_group.setdefault(record["group_id"], []).append(record)
        for record in new_records:
            new_by_group.setdefault(record["group_id"], []).append(record)

        comparisons: List[Dict[str, Any]] = []
        collapse_fixed_count = 0
        targeted_groups = sorted(set(old_by_group.keys()) & set(new_by_group.keys()))

        for group_id in targeted_groups:
            old_members = sorted(old_by_group[group_id], key=lambda r: r.get("group_index", 0))
            new_members = sorted(new_by_group[group_id], key=lambda r: r.get("group_index", 0))

            old_rewards = [
                float(v1_shaped.get(m["sample_id"], m.get("reward", 0.0)))
                if v1_shaped
                else float(m.get("reward", 0.0))
                for m in old_members
            ]
            new_rewards = [
                float(v2_shaped.get(m["sample_id"], m.get("reward", 0.0)))
                if v2_shaped
                else float(m.get("reward", 0.0))
                for m in new_members
            ]

            v1_spread = _spread(old_rewards)
            v2_spread = _spread(new_rewards)
            v1_diag = self.detect_group_collapse(old_members, metric_spread=v1_spread)
            v2_diag = self.detect_group_collapse(new_members, metric_spread=v2_spread)

            collapse_fixed = v1_diag["strategy_collapse"] and not v2_diag["strategy_collapse"]
            if collapse_fixed:
                collapse_fixed_count += 1

            original_query = old_members[0].get("extra_info", {}).get("original_query") or old_members[
                0
            ]["trajectory"].get("user_query", "")

            comparisons.append(
                {
                    "group_id": group_id,
                    "original_query": original_query,
                    "v1_unique_final_query_count": v1_diag["unique_final_query_count"],
                    "v2_unique_final_query_count": v2_diag["unique_final_query_count"],
                    "v1_avg_pairwise_final_query_jaccard": v1_diag["avg_pairwise_final_query_jaccard"],
                    "v2_avg_pairwise_final_query_jaccard": v2_diag["avg_pairwise_final_query_jaccard"],
                    f"v1_{candidate_name}_spread": v1_spread,
                    f"v2_{candidate_name}_spread": v2_spread,
                    "v1_zero_std": _std(old_rewards) <= EPS,
                    "v2_zero_std": _std(new_rewards) <= EPS,
                    "v1_strategy_collapse": v1_diag["strategy_collapse"],
                    "v2_strategy_collapse": v2_diag["strategy_collapse"],
                    "collapse_fixed": collapse_fixed,
                    "recommended_for_phase2": not v2_diag["strategy_collapse"] and v2_spread > EPS,
                    "v1_final_queries": v1_diag["final_queries"],
                    "v2_final_queries": v2_diag["final_queries"],
                }
            )

        targeted_collapse = sum(
            1 for row in comparisons if row["v1_strategy_collapse"]
        )
        return {
            "comparisons": comparisons,
            "targeted_collapse_group_count": targeted_collapse,
            "collapse_fixed_count": collapse_fixed_count,
            "targeted_fix_rate": collapse_fixed_count / targeted_collapse if targeted_collapse else 0.0,
            "collapse_fix_rate": collapse_fixed_count / targeted_collapse if targeted_collapse else 0.0,
        }
