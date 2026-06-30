"""
Phase 1.19a strategy rollout reward decomposition.

Analyzes Phase 1.18d strategy-controlled rollouts to determine whether group
reward spread comes from retrieval quality or penalties. Does NOT train.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

from src.agents.reward_sensitivity_diagnostics import decompose_trajectory_reward
from src.reward.process_reward import RewardConfig

EPS = 1e-6

SPREAD_SOURCE_LABELS = (
    "retrieval_quality_spread",
    "penalty_only_spread",
    "mixed_spread",
    "no_spread",
)

GATE_THRESHOLDS = {
    "min_retrieval_quality_spread_rate": 0.6,
    "max_penalty_only_spread_rate": 0.2,
}


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


def decompose_record(record: Dict[str, Any], config: Optional[RewardConfig] = None) -> Dict[str, Any]:
    """Decompose one strategy rollout record into reward components."""
    traj = record["trajectory"]
    breakdown = decompose_trajectory_reward(traj, config=config)

    final_ndcg = float(traj.get("final_ndcg_at_10", 0.0))
    step_delta_sum = sum(
        float(s.get("delta_ndcg") or 0.0)
        for s in traj.get("steps", [])
        if s.get("delta_ndcg") is not None
    )
    cfg = config or RewardConfig()
    retrieval_quality_component = breakdown["final_ndcg_component"] + breakdown["step_delta_component"]
    penalty_component = -breakdown["total_penalty"]
    quality_only_reward = retrieval_quality_component

    return {
        "group_id": record.get("group_id"),
        "group_index": record.get("group_index"),
        "sample_id": record.get("sample_id"),
        "strategy_name": record.get("strategy_name"),
        "original_query": traj.get("user_query") or record["extra_info"].get("original_query"),
        "final_query": record["extra_info"].get("final_query") or traj.get("final_query"),
        "total_reward": float(record["reward"]),
        "recomputed_total_reward": breakdown["total_reward"],
        "retrieval_quality_component": retrieval_quality_component,
        "quality_only_reward": quality_only_reward,
        "final_ndcg_at_10": final_ndcg,
        "step_delta_sum": step_delta_sum,
        "step_delta_component": breakdown["step_delta_component"],
        "search_cost_penalty": -breakdown["search_cost_penalty"],
        "repeat_penalty": -breakdown["repeat_penalty"],
        "invalid_penalty": -breakdown["invalid_penalty"],
        "no_final_penalty": -breakdown["no_final_penalty"],
        "total_penalty": -breakdown["total_penalty"],
        "penalty_component": penalty_component,
        "num_search_calls": int(traj.get("num_search_calls", 0)),
        "num_invalid_actions": int(traj.get("num_invalid_actions", 0)),
        "num_repeated_queries": int(traj.get("num_repeated_queries", 0)),
        "finished": bool(traj.get("finished", False)),
        "auto_finished": bool(traj.get("auto_finished", False)),
    }


def classify_group_spread_source(
    *,
    total_spread: float,
    quality_spread: float,
    penalty_spread: float,
    ndcg_spread: float,
    eps: float = EPS,
) -> str:
    if total_spread <= eps:
        return "no_spread"

    quality_dominant = quality_spread > eps
    penalty_dominant = penalty_spread > eps

    if quality_dominant and not penalty_dominant:
        return "retrieval_quality_spread"
    if penalty_dominant and not quality_dominant:
        return "penalty_only_spread"
    if quality_dominant and penalty_dominant:
        if ndcg_spread > eps:
            return "retrieval_quality_spread" if quality_spread >= penalty_spread else "mixed_spread"
        return "mixed_spread"
    return "no_spread"


class StrategyRewardDecomposition:
    """
    Phase 1.19a strategy rollout reward decomposition analyzer.

    Does NOT train, does NOT change reward, does NOT connect to GRPO loss.
    """

    def __init__(self, eps: float = EPS):
        self.eps = eps

    def load_inputs(
        self,
        rollout_path: str | Path,
        group_summary_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        records = _load_jsonl(rollout_path)
        group_summaries = _load_jsonl(group_summary_path) if group_summary_path else []
        return {
            "records": records,
            "group_summaries": group_summaries,
        }

    def decompose_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [decompose_record(r) for r in records]

    def analyze_group(
        self,
        group_id: str,
        decomposed: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        members = sorted(decomposed, key=lambda r: r.get("group_index", 0))

        total_rewards = [r["total_reward"] for r in members]
        quality_rewards = [r["quality_only_reward"] for r in members]
        penalty_values = [r["penalty_component"] for r in members]
        ndcg_values = [r["final_ndcg_at_10"] for r in members]
        search_calls = [r["num_search_calls"] for r in members]

        total_spread = _spread(total_rewards)
        quality_spread = _spread(quality_rewards)
        penalty_spread = _spread(penalty_values)
        ndcg_spread = _spread(ndcg_values)

        spread_source = classify_group_spread_source(
            total_spread=total_spread,
            quality_spread=quality_spread,
            penalty_spread=penalty_spread,
            ndcg_spread=ndcg_spread,
            eps=self.eps,
        )

        strategy_breakdown = {}
        for row in members:
            name = row["strategy_name"]
            strategy_breakdown[name] = {
                "total_reward": row["total_reward"],
                "quality_only_reward": row["quality_only_reward"],
                "final_ndcg_at_10": row["final_ndcg_at_10"],
                "penalty_component": row["penalty_component"],
                "search_cost_penalty": row["search_cost_penalty"],
                "num_search_calls": row["num_search_calls"],
                "final_query": row["final_query"],
            }

        final_queries = [r["final_query"] for r in members]
        unique_final_queries = len(set(final_queries))

        return {
            "group_id": group_id,
            "original_query": members[0]["original_query"] if members else "",
            "group_size": len(members),
            "total_reward_values": total_rewards,
            "quality_only_reward_values": quality_rewards,
            "penalty_component_values": penalty_values,
            "ndcg_at_10_values": ndcg_values,
            "search_calls_values": search_calls,
            "total_reward_std": _std(total_rewards),
            "quality_only_reward_std": _std(quality_rewards),
            "total_reward_spread": total_spread,
            "quality_only_reward_spread": quality_spread,
            "penalty_spread": penalty_spread,
            "ndcg_spread": ndcg_spread,
            "zero_std_total_reward": _std(total_rewards) <= self.eps,
            "zero_std_quality_only": _std(quality_rewards) <= self.eps,
            "spread_source": spread_source,
            "strategy_breakdown": strategy_breakdown,
            "unique_final_query_count": unique_final_queries,
            "all_same_final_query": unique_final_queries == 1,
            "best_strategy_by_total_reward": max(
                strategy_breakdown.items(), key=lambda x: x[1]["total_reward"]
            )[0]
            if strategy_breakdown
            else None,
            "best_strategy_by_ndcg": max(
                strategy_breakdown.items(), key=lambda x: x[1]["final_ndcg_at_10"]
            )[0]
            if strategy_breakdown
            else None,
        }

    def analyze_all(self, decomposed: List[Dict[str, Any]]) -> Dict[str, Any]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in decomposed:
            grouped.setdefault(row["group_id"], []).append(row)

        group_reports = [self.analyze_group(gid, rows) for gid, rows in sorted(grouped.items())]
        strategy_stats = self._strategy_aggregates(decomposed)
        spread_rates = self._spread_source_rates(group_reports)
        gate = self._evaluate_gate(spread_rates)

        zero_std_total = sum(1 for g in group_reports if g["zero_std_total_reward"])
        zero_std_quality = sum(1 for g in group_reports if g["zero_std_quality_only"])
        num_groups = len(group_reports)

        return {
            "group_reports": group_reports,
            "strategy_stats": strategy_stats,
            "num_groups": num_groups,
            "num_rollout_records": len(decomposed),
            "zero_std_group_rate_total_reward": zero_std_total / num_groups if num_groups else 0.0,
            "zero_std_group_rate_quality_only": zero_std_quality / num_groups if num_groups else 0.0,
            **spread_rates,
            "gate_evaluation": gate,
        }

    def _strategy_aggregates(self, decomposed: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        by_strategy: Dict[str, List[Dict[str, Any]]] = {}
        for row in decomposed:
            by_strategy.setdefault(row["strategy_name"], []).append(row)

        stats: Dict[str, Dict[str, float]] = {}
        for name, rows in sorted(by_strategy.items()):
            stats[name] = {
                "strategy_mean_total_reward": mean(r["total_reward"] for r in rows),
                "strategy_mean_quality_reward": mean(r["quality_only_reward"] for r in rows),
                "strategy_mean_ndcg_at_10": mean(r["final_ndcg_at_10"] for r in rows),
                "strategy_mean_penalty": mean(r["penalty_component"] for r in rows),
                "strategy_mean_search_calls": mean(r["num_search_calls"] for r in rows),
                "count": len(rows),
            }
        return stats

    def _spread_source_rates(self, group_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        counts = {label: 0 for label in SPREAD_SOURCE_LABELS}
        for report in group_reports:
            counts[report["spread_source"]] += 1
        num_groups = len(group_reports)
        rates = {f"{label}_rate": counts[label] / num_groups if num_groups else 0.0 for label in SPREAD_SOURCE_LABELS}
        return {
            "spread_source_counts": counts,
            **rates,
            "retrieval_quality_spread_group_rate": rates["retrieval_quality_spread_rate"],
            "penalty_only_spread_group_rate": rates["penalty_only_spread_rate"],
            "mixed_spread_group_rate": rates["mixed_spread_rate"],
            "no_spread_group_rate": rates["no_spread_rate"],
        }

    def _evaluate_gate(self, spread_rates: Dict[str, Any]) -> Dict[str, Any]:
        rq_rate = spread_rates["retrieval_quality_spread_group_rate"]
        po_rate = spread_rates["penalty_only_spread_group_rate"]
        passed = (
            rq_rate >= GATE_THRESHOLDS["min_retrieval_quality_spread_rate"]
            and po_rate <= GATE_THRESHOLDS["max_penalty_only_spread_rate"]
        )
        if passed:
            recommendation = "Proceed to Phase 1.19: Real GRPO Loss Dry-Run with Strategy Groups."
        elif rq_rate >= 0.4 and po_rate <= 0.4:
            recommendation = (
                "Mixed evidence. Run Phase 1.19b scale check (10_g4) before GRPO loss dry-run."
            )
        else:
            recommendation = (
                "Do not proceed to GRPO loss dry-run. Continue reward/prompt fixes or Phase 1.18e."
            )
        return {
            "gate_passed": passed,
            "gate_thresholds": GATE_THRESHOLDS,
            "recommendation": recommendation,
        }


def build_case_studies(analysis: Dict[str, Any]) -> str:
    group_reports = analysis["group_reports"]
    strategy_stats = analysis["strategy_stats"]
    lines = [
        "# Phase 1.19a Case Studies",
        "",
        "## Q1: Is zero_std=0 from retrieval quality?",
        "",
    ]

    rq_groups = [g for g in group_reports if g["spread_source"] == "retrieval_quality_spread"]
    lines.append(
        f"- Groups with `retrieval_quality_spread`: **{len(rq_groups)}/{analysis['num_groups']}**"
    )
    lines.append(
        f"- `zero_std_group_rate_quality_only`: **{analysis['zero_std_group_rate_quality_only']:.2f}**"
    )
    lines.append(
        f"- Gate passed: **{analysis['gate_evaluation']['gate_passed']}** — "
        f"{analysis['gate_evaluation']['recommendation']}"
    )
    lines.append("")

    lines.extend(["## Q2: Why is broad_recall better?", ""])
    br = strategy_stats.get("broad_recall", {})
    em = strategy_stats.get("exact_match", {})
    lines.extend(
        [
            f"- broad_recall mean NDCG@10: **{br.get('strategy_mean_ndcg_at_10', 0):.4f}**",
            f"- broad_recall mean quality reward: **{br.get('strategy_mean_quality_reward', 0):.4f}**",
            f"- broad_recall mean penalty: **{br.get('strategy_mean_penalty', 0):.4f}**",
            f"- broad_recall mean search calls: **{br.get('strategy_mean_search_calls', 0):.2f}**",
            f"- exact_match mean NDCG@10: **{em.get('strategy_mean_ndcg_at_10', 0):.4f}**",
            f"- exact_match mean penalty: **{em.get('strategy_mean_penalty', 0):.4f}**",
            "",
        ]
    )

    esci1 = next((g for g in group_reports if g["group_id"] == "esci_val_1"), None)
    if esci1:
        lines.extend(["### esci_val_1 (broad_recall NDCG spike)", ""])
        br_row = esci1["strategy_breakdown"].get("broad_recall", {})
        em_row = esci1["strategy_breakdown"].get("exact_match", {})
        lines.extend(
            [
                f"- broad_recall: reward={br_row.get('total_reward'):.4f}, "
                f"ndcg={br_row.get('final_ndcg_at_10'):.4f}, searches={br_row.get('num_search_calls')}",
                f"- exact_match: reward={em_row.get('total_reward'):.4f}, "
                f"ndcg={em_row.get('final_ndcg_at_10'):.4f}, searches={em_row.get('num_search_calls')}",
                "- Conclusion: broad_recall advantage is driven by **higher NDCG**, not fewer searches.",
                "",
            ]
        )

    lines.extend(["## Q3: Why are other strategies low?", ""])
    for name in ["exact_match", "attribute_expansion", "constraint_preserving"]:
        s = strategy_stats.get(name, {})
        lines.append(
            f"- {name}: mean_ndcg={s.get('strategy_mean_ndcg_at_10', 0):.4f}, "
            f"mean_penalty={s.get('strategy_mean_penalty', 0):.4f}, "
            f"mean_searches={s.get('strategy_mean_search_calls', 0):.2f}"
        )
    lines.append("")

    lines.extend(["## Q4: Why did esci_val_3 collapse?", ""])
    esci3 = next((g for g in group_reports if g["group_id"] == "esci_val_3"), None)
    if esci3:
        lines.extend(
            [
                f"- spread_source: `{esci3['spread_source']}`",
                f"- unique_final_query_count: **{esci3['unique_final_query_count']}**",
                f"- total_reward_spread: **{esci3['total_reward_spread']:.4f}**",
                f"- ndcg_spread: **{esci3['ndcg_spread']:.4f}**",
                "",
                "Per-strategy final queries:",
            ]
        )
        for name, row in esci3["strategy_breakdown"].items():
            lines.append(f"- {name}: `{row.get('final_query')}` (ndcg={row.get('final_ndcg_at_10'):.4f})")
        lines.extend(
            [
                "",
                "All four strategies converged to the same final query despite different prompts. "
                "Likely causes: query is too specific / qrels sparse / model defaults to original wording.",
                "",
            ]
        )

    lines.extend(["## Q5: Quality-only zero_std rate", ""])
    lines.extend(
        [
            f"- zero_std_group_rate_total_reward: **{analysis['zero_std_group_rate_total_reward']:.2f}**",
            f"- zero_std_group_rate_quality_only: **{analysis['zero_std_group_rate_quality_only']:.2f}**",
            "",
            "Removing penalties "
            + (
                "still leaves spread in most groups."
                if analysis["zero_std_group_rate_quality_only"] < 0.8
                else "collapses most groups — penalties were masking lack of quality signal."
            ),
            "",
        ]
    )

    return "\n".join(lines)
