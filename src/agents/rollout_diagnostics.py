"""
Phase 1.18a rollout diversity / reward variance diagnostics.

Analyzes real multi-sample rollout records without training or changing rewards.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

DOC_ID_PATTERN = re.compile(r"\bid=([A-Z0-9]+)\b")

DIAGNOSIS_NOTES = {
    "diverse_trajectory_reward_spread": (
        "Trajectories differ and rewards differ. This group can produce meaningful GRPO advantages."
    ),
    "diverse_trajectory_zero_reward": (
        "Trajectories differ, but final reward is identical. "
        "BM25/NDCG may be insensitive to these query rewrites."
    ),
    "same_trajectory_zero_reward": (
        "All trajectories are identical and rewards are identical. "
        "Group training signal fully collapses."
    ),
    "same_trajectory_reward_spread": (
        "Trajectories are identical but rewards differ. "
        "Check reward computation or environment randomness."
    ),
    "invalid_or_unfinished": (
        "Rollout protocol issue: invalid actions or unfinished episodes. "
        "Fix rollout stability before GRPO."
    ),
}

MAIN_DIAGNOSIS_TEMPLATES = {
    "ready_for_grpo": (
        "Enough groups show both trajectory diversity and reward spread. "
        "Consider Phase 1.18 Real GRPO Loss Dry-Run."
    ),
    "reward_sensitivity": (
        "Trajectory diversity exists, but most groups still have zero reward variance. "
        "Before GRPO training, improve reward sensitivity or rollout diversity."
    ),
    "rollout_diversity": (
        "Many groups collapse to identical trajectories with zero reward variance. "
        "Focus on rollout diversity (prompt / sampling) before GRPO."
    ),
    "protocol_issues": (
        "Invalid or unfinished rollouts detected. Fix rollout protocol before GRPO."
    ),
}


def token_jaccard(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def _pairwise_average(values: List[str], metric_fn) -> float:
    if len(values) < 2:
        return 1.0 if values else 0.0
    scores: List[float] = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            scores.append(metric_fn(values[i], values[j]))
    return mean(scores) if scores else 0.0


def _extract_search_query_sequences(trajectory: Dict[str, Any]) -> List[str]:
    queries: List[str] = []
    for step in trajectory.get("steps", []):
        action = step.get("action") or {}
        if action.get("tool") == "bm25_search":
            query = action.get("query", "")
            if query:
                queries.append(str(query))
    return queries


def _trajectory_fingerprint(trajectory: Dict[str, Any]) -> str:
    parts: List[str] = []
    for step in trajectory.get("steps", []):
        action = step.get("action") or {}
        tool = action.get("tool", "")
        if tool == "bm25_search":
            parts.append(f"search:{action.get('query', '')}")
        elif tool == "final_answer":
            parts.append(f"final:{action.get('final_query', '')}")
        elif tool == "invalid":
            parts.append("invalid")
        else:
            parts.append(str(tool or "none"))
    return "|".join(parts)


def _trajectory_summary(trajectory: Dict[str, Any]) -> str:
    parts: List[str] = []
    for step in trajectory.get("steps", []):
        action = step.get("action") or {}
        tool = action.get("tool", "")
        if tool == "bm25_search":
            parts.append(f"search('{action.get('query', '')}')")
        elif tool == "final_answer":
            parts.append(f"final('{action.get('final_query', '')}')")
        elif tool == "invalid":
            parts.append("invalid")
    return " -> ".join(parts) if parts else "(empty)"


def _extract_topk_doc_ids(trajectory: Dict[str, Any]) -> List[str]:
    """Extract doc ids from the last step observation that contains BM25 hits."""
    for step in reversed(trajectory.get("steps", [])):
        obs = step.get("observation") or ""
        if not obs or "hits=" not in obs:
            continue
        doc_ids = DOC_ID_PATTERN.findall(obs)
        if doc_ids:
            return doc_ids
    return []


def _set_jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


def _classify_group(
    *,
    unique_trajectory_count: int,
    reward_std: float,
    finish_rate: float,
    invalid_action_rate: float,
    eps: float = 1e-6,
) -> str:
    if invalid_action_rate > 0 or finish_rate < 1.0:
        return "invalid_or_unfinished"
    if unique_trajectory_count == 1 and reward_std > eps:
        return "same_trajectory_reward_spread"
    if unique_trajectory_count > 1 and reward_std > eps:
        return "diverse_trajectory_reward_spread"
    if unique_trajectory_count > 1 and reward_std <= eps:
        return "diverse_trajectory_zero_reward"
    return "same_trajectory_zero_reward"


class RolloutDiagnostics:
    """
    Phase 1.18a rollout diversity / reward variance diagnostics.

    This class analyzes real multi-sample rollout records.
    It does NOT train and does NOT change rewards.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def load_rollouts(self, rollout_path: str | Path) -> List[Dict[str, Any]]:
        path = Path(rollout_path)
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def group_rollouts(self, records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            group_id = record.get("group_id") or record.get("sample_id", "unknown")
            grouped.setdefault(group_id, []).append(record)
        for gid in grouped:
            grouped[gid].sort(key=lambda r: r.get("group_index", 0))
        return grouped

    def analyze_group(self, group_id: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        rewards = [float(r["reward"]) for r in records]
        ndcg_values = [float(r["metrics"]["ndcg_at_10"]) for r in records]

        reward_mean = mean(rewards) if rewards else 0.0
        reward_std = pstdev(rewards) if len(rewards) > 1 else 0.0
        reward_min = min(rewards) if rewards else 0.0
        reward_max = max(rewards) if rewards else 0.0
        reward_spread = reward_max - reward_min

        ndcg_std = pstdev(ndcg_values) if len(ndcg_values) > 1 else 0.0
        ndcg_spread = (max(ndcg_values) - min(ndcg_values)) if ndcg_values else 0.0

        final_queries = [
            str(r["extra_info"].get("final_query") or r["trajectory"].get("final_query") or "")
            for r in records
        ]
        search_query_sequences = [_extract_search_query_sequences(r["trajectory"]) for r in records]
        trajectory_fingerprints = [_trajectory_fingerprint(r["trajectory"]) for r in records]

        all_search_queries = [q for seq in search_query_sequences for q in seq]
        unique_final_query_count = len(set(final_queries))
        unique_search_query_count = len(set(all_search_queries))
        unique_trajectory_count = len(set(trajectory_fingerprints))

        total_env_steps = sum(len(r["trajectory"].get("steps", [])) for r in records)
        total_invalid = sum(int(r["metrics"].get("num_invalid_actions", 0)) for r in records)
        finish_rate = sum(1 for r in records if r["metrics"].get("finished")) / len(records)
        llm_finish_rate = sum(1 for r in records if r["metrics"].get("llm_finished")) / len(records)
        auto_finish_rate = sum(1 for r in records if r["metrics"].get("auto_finished")) / len(records)
        invalid_action_rate = total_invalid / total_env_steps if total_env_steps else 0.0

        avg_pairwise_final_query_jaccard = _pairwise_average(final_queries, token_jaccard)

        flat_search_queries = [q for seq in search_query_sequences for q in seq]
        avg_pairwise_search_query_jaccard = _pairwise_average(flat_search_queries, token_jaccard)
        avg_pairwise_trajectory_jaccard = _pairwise_average(
            trajectory_fingerprints,
            lambda a, b: token_jaccard(a.replace("|", " "), b.replace("|", " ")),
        )

        topk_sets = [_extract_topk_doc_ids(r["trajectory"]) for r in records]
        topk_overlap_available = any(len(s) > 0 for s in topk_sets)
        avg_pairwise_topk_overlap: Optional[float] = None
        if topk_overlap_available:
            nonempty = [s for s in topk_sets if s]
            avg_pairwise_topk_overlap = _pairwise_average(nonempty, _set_jaccard)

        original_query = records[0]["extra_info"].get("original_query") or records[0]["trajectory"].get(
            "user_query", ""
        )

        diagnosis_type = _classify_group(
            unique_trajectory_count=unique_trajectory_count,
            reward_std=reward_std,
            finish_rate=finish_rate,
            invalid_action_rate=invalid_action_rate,
            eps=self.eps,
        )

        report: Dict[str, Any] = {
            "group_id": group_id,
            "original_query": original_query,
            "group_size": len(records),
            "rewards": rewards,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "reward_min": reward_min,
            "reward_max": reward_max,
            "reward_spread": reward_spread,
            "zero_reward_std": reward_std <= self.eps,
            "ndcg_at_10_values": ndcg_values,
            "ndcg_std": ndcg_std,
            "ndcg_spread": ndcg_spread,
            "final_queries": final_queries,
            "unique_final_query_count": unique_final_query_count,
            "all_same_final_query": unique_final_query_count == 1,
            "search_query_sequences": search_query_sequences,
            "unique_search_query_count": unique_search_query_count,
            "unique_trajectory_count": unique_trajectory_count,
            "all_same_trajectory": unique_trajectory_count == 1,
            "avg_pairwise_final_query_jaccard": avg_pairwise_final_query_jaccard,
            "avg_pairwise_search_query_jaccard": avg_pairwise_search_query_jaccard,
            "avg_pairwise_trajectory_jaccard": avg_pairwise_trajectory_jaccard,
            "finish_rate": finish_rate,
            "llm_finish_rate": llm_finish_rate,
            "auto_finish_rate": auto_finish_rate,
            "invalid_action_rate": invalid_action_rate,
            "diagnosis_type": diagnosis_type,
            "diagnosis_note": DIAGNOSIS_NOTES[diagnosis_type],
            "trajectory_summaries": [_trajectory_summary(r["trajectory"]) for r in records],
        }

        if topk_overlap_available:
            report["topk_overlap_available"] = True
            report["avg_pairwise_topk_overlap"] = avg_pairwise_topk_overlap
            report["topk_doc_id_sets"] = topk_sets
        else:
            report["topk_overlap_available"] = False

        return report

    def analyze_all_groups(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        grouped = self.group_rollouts(records)
        group_reports = [self.analyze_group(gid, members) for gid, members in sorted(grouped.items())]
        classification = self.classify_groups(group_reports)
        return {
            "group_reports": group_reports,
            "classification": classification,
            "num_groups": len(group_reports),
            "num_rollout_records": len(records),
            "group_size": group_reports[0]["group_size"] if group_reports else 0,
        }

    def classify_groups(self, group_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        category_counts: Dict[str, int] = {
            "diverse_trajectory_reward_spread": 0,
            "diverse_trajectory_zero_reward": 0,
            "same_trajectory_zero_reward": 0,
            "same_trajectory_reward_spread": 0,
            "invalid_or_unfinished": 0,
        }
        for report in group_reports:
            category_counts[report["diagnosis_type"]] += 1

        num_groups = len(group_reports)
        category_rates = {
            key: (count / num_groups if num_groups else 0.0) for key, count in category_counts.items()
        }

        zero_std_group_count = sum(1 for r in group_reports if r["zero_reward_std"])
        zero_std_group_rate = zero_std_group_count / num_groups if num_groups else 0.0

        def avg_field(key: str) -> float:
            vals = [float(r[key]) for r in group_reports]
            return mean(vals) if vals else 0.0

        spread_rate = category_rates["diverse_trajectory_reward_spread"]
        zero_reward_rate = category_rates["diverse_trajectory_zero_reward"]
        same_traj_rate = category_rates["same_trajectory_zero_reward"]
        invalid_rate = category_rates["invalid_or_unfinished"]

        if invalid_rate > 0.2:
            main_diagnosis = MAIN_DIAGNOSIS_TEMPLATES["protocol_issues"]
            next_phase = "Phase 1.18c (after fixing rollout protocol)"
        elif spread_rate >= 0.5:
            main_diagnosis = MAIN_DIAGNOSIS_TEMPLATES["ready_for_grpo"]
            next_phase = "Phase 1.18: Real GRPO Loss Dry-Run"
        elif zero_reward_rate >= 0.5:
            main_diagnosis = MAIN_DIAGNOSIS_TEMPLATES["reward_sensitivity"]
            next_phase = "Phase 1.18b: Reward Sensitivity Diagnostics"
        elif same_traj_rate >= 0.5:
            main_diagnosis = MAIN_DIAGNOSIS_TEMPLATES["rollout_diversity"]
            next_phase = "Phase 1.18c: Rollout Diversity Prompt / Sampling Fix"
        else:
            main_diagnosis = MAIN_DIAGNOSIS_TEMPLATES["reward_sensitivity"]
            next_phase = "Phase 1.18b: Reward Sensitivity Diagnostics"

        return {
            "category_counts": category_counts,
            "category_rates": category_rates,
            "zero_std_group_count": zero_std_group_count,
            "zero_std_group_rate": zero_std_group_rate,
            "mean_group_reward_std": avg_field("reward_std"),
            "mean_reward_spread": avg_field("reward_spread"),
            "mean_unique_trajectory_count": avg_field("unique_trajectory_count"),
            "mean_unique_final_query_count": avg_field("unique_final_query_count"),
            "avg_pairwise_final_query_jaccard": avg_field("avg_pairwise_final_query_jaccard"),
            "avg_pairwise_search_query_jaccard": avg_field("avg_pairwise_search_query_jaccard"),
            "avg_pairwise_trajectory_jaccard": avg_field("avg_pairwise_trajectory_jaccard"),
            "main_diagnosis": main_diagnosis,
            "recommended_next_phase": next_phase,
        }


def build_case_studies(group_reports: List[Dict[str, Any]]) -> str:
    lines: List[str] = ["# Phase 1.18a Case Studies", ""]

    spread_groups = [r for r in group_reports if r["diagnosis_type"] == "diverse_trajectory_reward_spread"]
    zero_reward_groups = [r for r in group_reports if r["diagnosis_type"] == "diverse_trajectory_zero_reward"]
    collapsed_groups = [r for r in group_reports if r["diagnosis_type"] == "same_trajectory_zero_reward"]

    lines.append("## Case 1: Group with reward spread (good for GRPO)")
    lines.append("")
    if spread_groups:
        g = spread_groups[0]
        lines.extend(_format_case_block(g, include_why=False))
        lines.append("**Diagnosis:** Trajectories and rewards both vary; GRPO advantage should be non-zero.")
    else:
        lines.append("No group with `diverse_trajectory_reward_spread` found in this run.")
    lines.append("")

    lines.append("## Case 2: Diverse trajectories but identical reward")
    lines.append("")
    if zero_reward_groups:
        g = zero_reward_groups[0]
        lines.extend(_format_case_block(g, include_why=True))
        lines.append(
            "**Why this matters:** Different trajectories do not change BM25/NDCG reward, "
            "so GRPO advantage collapses."
        )
        if g.get("topk_overlap_available"):
            lines.append(
                f"**TopK overlap:** avg_pairwise_topk_overlap={g.get('avg_pairwise_topk_overlap', 0):.3f}"
            )
            if g.get("avg_pairwise_final_query_jaccard", 1.0) > 0.5 and g.get("avg_pairwise_topk_overlap", 0) > 0.8:
                lines.append(
                    "Queries differ somewhat but retrieved topK sets are highly overlapping — "
                    "BM25 may return similar results."
                )
            elif g.get("avg_pairwise_final_query_jaccard", 1.0) <= 0.5:
                lines.append(
                    "Final queries differ substantially yet reward is unchanged — "
                    "NDCG@10 may be too coarse or insensitive."
                )
    else:
        lines.append("No group with `diverse_trajectory_zero_reward` found.")
    lines.append("")

    lines.append("## Case 3: Fully collapsed same-trajectory group")
    lines.append("")
    if collapsed_groups:
        g = collapsed_groups[0]
        lines.extend(_format_case_block(g, include_why=False))
        lines.append("**Diagnosis:** Sampling produced identical trajectories and identical rewards.")
    else:
        lines.append("No fully collapsed same-trajectory group found.")

    return "\n".join(lines) + "\n"


def _format_case_block(report: Dict[str, Any], include_why: bool) -> List[str]:
    lines = [
        f"**Group ID:** `{report['group_id']}`",
        f"**Original query:** {report['original_query']}",
        f"**Rewards:** {report['rewards']}",
        f"**NDCG@10 values:** {report['ndcg_at_10_values']}",
        f"**Final queries:** {report['final_queries']}",
    ]
    if include_why:
        lines.append(f"**Search query sequences:** {report['search_query_sequences']}")
    lines.append("**Trajectory summaries:**")
    for i, summary in enumerate(report.get("trajectory_summaries", [])):
        lines.append(f"- g{i}: {summary}")
    lines.append(f"**Diagnosis type:** `{report['diagnosis_type']}`")
    lines.append(f"**Note:** {report['diagnosis_note']}")
    return lines
