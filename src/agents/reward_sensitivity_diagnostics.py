"""
Phase 1.18b reward sensitivity diagnostics.

Analyzes whether the current reward is too sparse or too insensitive for real
multi-sample GRPO groups. Does NOT train or change the reward formula.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.reward.outcome_reward import compute_ndcg, compute_recall
from src.reward.process_reward import RewardConfig, compute_episode_reward

DOC_ID_PATTERN = re.compile(r"\bid=([A-Z0-9]+)\b")
EPS = 1e-6

REWARD_ISSUE_RECOMMENDATIONS = {
    "current_reward_sensitive": (
        "Current NDCG@10-based reward already distinguishes retrieval quality within this group."
    ),
    "penalty_only_spread": (
        "Reward spread comes from search/repeat/invalid penalties, not retrieval quality. "
        "Separate retrieval-quality reward from cost penalties before GRPO."
    ),
    "ndcg10_blind_but_recall_sensitive": (
        "Consider adding Recall@50/MRR@50 diagnostic reward or best-step retrieval signal."
    ),
    "retrieval_results_change_but_metric_blind": (
        "BM25 topK changes but qrels/metrics do not reflect differences. "
        "Investigate label sparsity or add topK overlap / new-candidate signals."
    ),
    "query_too_similar": (
        "Query rewrites are semantically too close and retrieve similar topK. "
        "Improve rollout prompt diversity before changing reward."
    ),
    "label_sparse_or_all_zero": (
        "All metrics near zero with no spread — qrels may be sparse for this query. "
        "Consider richer labels or alternative metrics."
    ),
}

MAIN_CONCLUSION_TEMPLATE = (
    "Current NDCG@10 reward is too sparse for GRPO grouping. "
    "Reward variance mostly comes from penalties rather than retrieval quality. "
    "Recall@50/MRR@50 or best-step retrieval signals should be considered before training."
)


def token_jaccard(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def compute_mrr(retrieved: Sequence[str], targets: Sequence[str], k: int) -> float:
    target_set = set(targets)
    if not target_set:
        return 0.0
    for rank, doc_id in enumerate(list(retrieved)[:k], start=1):
        if doc_id in target_set:
            return 1.0 / rank
    return 0.0


def compute_hit(retrieved: Sequence[str], targets: Sequence[str], k: int) -> float:
    target_set = set(targets)
    if not target_set:
        return 0.0
    retrieved_k = set(list(retrieved)[:k])
    return 1.0 if retrieved_k & target_set else 0.0


def _pairwise_average(values: List[Any], metric_fn: Callable[[Any, Any], float]) -> float:
    if len(values) < 2:
        return 1.0 if values else 0.0
    scores = [metric_fn(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values))]
    return mean(scores) if scores else 0.0


def _spread(values: List[float]) -> float:
    if not values:
        return 0.0
    return max(values) - min(values)


def _std(values: List[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def _set_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


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


def decompose_trajectory_reward(trajectory: Dict[str, Any], config: Optional[RewardConfig] = None) -> Dict[str, float]:
    """Split current episode reward into retrieval vs penalty components."""
    cfg = config or RewardConfig()
    delta_ndcg_list = [
        float(s.get("delta_ndcg", 0.0))
        for s in trajectory.get("steps", [])
        if s.get("delta_ndcg") is not None
    ]
    breakdown = compute_episode_reward(
        final_ndcg=float(trajectory.get("final_ndcg_at_10", 0.0)),
        delta_ndcg_list=delta_ndcg_list,
        num_search_calls=int(trajectory.get("num_search_calls", 0)),
        num_invalid=int(trajectory.get("num_invalid_actions", 0)),
        num_repeated=int(trajectory.get("num_repeated_queries", 0)),
        has_final_answer=bool(trajectory.get("finished", False)),
        config=cfg,
    )
    search_cost = breakdown.penalties.get("search_cost", 0.0)
    repeat_penalty = breakdown.penalties.get("repeated_query", 0.0)
    invalid_penalty = breakdown.penalties.get("invalid_action", 0.0)
    no_final_penalty = breakdown.penalties.get("no_final_answer", 0.0)
    step_delta_component = cfg.lambda_process * breakdown.process_reward_sum
    final_ndcg_component = breakdown.final_reward

    return {
        "final_ndcg_component": final_ndcg_component,
        "step_delta_component": step_delta_component,
        "search_cost_penalty": search_cost,
        "repeat_penalty": repeat_penalty,
        "invalid_penalty": invalid_penalty,
        "no_final_penalty": no_final_penalty,
        "total_penalty": breakdown.total_penalty,
        "total_reward": breakdown.total_reward,
    }


def compute_metrics_for_retrieved(
    retrieved: Sequence[str],
    target_items: Sequence[str],
    rel_scores: Optional[Sequence[float]],
    topk_list: List[int],
) -> Dict[str, Any]:
    has_qrels = len(target_items) > 0
    metrics: Dict[str, Any] = {"has_qrels": has_qrels, "num_targets": len(target_items)}
    if not has_qrels:
        return metrics

    rel = list(rel_scores) if rel_scores is not None else [1.0] * len(target_items)
    for k in topk_list:
        metrics[f"ndcg@{k}"] = compute_ndcg(retrieved, list(target_items), k, rel_scores=rel)
        metrics[f"recall@{k}"] = compute_recall(retrieved, list(target_items), k)
        metrics[f"mrr@{k}"] = compute_mrr(retrieved, list(target_items), k)
        metrics[f"hit@{k}"] = compute_hit(retrieved, list(target_items), k)
    return metrics


class RewardSensitivityDiagnostics:
    """
    Phase 1.18b reward sensitivity diagnostics.

    This class analyzes whether the current reward is too sparse or too
    insensitive for real multi-sample GRPO groups.

    It does NOT train and does NOT change the reward formula.
    """

    def __init__(self, topk_list: Optional[List[int]] = None):
        self.topk_list = topk_list or [10, 50, 100]
        self.max_topk = max(self.topk_list)
        self._search_cache: Dict[Tuple[str, int], List[str]] = {}

    def load_rollouts(self, rollout_path: str | Path) -> List[Dict[str, Any]]:
        path = Path(rollout_path)
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def extract_queries(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        seen: set[tuple] = set()

        def add_item(**kwargs: Any) -> None:
            key = (kwargs["group_id"], kwargs["sample_id"], kwargs["query_type"], kwargs.get("step_id"), kwargs["query"])
            if key in seen:
                return
            seen.add(key)
            items.append(kwargs)

        for record in records:
            traj = record["trajectory"]
            group_id = record.get("group_id", "unknown")
            sample_id = record.get("sample_id", traj.get("qid", "unknown"))
            original_query = traj.get("user_query") or record["extra_info"].get("original_query", "")
            target_items = traj.get("target_items", [])
            rel_scores = traj.get("rel_scores")

            add_item(
                group_id=group_id,
                sample_id=sample_id,
                query_type="original_query",
                query=original_query,
                step_id=None,
                trajectory_id=sample_id,
                original_reward=float(record["reward"]),
                original_ndcg_at_10=float(record["metrics"]["ndcg_at_10"]),
                target_items=target_items,
                rel_scores=rel_scores,
            )

            best_query = record["extra_info"].get("best_query_by_ndcg") or traj.get("best_query_by_ndcg", original_query)
            add_item(
                group_id=group_id,
                sample_id=sample_id,
                query_type="best_query_by_ndcg",
                query=str(best_query),
                step_id=None,
                trajectory_id=sample_id,
                original_reward=float(record["reward"]),
                original_ndcg_at_10=float(record["metrics"]["ndcg_at_10"]),
                target_items=target_items,
                rel_scores=rel_scores,
            )

            final_query = record["extra_info"].get("final_query") or traj.get("final_query") or best_query
            add_item(
                group_id=group_id,
                sample_id=sample_id,
                query_type="final_query",
                query=str(final_query),
                step_id=None,
                trajectory_id=sample_id,
                original_reward=float(record["reward"]),
                original_ndcg_at_10=float(record["metrics"]["ndcg_at_10"]),
                target_items=target_items,
                rel_scores=rel_scores,
            )

            for step in traj.get("steps", []):
                action = step.get("action") or {}
                if action.get("tool") != "bm25_search":
                    continue
                query = action.get("query", "")
                if not query:
                    continue
                add_item(
                    group_id=group_id,
                    sample_id=sample_id,
                    query_type="search_query",
                    query=str(query),
                    step_id=int(step.get("step_id", 0)),
                    trajectory_id=sample_id,
                    original_reward=float(record["reward"]),
                    original_ndcg_at_10=float(step.get("ndcg_at_10") or 0.0),
                    target_items=target_items,
                    rel_scores=rel_scores,
                )
        return items

    def _retrieve_doc_ids(self, search_tool: Any, query: str, topk: int) -> List[str]:
        key = (query, topk)
        if key not in self._search_cache:
            self._search_cache[key] = search_tool.retrieved_ids(query, topk=topk)
        return self._search_cache[key]

    def recompute_metrics_for_queries(
        self,
        query_items: List[Dict[str, Any]],
        search_tool: Optional[Any] = None,
        skip_bm25_recompute: bool = False,
    ) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        for item in query_items:
            row = dict(item)
            target_items = item.get("target_items") or []
            rel_scores = item.get("rel_scores")
            query = item["query"]

            if skip_bm25_recompute or search_tool is None:
                row["bm25_recomputed"] = False
                row["metrics"] = {"has_qrels": len(target_items) > 0, "num_targets": len(target_items)}
                enriched.append(row)
                continue

            retrieved = self._retrieve_doc_ids(search_tool, query, self.max_topk)
            metrics = compute_metrics_for_retrieved(retrieved, target_items, rel_scores, self.topk_list)
            row["bm25_recomputed"] = True
            row["retrieved_doc_ids"] = retrieved
            row["metrics"] = metrics
            enriched.append(row)
        return enriched

    def _record_final_query_metrics(
        self,
        record: Dict[str, Any],
        query_metrics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        sample_id = record.get("sample_id")
        for item in query_metrics:
            if item["sample_id"] == sample_id and item["query_type"] == "final_query":
                return item.get("metrics", {})
        return {}

    def analyze_group_sensitivity(
        self,
        group_id: str,
        records: List[Dict[str, Any]],
        query_metrics: List[Dict[str, Any]],
        phase118a_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        current_rewards = [float(r["reward"]) for r in records]
        reward_std = _std(current_rewards)
        reward_spread = _spread(current_rewards)

        decompositions = [decompose_trajectory_reward(r["trajectory"]) for r in records]
        ndcg_components = [d["final_ndcg_component"] for d in decompositions]
        step_delta_components = [d["step_delta_component"] for d in decompositions]
        search_costs = [d["search_cost_penalty"] for d in decompositions]
        repeat_penalties = [d["repeat_penalty"] for d in decompositions]

        ndcg_spread = _spread(ndcg_components)
        search_cost_spread = _spread(search_costs)
        repeat_penalty_spread = _spread(repeat_penalties)
        step_delta_spread = _spread(step_delta_components)

        penalty_spread = max(search_cost_spread, repeat_penalty_spread, _spread([d["invalid_penalty"] for d in decompositions]))

        if reward_spread > EPS and ndcg_spread > EPS:
            main_reward_spread_source = "retrieval_quality"
        elif reward_spread > EPS and penalty_spread > EPS:
            main_reward_spread_source = "penalty_only"
        else:
            main_reward_spread_source = "no_spread"

        final_metrics_rows = [self._record_final_query_metrics(r, query_metrics) for r in records]

        def metric_values(key: str) -> List[float]:
            return [float(m.get(key, 0.0)) for m in final_metrics_rows if m.get("has_qrels", True)]

        ndcg10_values = metric_values(f"ndcg@{10}") or [float(r["metrics"]["ndcg_at_10"]) for r in records]
        ndcg50_values = metric_values(f"ndcg@{50}")
        recall50_values = metric_values(f"recall@{50}")
        mrr50_values = metric_values(f"mrr@{50}")

        ndcg10_spread = _spread(ndcg10_values)
        ndcg50_spread = _spread(ndcg50_values) if ndcg50_values else 0.0
        recall50_spread = _spread(recall50_values) if recall50_values else 0.0
        mrr50_spread = _spread(mrr50_values) if mrr50_values else 0.0

        final_queries = [
            str(r["extra_info"].get("final_query") or r["trajectory"].get("final_query") or "")
            for r in records
        ]
        fingerprints = [_trajectory_fingerprint(r["trajectory"]) for r in records]

        avg_pairwise_final_query_jaccard = _pairwise_average(final_queries, token_jaccard)
        avg_pairwise_trajectory_jaccard = _pairwise_average(
            fingerprints,
            lambda a, b: token_jaccard(a.replace("|", " "), b.replace("|", " ")),
        )

        if phase118a_report:
            avg_pairwise_final_query_jaccard = phase118a_report.get(
                "avg_pairwise_final_query_jaccard", avg_pairwise_final_query_jaccard
            )
            avg_pairwise_trajectory_jaccard = phase118a_report.get(
                "avg_pairwise_trajectory_jaccard", avg_pairwise_trajectory_jaccard
            )

        topk_sets_by_k: Dict[int, List[List[str]]] = {k: [] for k in self.topk_list}
        for record in records:
            sample_id = record.get("sample_id")
            final_item = next(
                (q for q in query_metrics if q["sample_id"] == sample_id and q["query_type"] == "final_query"),
                None,
            )
            if final_item and final_item.get("retrieved_doc_ids"):
                retrieved = final_item["retrieved_doc_ids"]
                for k in self.topk_list:
                    topk_sets_by_k[k].append(retrieved[:k])

        overlap_stats: Dict[str, Any] = {}
        for k in self.topk_list:
            sets = topk_sets_by_k[k]
            if len(sets) >= 2 and any(len(s) > 0 for s in sets):
                overlap_stats[f"avg_pairwise_top{k}_overlap"] = _pairwise_average(sets, _set_jaccard)
            else:
                overlap_stats[f"avg_pairwise_top{k}_overlap"] = None

        original_query = records[0]["trajectory"].get("user_query") or records[0]["extra_info"].get("original_query", "")

        report = {
            "group_id": group_id,
            "original_query": original_query,
            "group_size": len(records),
            "current_reward_values": current_rewards,
            "current_reward_std": reward_std,
            "current_reward_spread": reward_spread,
            "ndcg10_values": ndcg10_values,
            "ndcg50_values": ndcg50_values,
            "recall50_values": recall50_values,
            "mrr50_values": mrr50_values,
            "ndcg10_spread": ndcg10_spread,
            "ndcg50_spread": ndcg50_spread,
            "recall50_spread": recall50_spread,
            "mrr50_spread": mrr50_spread,
            "avg_pairwise_final_query_jaccard": avg_pairwise_final_query_jaccard,
            "avg_pairwise_trajectory_jaccard": avg_pairwise_trajectory_jaccard,
            "reward_decompositions": decompositions,
            "ndcg_spread": ndcg_spread,
            "search_cost_spread": search_cost_spread,
            "repeat_penalty_spread": repeat_penalty_spread,
            "step_delta_spread": step_delta_spread,
            "main_reward_spread_source": main_reward_spread_source,
            **overlap_stats,
        }

        issue = self.classify_reward_issue(report)
        report["reward_issue_type"] = issue
        report["recommendation"] = REWARD_ISSUE_RECOMMENDATIONS[issue]
        return report

    def classify_reward_issue(self, group_report: Dict[str, Any]) -> str:
        reward_std = group_report["current_reward_std"]
        main_source = group_report["main_reward_spread_source"]
        recall50_spread = group_report.get("recall50_spread", 0.0)
        mrr50_spread = group_report.get("mrr50_spread", 0.0)
        ndcg10_spread = group_report.get("ndcg10_spread", 0.0)
        top50_overlap = group_report.get("avg_pairwise_top50_overlap")
        final_jaccard = group_report.get("avg_pairwise_final_query_jaccard", 0.0)

        if reward_std > EPS and main_source == "retrieval_quality":
            return "current_reward_sensitive"
        if reward_std > EPS and main_source == "penalty_only":
            return "penalty_only_spread"

        if recall50_spread > EPS or mrr50_spread > EPS:
            return "ndcg10_blind_but_recall_sensitive"

        if (
            final_jaccard > 0.85
            and top50_overlap is not None
            and top50_overlap > 0.8
        ):
            return "query_too_similar"

        if (
            top50_overlap is not None
            and top50_overlap < 0.8
            and recall50_spread <= EPS
            and mrr50_spread <= EPS
        ):
            return "retrieval_results_change_but_metric_blind"

        ndcg10_values = group_report.get("ndcg10_values", [])
        recall50_values = group_report.get("recall50_values", [])
        if (
            ndcg10_spread <= EPS
            and recall50_spread <= EPS
            and mrr50_spread <= EPS
            and all(abs(v) <= EPS for v in ndcg10_values)
            and (not recall50_values or all(abs(v) <= EPS for v in recall50_values))
        ):
            return "label_sparse_or_all_zero"

        if final_jaccard > 0.85:
            return "query_too_similar"

        return "label_sparse_or_all_zero"

    def analyze_all(
        self,
        records: List[Dict[str, Any]],
        search_tool: Optional[Any] = None,
        skip_bm25_recompute: bool = False,
        phase118a_reports: Optional[Dict[str, Dict[str, Any]]] = None,
        max_groups: Optional[int] = None,
    ) -> Dict[str, Any]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            gid = record.get("group_id", "unknown")
            grouped.setdefault(gid, []).append(record)
        for gid in grouped:
            grouped[gid].sort(key=lambda r: r.get("group_index", 0))

        group_ids = sorted(grouped.keys())
        if max_groups is not None:
            group_ids = group_ids[:max_groups]

        query_items = self.extract_queries(records)
        query_metrics = self.recompute_metrics_for_queries(
            query_items,
            search_tool=search_tool,
            skip_bm25_recompute=skip_bm25_recompute,
        )

        group_reports: List[Dict[str, Any]] = []
        for gid in group_ids:
            phase118a = (phase118a_reports or {}).get(gid)
            group_reports.append(
                self.analyze_group_sensitivity(gid, grouped[gid], query_metrics, phase118a_report=phase118a)
            )

        summary = self._build_summary(group_reports, len(records))
        return {
            "group_reports": group_reports,
            "query_metrics": query_metrics,
            "summary": summary,
        }

    def _build_summary(self, group_reports: List[Dict[str, Any]], num_records: int) -> Dict[str, Any]:
        num_groups = len(group_reports)
        issue_counts: Dict[str, int] = {key: 0 for key in REWARD_ISSUE_RECOMMENDATIONS}
        for report in group_reports:
            issue_counts[report["reward_issue_type"]] += 1

        issue_rates = {k: (v / num_groups if num_groups else 0.0) for k, v in issue_counts.items()}

        zero_std_count = sum(1 for r in group_reports if r["current_reward_std"] <= EPS)
        sensitive_count = issue_counts["current_reward_sensitive"]
        penalty_count = issue_counts["penalty_only_spread"]

        def avg_field(key: str) -> float:
            vals = [float(r[key]) for r in group_reports if r.get(key) is not None]
            return mean(vals) if vals else 0.0

        overlap_means = {}
        for k in self.topk_list:
            vals = [float(r[f"avg_pairwise_top{k}_overlap"]) for r in group_reports if r.get(f"avg_pairwise_top{k}_overlap") is not None]
            overlap_means[f"mean_top{k}_overlap"] = mean(vals) if vals else None

        main_conclusion = self._derive_main_conclusion(issue_rates, penalty_count, num_groups)

        return {
            "num_groups": num_groups,
            "group_size": group_reports[0]["group_size"] if group_reports else 0,
            "num_rollout_records": num_records,
            "current_zero_std_group_rate": zero_std_count / num_groups if num_groups else 0.0,
            "current_reward_sensitive_rate": issue_rates["current_reward_sensitive"],
            "penalty_only_spread_rate": issue_rates["penalty_only_spread"],
            "ndcg10_blind_but_recall_sensitive_rate": issue_rates["ndcg10_blind_but_recall_sensitive"],
            "retrieval_results_change_but_metric_blind_rate": issue_rates["retrieval_results_change_but_metric_blind"],
            "query_too_similar_rate": issue_rates["query_too_similar"],
            "label_sparse_or_all_zero_rate": issue_rates["label_sparse_or_all_zero"],
            "category_counts": issue_counts,
            "category_rates": issue_rates,
            "mean_ndcg10_spread": avg_field("ndcg10_spread"),
            "mean_ndcg50_spread": avg_field("ndcg50_spread"),
            "mean_recall50_spread": avg_field("recall50_spread"),
            "mean_mrr50_spread": avg_field("mrr50_spread"),
            **overlap_means,
            "main_conclusion": main_conclusion,
            "recommended_next_phase": self._recommend_next_phase(issue_rates),
        }

    def _derive_main_conclusion(self, issue_rates: Dict[str, float], penalty_count: int, num_groups: int) -> str:
        if issue_rates["current_reward_sensitive"] >= 0.5:
            return (
                "Enough groups show retrieval-quality reward spread under current NDCG@10. "
                "Consider Phase 1.18 Real GRPO Loss Dry-Run."
            )
        if issue_rates["penalty_only_spread"] >= 0.2 or penalty_count > 0:
            base = MAIN_CONCLUSION_TEMPLATE
            if issue_rates["ndcg10_blind_but_recall_sensitive"] >= 0.2:
                base += " Recall@50/MRR@50 show additional sensitivity in some groups."
            return base
        if issue_rates["query_too_similar"] >= 0.4:
            return (
                "Query rewrites are too similar and retrieve overlapping topK. "
                "Improve rollout prompt diversity before reward shaping."
            )
        return MAIN_CONCLUSION_TEMPLATE

    def _recommend_next_phase(self, issue_rates: Dict[str, float]) -> str:
        if issue_rates["current_reward_sensitive"] >= 0.5:
            return "Phase 1.18: Real GRPO Loss Dry-Run"
        if issue_rates["penalty_only_spread"] >= 0.2:
            return "Phase 1.18c: Reward Shaping Proposal + Dry-Run (separate retrieval quality from penalties)"
        if issue_rates["ndcg10_blind_but_recall_sensitive"] >= 0.3:
            return "Phase 1.18c: Reward Shaping Proposal + Dry-Run (Recall@50/MRR@50/best_step_delta)"
        if issue_rates["retrieval_results_change_but_metric_blind"] >= 0.3:
            return "Phase 1.18c: TopK Overlap / Retrieval Diversity Reward Dry-Run"
        if issue_rates["query_too_similar"] >= 0.3:
            return "Phase 1.18c: Rollout Diversity Prompt Fix"
        return "Phase 1.18c: Reward Shaping Proposal + Dry-Run"


def load_group_diagnostics(path: str | Path) -> Dict[str, Dict[str, Any]]:
    reports: Dict[str, Dict[str, Any]] = {}
    p = Path(path)
    if not p.exists():
        return reports
    with p.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                row = json.loads(line)
                reports[row["group_id"]] = row
    return reports


def build_reward_recommendations(analysis: Dict[str, Any]) -> str:
    summary = analysis["summary"]
    group_reports = analysis["group_reports"]
    lines = [
        "# Phase 1.18b Reward Sensitivity Diagnostics",
        "",
        "## Main Finding",
        "",
        summary["main_conclusion"],
        "",
        f"**Recommended next phase:** {summary['recommended_next_phase']}",
        "",
        "## Current Reward Problem",
        "",
        f"- Zero-std group rate: **{summary['current_zero_std_group_rate']:.2f}**",
        f"- Penalty-only spread rate: **{summary['penalty_only_spread_rate']:.2f}**",
        f"- Retrieval-sensitive rate: **{summary['current_reward_sensitive_rate']:.2f}**",
        "",
        "Current total reward mixes final NDCG@10, step ΔNDCG, and penalties (search/repeat/invalid). "
        "Most GRPO groups collapse because retrieval-quality components do not spread across samples.",
        "",
        "## Group-Level Evidence",
        "",
    ]

    for report in group_reports:
        lines.extend(
            [
                f"### `{report['group_id']}` — `{report['reward_issue_type']}`",
                "",
                f"- Original query: {report['original_query']}",
                f"- Rewards: {report['current_reward_values']}",
                f"- Main spread source: **{report['main_reward_spread_source']}**",
                f"- NDCG@10 spread: {report['ndcg10_spread']:.4f}, Recall@50 spread: {report['recall50_spread']:.4f}",
                f"- Recommendation: {report['recommendation']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Metric Sensitivity",
            "",
            f"- Mean NDCG@10 spread: **{summary['mean_ndcg10_spread']:.4f}**",
            f"- Mean NDCG@50 spread: **{summary['mean_ndcg50_spread']:.4f}**",
            f"- Mean Recall@50 spread: **{summary['mean_recall50_spread']:.4f}**",
            f"- Mean MRR@50 spread: **{summary['mean_mrr50_spread']:.4f}**",
            "",
            "When NDCG@10 is uniformly zero but Recall@50/MRR@50 spread > 0, NDCG@10 is too coarse for GRPO grouping.",
            "",
            "## TopK Overlap Analysis",
            "",
        ]
    )

    for k in [10, 50, 100]:
        key = f"mean_top{k}_overlap"
        val = summary.get(key)
        if val is not None:
            lines.append(f"- Mean pairwise top-{k} overlap: **{val:.3f}**")
    lines.extend(
        [
            "",
            "High overlap + zero reward spread suggests query rewrites retrieve similar documents. "
            "Low overlap + zero metric spread suggests labels/metrics are blind to retrieval changes.",
            "",
            "## Reward Spread Source",
            "",
        ]
    )

    penalty_groups = [r for r in group_reports if r["main_reward_spread_source"] == "penalty_only"]
    if penalty_groups:
        lines.append("Groups where spread comes from penalties (not NDCG):")
        for r in penalty_groups:
            lines.append(
                f"- `{r['group_id']}`: rewards={r['current_reward_values']}, "
                f"search_cost_spread={r['search_cost_spread']:.3f}, ndcg_spread={r['ndcg_spread']:.3f}"
            )
    else:
        lines.append("No penalty-only spread groups found.")

    lines.extend(
        [
            "",
            "## Recommendation for Phase 1.18c / 1.19",
            "",
            f"**{summary['recommended_next_phase']}**",
            "",
            "Proposed shaping dry-run (do not change formal reward yet):",
            "",
            "```text",
            "R = NDCG@10",
            "  + alpha * Recall@50",
            "  + beta * MRR@50",
            "  + gamma * best_step_delta",
            "  - penalties (tracked separately for GRPO advantage)",
            "```",
            "",
            "GRPO advantage should prioritize retrieval-quality terms; "
            "cost penalties should not be the primary source of group spread.",
            "",
        ]
    )
    return "\n".join(lines)
