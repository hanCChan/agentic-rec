"""
Phase 1.18e: Qrels / Metric Blindness analysis.

Diagnoses whether ESCI smoke qrels and IR metrics can produce retrieval-quality
group spread for strategy-controlled rollouts. Does NOT train or change reward.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.agents.reward_sensitivity_diagnostics import (
    _pairwise_average,
    _set_jaccard,
    _spread,
    _std,
    compute_metrics_for_retrieved,
    token_jaccard,
)
from src.reward.outcome_reward import compute_recall

EPS = 1e-6
COVERAGE_KS = [10, 50, 100, 1000]
DEFAULT_K_LIST = [10, 50, 100, 1000]

METRIC_BLINDNESS_TYPES = {
    "small_k_blind_large_k_signal": (
        "NDCG@10 spread is zero but larger-K NDCG/Recall/MRR shows group spread. "
        "Consider Rec-R1-style larger-K reward or quality-only advantage at @100."
    ),
    "qrels_sparse_all_k_blind": (
        "No retrieval-quality spread at any K; qrels may be too sparse or BM25 cannot "
        "surface relevant docs for this query. Replace or expand smoke samples."
    ),
    "bm25_retrieval_failure": (
        "Query has relevant docs in qrels but BM25 top1000 does not retrieve them. "
        "BM25 tool cannot provide learnable feedback for this query."
    ),
    "strategy_query_too_similar": (
        "Strategy final queries and BM25 topK overlap are too similar; metric spread "
        "is zero despite query rewrites. Improve strategy prompt differentiation."
    ),
    "metric_has_quality_signal": (
        "NDCG/Recall/MRR at some K produces group spread. Candidate metric K exists "
        "for reward dry-run after Phase 1.18f."
    ),
}


def _metric_key(metric: str, k: int) -> str:
    return f"{metric}@{k}"


def _values_spread(values: List[float]) -> float:
    return _spread(values)


def _best_relevant_rank(retrieved: Sequence[str], targets: Sequence[str]) -> Optional[int]:
    target_set = set(targets)
    if not target_set:
        return None
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in target_set:
            return rank
    return None


def _count_relevant_in_topk(retrieved: Sequence[str], targets: Sequence[str], k: int) -> int:
    target_set = set(targets)
    if not target_set:
        return 0
    return sum(1 for doc_id in list(retrieved)[:k] if doc_id in target_set)


class QrelsMetricBlindness:
    """
    Phase 1.18e qrels / metric blindness analyzer.

    Uses Phase 1.18d strategy rollout records plus BM25 re-retrieval to diagnose
    whether IR metrics can support GRPO group advantage from retrieval quality.
    """

    def __init__(self, k_list: Optional[List[int]] = None):
        self.k_list = sorted(set(k_list or DEFAULT_K_LIST))
        self.max_k = max(self.k_list)
        self._search_cache: Dict[Tuple[str, int], List[str]] = {}

    def load_inputs(
        self,
        rollout_path: str | Path,
        group_summary_path: Optional[str | Path] = None,
        reward_source_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        records = self.load_rollouts(rollout_path)
        group_summaries = self._load_jsonl(group_summary_path) if group_summary_path else []
        reward_source_reports = self._load_jsonl(reward_source_path) if reward_source_path else []
        return {
            "records": records,
            "group_summaries": {r["group_id"]: r for r in group_summaries},
            "reward_source_reports": {r["group_id"]: r for r in reward_source_reports},
        }

    @staticmethod
    def load_rollouts(rollout_path: str | Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with Path(rollout_path).open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _retrieve_doc_ids(self, search_tool: Any, query: str, topk: int) -> List[str]:
        key = (query, topk)
        if key not in self._search_cache:
            self._search_cache[key] = search_tool.retrieved_ids(query, topk=topk)
        return self._search_cache[key]

    def analyze_query_coverage(
        self,
        group_id: str,
        original_query: str,
        target_items: Sequence[str],
        rel_scores: Optional[Sequence[float]],
        search_tool: Any,
    ) -> Dict[str, Any]:
        retrieved = self._retrieve_doc_ids(search_tool, original_query, self.max_k)
        best_rank = _best_relevant_rank(retrieved, target_items)
        coverage = {
            f"relevant_in_bm25_top{k}": _count_relevant_in_topk(retrieved, target_items, k)
            for k in COVERAGE_KS
        }

        num_relevant = len(target_items)
        num_highly_relevant = num_relevant
        if rel_scores is not None:
            num_highly_relevant = sum(1 for s in rel_scores if float(s) >= 2.0)

        qrels_sparse = num_relevant <= 1 or (
            num_relevant <= 3 and (best_rank is None or best_rank > 100)
        )
        bm25_can_retrieve = best_rank is not None and best_rank <= 1000
        ndcg10 = compute_metrics_for_retrieved(
            retrieved, target_items, rel_scores, [10]
        ).get("ndcg@10", 0.0)
        ndcg100 = compute_metrics_for_retrieved(
            retrieved, target_items, rel_scores, [100]
        ).get("ndcg@100", 0.0)

        return {
            "group_id": group_id,
            "original_query": original_query,
            "num_relevant_docs": num_relevant,
            "num_highly_relevant_docs": num_highly_relevant,
            **coverage,
            "best_relevant_rank": best_rank,
            "qrels_sparse": qrels_sparse,
            "bm25_can_retrieve_relevant": bm25_can_retrieve,
            "ndcg10_blind": float(ndcg10) <= EPS,
            "ndcg100_has_signal": float(ndcg100) > EPS,
            "original_query_ndcg_at_10": float(ndcg10),
            "original_query_ndcg_at_100": float(ndcg100),
            "original_query_recall_at_100": float(
                compute_recall(retrieved, list(target_items), 100)
            ),
        }

    def analyze_strategy_final_metrics(
        self,
        record: Dict[str, Any],
        search_tool: Any,
    ) -> Dict[str, Any]:
        traj = record["trajectory"]
        target_items = traj.get("target_items", [])
        rel_scores = traj.get("rel_scores")
        final_query = (
            record.get("extra_info", {}).get("final_query")
            or traj.get("final_query")
            or traj.get("user_query")
            or ""
        )
        strategy_name = record.get("strategy_name") or record.get("extra_info", {}).get(
            "strategy_name", "unknown"
        )

        retrieved = self._retrieve_doc_ids(search_tool, str(final_query), self.max_k)
        metrics = compute_metrics_for_retrieved(
            retrieved, target_items, rel_scores, self.k_list
        )

        row: Dict[str, Any] = {
            "group_id": record.get("group_id"),
            "sample_id": record.get("sample_id"),
            "strategy_name": strategy_name,
            "final_query": final_query,
            "target_items_count": len(target_items),
            "retrieved_count": len(retrieved),
            "metrics": metrics,
        }
        for k in self.k_list:
            row[_metric_key("ndcg", k)] = float(metrics.get(f"ndcg@{k}", 0.0))
            row[_metric_key("recall", k)] = float(metrics.get(f"recall@{k}", 0.0))
            row[_metric_key("mrr", k)] = float(metrics.get(f"mrr@{k}", 0.0))
            row[_metric_key("hit", k)] = float(metrics.get(f"hit@{k}", 0.0))
        return row

    def analyze_group_metric_spread(
        self,
        group_id: str,
        records: List[Dict[str, Any]],
        strategy_metrics: List[Dict[str, Any]],
        query_coverage: Dict[str, Any],
        reward_source_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        strategies = [r.get("strategy_name", "unknown") for r in records]
        metric_values: Dict[str, List[float]] = {}
        for metric in ("ndcg", "recall", "mrr", "hit"):
            for k in self.k_list:
                key = _metric_key(metric, k)
                metric_values[key] = [float(row.get(key, 0.0)) for row in strategy_metrics]

        spreads = {f"{key}_spread": _values_spread(vals) for key, vals in metric_values.items()}
        stds = {f"{key}_std": _std(vals) for key, vals in metric_values.items()}

        final_queries = [str(r.get("final_query", "")) for r in strategy_metrics]
        retrieved_sets = {
            k: [self._search_cache.get((str(r["final_query"]), self.max_k), [])[:k] for r in strategy_metrics]
            for k in self.k_list
        }

        avg_pairwise_final_query_jaccard = _pairwise_average(final_queries, token_jaccard)
        overlap_by_k = {}
        for k in self.k_list:
            sets = retrieved_sets[k]
            overlap_by_k[f"avg_pairwise_top{k}_overlap"] = (
                _pairwise_average(sets, _set_jaccard) if len(sets) >= 2 else 1.0
            )

        metric_blindness_type = self.classify_group(
            metric_values=metric_values,
            spreads=spreads,
            query_coverage=query_coverage,
            avg_pairwise_final_query_jaccard=avg_pairwise_final_query_jaccard,
            overlap_by_k=overlap_by_k,
        )

        report = {
            "group_id": group_id,
            "original_query": query_coverage.get("original_query"),
            "strategies": strategies,
            "group_size": len(records),
            "unique_final_query_count": len(set(final_queries)),
            "avg_pairwise_final_query_jaccard": avg_pairwise_final_query_jaccard,
            **overlap_by_k,
            **{f"{k}_values": v for k, v in metric_values.items()},
            **spreads,
            **stds,
            "metric_blindness_type": metric_blindness_type,
            "metric_blindness_recommendation": METRIC_BLINDNESS_TYPES[metric_blindness_type],
            "query_coverage": {
                k: query_coverage[k]
                for k in (
                    "num_relevant_docs",
                    "best_relevant_rank",
                    "qrels_sparse",
                    "bm25_can_retrieve_relevant",
                    "relevant_in_bm25_top1000",
                )
            },
        }

        if reward_source_report:
            report["phase119a_spread_source"] = reward_source_report.get("spread_source")
            report["phase119a_quality_only_spread"] = reward_source_report.get(
                "quality_only_reward_spread"
            )

        best_k_candidates = self._best_k_candidates(spreads)
        report["candidate_reward_k"] = best_k_candidates
        return report

    def classify_group(
        self,
        metric_values: Dict[str, List[float]],
        spreads: Dict[str, float],
        query_coverage: Dict[str, Any],
        avg_pairwise_final_query_jaccard: float,
        overlap_by_k: Dict[str, Optional[float]],
    ) -> str:
        ndcg10_spread = spreads.get("ndcg@10_spread", 0.0)
        recall100_spread = spreads.get("recall@100_spread", 0.0)
        recall1000_spread = spreads.get("recall@1000_spread", 0.0)
        ndcg100_spread = spreads.get("ndcg@100_spread", 0.0)
        ndcg1000_spread = spreads.get("ndcg@1000_spread", 0.0)
        mrr100_spread = spreads.get("mrr@100_spread", 0.0)
        mrr1000_spread = spreads.get("mrr@1000_spread", 0.0)

        large_k_spread = any(
            v > EPS
            for v in (
                ndcg100_spread,
                ndcg1000_spread,
                recall100_spread,
                recall1000_spread,
                mrr100_spread,
                mrr1000_spread,
            )
        )
        any_quality_spread = any(
            spreads.get(f"{metric}@{k}_spread", 0.0) > EPS
            for metric in ("ndcg", "recall", "mrr")
            for k in self.k_list
        )

        if ndcg10_spread <= EPS and large_k_spread:
            return "small_k_blind_large_k_signal"

        if any_quality_spread:
            return "metric_has_quality_signal"

        if (
            query_coverage.get("num_relevant_docs", 0) > 0
            and not query_coverage.get("bm25_can_retrieve_relevant", False)
        ):
            return "bm25_retrieval_failure"

        top100_overlap = overlap_by_k.get("avg_pairwise_top100_overlap")
        if (
            avg_pairwise_final_query_jaccard > 0.85
            and (top100_overlap is None or top100_overlap > 0.8)
        ):
            return "strategy_query_too_similar"

        return "qrels_sparse_all_k_blind"

    def _best_k_candidates(self, spreads: Dict[str, float]) -> List[str]:
        candidates: List[Tuple[float, str]] = []
        for metric in ("ndcg", "recall", "mrr"):
            for k in self.k_list:
                key = f"{metric}@{k}_spread"
                spread = spreads.get(key, 0.0)
                if spread > EPS:
                    candidates.append((spread, f"{metric}@{k}"))
        candidates.sort(reverse=True)
        return [name for _, name in candidates[:5]]

    def analyze_all(
        self,
        inputs: Dict[str, Any],
        search_tool: Any,
    ) -> Dict[str, Any]:
        records = inputs["records"]
        reward_source_reports = inputs.get("reward_source_reports", {})

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[record.get("group_id", "unknown")].append(record)
        for gid in grouped:
            grouped[gid].sort(key=lambda r: r.get("group_index", 0))

        query_coverage_rows: List[Dict[str, Any]] = []
        metric_by_k_rows: List[Dict[str, Any]] = []
        group_spread_rows: List[Dict[str, Any]] = []

        for group_id in sorted(grouped.keys()):
            group_records = grouped[group_id]
            traj0 = group_records[0]["trajectory"]
            original_query = traj0.get("user_query") or group_records[0]["extra_info"].get(
                "original_query", ""
            )
            target_items = traj0.get("target_items", [])
            rel_scores = traj0.get("rel_scores")

            coverage = self.analyze_query_coverage(
                group_id=group_id,
                original_query=original_query,
                target_items=target_items,
                rel_scores=rel_scores,
                search_tool=search_tool,
            )
            query_coverage_rows.append(coverage)

            strategy_metrics = [
                self.analyze_strategy_final_metrics(record, search_tool)
                for record in group_records
            ]
            for row in strategy_metrics:
                slim = {
                    k: v
                    for k, v in row.items()
                    if k not in ("metrics",)
                }
                metric_by_k_rows.append(slim)

            group_spread = self.analyze_group_metric_spread(
                group_id=group_id,
                records=group_records,
                strategy_metrics=strategy_metrics,
                query_coverage=coverage,
                reward_source_report=reward_source_reports.get(group_id),
            )
            group_spread_rows.append(group_spread)

        qrels_summary = self.build_qrels_summary(query_coverage_rows, group_spread_rows)
        summary = self.build_summary(
            query_coverage_rows=query_coverage_rows,
            group_spread_rows=group_spread_rows,
            num_records=len(records),
        )
        return {
            "query_coverage_rows": query_coverage_rows,
            "metric_by_k_rows": metric_by_k_rows,
            "group_spread_rows": group_spread_rows,
            "qrels_summary": qrels_summary,
            "summary": summary,
        }

    def build_qrels_summary(
        self,
        query_coverage_rows: List[Dict[str, Any]],
        group_spread_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        num_groups = len(query_coverage_rows)
        type_counts: Dict[str, int] = defaultdict(int)
        for row in group_spread_rows:
            type_counts[row["metric_blindness_type"]] += 1

        return {
            "num_groups": num_groups,
            "num_queries": num_groups,
            "mean_num_relevant_docs": mean(r["num_relevant_docs"] for r in query_coverage_rows),
            "min_num_relevant_docs": min(r["num_relevant_docs"] for r in query_coverage_rows),
            "max_num_relevant_docs": max(r["num_relevant_docs"] for r in query_coverage_rows),
            "qrels_sparse_query_count": sum(1 for r in query_coverage_rows if r["qrels_sparse"]),
            "qrels_sparse_query_rate": sum(1 for r in query_coverage_rows if r["qrels_sparse"]) / max(
                1, num_groups
            ),
            "bm25_can_retrieve_query_count": sum(
                1 for r in query_coverage_rows if r["bm25_can_retrieve_relevant"]
            ),
            "ndcg10_blind_query_count": sum(1 for r in query_coverage_rows if r["ndcg10_blind"]),
            "ndcg100_has_signal_query_count": sum(
                1 for r in query_coverage_rows if r["ndcg100_has_signal"]
            ),
            "metric_blindness_type_counts": dict(type_counts),
            "metric_blindness_type_rates": {
                k: v / max(1, num_groups) for k, v in type_counts.items()
            },
            "per_query": query_coverage_rows,
        }

    def build_summary(
        self,
        query_coverage_rows: List[Dict[str, Any]],
        group_spread_rows: List[Dict[str, Any]],
        num_records: int,
    ) -> Dict[str, Any]:
        num_groups = len(group_spread_rows)
        type_counts: Dict[str, int] = defaultdict(int)
        for row in group_spread_rows:
            type_counts[row["metric_blindness_type"]] += 1

        def mean_spread(suffix: str) -> float:
            vals = [float(r.get(suffix, 0.0)) for r in group_spread_rows]
            return mean(vals) if vals else 0.0

        has_signal_rate = type_counts.get("metric_has_quality_signal", 0) / max(1, num_groups)
        small_k_rate = type_counts.get("small_k_blind_large_k_signal", 0) / max(1, num_groups)
        sparse_rate = type_counts.get("qrels_sparse_all_k_blind", 0) / max(1, num_groups)
        bm25_fail_rate = type_counts.get("bm25_retrieval_failure", 0) / max(1, num_groups)
        similar_rate = type_counts.get("strategy_query_too_similar", 0) / max(1, num_groups)

        if has_signal_rate >= 0.4 or small_k_rate >= 0.4:
            next_phase = "Phase 1.18f: Large-K Reward Candidate Dry-Run"
            recommendation = (
                "Some groups show retrieval-quality spread at larger K. "
                "Proceed to Phase 1.18f reward candidate dry-run, then re-run Phase 1.19a gate."
            )
        elif sparse_rate >= 0.6 or bm25_fail_rate >= 0.4:
            next_phase = "Phase 1.18g: Smoke Set Expansion / Query Selection"
            recommendation = (
                "Most groups lack metric signal due to sparse qrels or BM25 retrieval failure. "
                "Expand or re-select smoke queries before reward/prompt fixes."
            )
        elif similar_rate >= 0.4:
            next_phase = "Phase 1.18h: Strategy Prompt V2"
            recommendation = (
                "Strategy queries collapse to similar final queries with no metric spread. "
                "Improve strategy prompt differentiation before changing reward."
            )
        else:
            next_phase = "Phase 1.18g or 1.18h depending on per-group reports"
            recommendation = (
                "Mixed metric blindness patterns. Inspect metric_blindness_report.md per group."
            )

        return {
            "phase": "1.18e",
            "num_groups": num_groups,
            "num_rollout_records": num_records,
            "k_list": self.k_list,
            "metric_blindness_type_counts": dict(type_counts),
            "metric_has_quality_signal_group_rate": has_signal_rate,
            "small_k_blind_large_k_signal_group_rate": small_k_rate,
            "qrels_sparse_all_k_blind_group_rate": sparse_rate,
            "bm25_retrieval_failure_group_rate": bm25_fail_rate,
            "strategy_query_too_similar_group_rate": similar_rate,
            "mean_ndcg10_spread": mean_spread("ndcg@10_spread"),
            "mean_ndcg100_spread": mean_spread("ndcg@100_spread"),
            "mean_ndcg1000_spread": mean_spread("ndcg@1000_spread"),
            "mean_recall100_spread": mean_spread("recall@100_spread"),
            "mean_recall1000_spread": mean_spread("recall@1000_spread"),
            "mean_mrr100_spread": mean_spread("mrr@100_spread"),
            "qrels_sparse_query_rate": sum(1 for r in query_coverage_rows if r["qrels_sparse"]) / max(
                1, num_groups
            ),
            "bm25_can_retrieve_query_rate": sum(
                1 for r in query_coverage_rows if r["bm25_can_retrieve_relevant"]
            )
            / max(1, num_groups),
            "recommended_next_phase": next_phase,
            "recommendation": recommendation,
            "is_training": False,
        }


def build_metric_blindness_report(
    query_coverage_rows: List[Dict[str, Any]],
    group_spread_rows: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> str:
    lines = [
        "# Phase 1.18e Metric Blindness Report",
        "",
        "## Executive Summary",
        "",
        f"- Groups analyzed: **{summary['num_groups']}**",
        f"- metric_has_quality_signal: **{summary['metric_has_quality_signal_group_rate']:.2f}**",
        f"- small_k_blind_large_k_signal: **{summary['small_k_blind_large_k_signal_group_rate']:.2f}**",
        f"- qrels_sparse_all_k_blind: **{summary['qrels_sparse_all_k_blind_group_rate']:.2f}**",
        f"- bm25_retrieval_failure: **{summary['bm25_retrieval_failure_group_rate']:.2f}**",
        f"- strategy_query_too_similar: **{summary['strategy_query_too_similar_group_rate']:.2f}**",
        f"- Recommended next phase: **{summary['recommended_next_phase']}**",
        "",
        summary["recommendation"],
        "",
        "## Q1-Q8 Answers",
        "",
        "### Q1: How many relevant docs per ESCI query?",
        "",
    ]

    for row in query_coverage_rows:
        lines.append(
            f"- `{row['group_id']}`: **{row['num_relevant_docs']}** relevant docs "
            f"(highly relevant: {row['num_highly_relevant_docs']})"
        )

    lines.extend(
        [
            "",
            "### Q2: Are smoke qrels too sparse?",
            "",
            f"- qrels_sparse_query_rate: **{summary['qrels_sparse_query_rate']:.2f}**",
            "",
            "### Q3: Are relevant docs in BM25 top100 / top1000?",
            "",
        ]
    )
    for row in query_coverage_rows:
        lines.append(
            f"- `{row['group_id']}`: top10={row['relevant_in_bm25_top10']}, "
            f"top50={row['relevant_in_bm25_top50']}, top100={row['relevant_in_bm25_top100']}, "
            f"top1000={row['relevant_in_bm25_top1000']}, best_rank={row['best_relevant_rank']}"
        )

    lines.extend(["", "### Q4: When NDCG@10=0, does larger K have signal?", ""])
    for group in group_spread_rows:
        ndcg10 = group.get("ndcg@10_values", [])
        ndcg100 = group.get("ndcg@100_values", [])
        ndcg1000 = group.get("ndcg@1000_values", [])
        lines.append(
            f"- `{group['group_id']}`: ndcg@10={ndcg10}, ndcg@100={ndcg100}, "
            f"ndcg@1000={ndcg1000}, type=`{group['metric_blindness_type']}`"
        )

    lines.extend(["", "### Q5-Q6: Recall/MRR group spread by K", ""])
    for group in group_spread_rows:
        lines.append(
            f"- `{group['group_id']}`: recall@100_spread={group.get('recall@100_spread', 0):.4f}, "
            f"recall@1000_spread={group.get('recall@1000_spread', 0):.4f}, "
            f"mrr@100_spread={group.get('mrr@100_spread', 0):.4f}, "
            f"candidate_k={group.get('candidate_reward_k', [])}"
        )

    lines.extend(["", "### Q7: Why metric unchanged when topK changes?", ""])
    for group in group_spread_rows:
        overlap100 = group.get("avg_pairwise_top100_overlap")
        overlap_str = f"{overlap100:.3f}" if overlap100 is not None else "n/a"
        jaccard = group.get("avg_pairwise_final_query_jaccard", 0.0)
        lines.append(
            f"- `{group['group_id']}`: final_query_jaccard={jaccard:.3f}, "
            f"top100_overlap={overlap_str}, "
            f"type=`{group['metric_blindness_type']}` — "
            f"{group['metric_blindness_recommendation']}"
        )

    lines.extend(
        [
            "",
            "### Q8: Metric K too small, qrels wrong, or bad smoke sample?",
            "",
            f"- **Recommendation:** {summary['recommended_next_phase']}",
            "",
            "## Per-Group Classification",
            "",
        ]
    )
    for group in group_spread_rows:
        lines.extend(
            [
                f"### {group['group_id']}",
                "",
                f"- Type: `{group['metric_blindness_type']}`",
                f"- Original query: `{group.get('original_query')}`",
                f"- Strategies: {', '.join(group.get('strategies', []))}",
                f"- unique_final_query_count: {group.get('unique_final_query_count')}",
                f"- ndcg@10_spread: {group.get('ndcg@10_spread', 0):.4f}",
                f"- ndcg@100_spread: {group.get('ndcg@100_spread', 0):.4f}",
                f"- recall@100_spread: {group.get('recall@100_spread', 0):.4f}",
                "",
            ]
        )

    return "\n".join(lines)
