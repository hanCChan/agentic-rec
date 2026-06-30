"""
Phase 1.18g: BM25 failure / unlearnable sample cleanup.

Identifies queries that cannot provide useful BM25-based retrieval-quality reward
signal and optionally proposes replacement candidates from ESCI split.

Does NOT train or modify official reward.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.agents.episode_runner import load_esci_samples
from src.agents.qrels_metric_blindness import (
    EPS,
    _best_relevant_rank,
    _count_relevant_in_topk,
)

LEARNABILITY_TYPES = (
    "learnable_large_k",
    "learnable_small_k",
    "bm25_retrieval_failure",
    "qrels_sparse_all_k_blind",
    "strategy_collapse",
    "ambiguous_keep_for_analysis",
)

NEXT_ACTIONS = {
    "learnable_large_k": "keep_for_phase2",
    "learnable_small_k": "keep_for_phase2",
    "bm25_retrieval_failure": "replace_sample",
    "qrels_sparse_all_k_blind": "replace_sample",
    "strategy_collapse": "fix_strategy_prompt_phase118h",
    "ambiguous_keep_for_analysis": "manual_review",
}

FAILURE_REASONS = {
    "bm25_retrieval_failure": (
        "Relevant documents exist in qrels, but BM25 top1000 did not retrieve "
        "any relevant document."
    ),
    "qrels_sparse_all_k_blind": (
        "Qrels are sparse and NDCG/Recall/MRR show no group spread at any K."
    ),
    "strategy_collapse": (
        "Strategy final queries collapsed to identical or highly similar queries, "
        "producing zero metric spread."
    ),
}


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class BM25FailureCleanup:
    """
    Phase 1.18g BM25 failure / unlearnable sample cleanup.

    This class identifies queries that cannot provide useful BM25-based
    retrieval-quality reward signal, and optionally proposes replacement
    candidates from the available ESCI split.

    It does NOT train and does NOT modify the official reward.
    """

    def __init__(
        self,
        k_list: Optional[List[int]] = None,
        min_relevant_docs: int = 1,
        require_bm25_hit_at_1000: bool = True,
    ):
        self.k_list = sorted(set(k_list or [10, 50, 100, 1000]))
        self.min_relevant_docs = min_relevant_docs
        self.require_bm25_hit_at_1000 = require_bm25_hit_at_1000

    def load_existing_reports(
        self,
        rollout_path: str | Path,
        query_coverage_path: str | Path,
        group_metric_path: str | Path,
        large_k_group_report_path: str | Path,
        candidate_name: str = "reward_largek_mix_1000",
    ) -> Dict[str, Any]:
        rollout_records = _load_jsonl(rollout_path)
        coverage_by_group = {r["group_id"]: r for r in _load_jsonl(query_coverage_path)}
        metric_by_group = {r["group_id"]: r for r in _load_jsonl(group_metric_path)}

        large_k_rows = _load_jsonl(large_k_group_report_path)
        large_k_by_group = {
            r["group_id"]: r
            for r in large_k_rows
            if r.get("candidate_name") == candidate_name
        }

        grouped_queries: Dict[str, str] = {}
        for record in rollout_records:
            gid = record.get("group_id")
            if gid and gid not in grouped_queries:
                grouped_queries[gid] = (
                    record.get("extra_info", {}).get("original_query")
                    or record["trajectory"].get("user_query", "")
                )

        group_ids = sorted(set(coverage_by_group.keys()) | set(metric_by_group.keys()))
        return {
            "rollout_records": rollout_records,
            "coverage_by_group": coverage_by_group,
            "metric_by_group": metric_by_group,
            "large_k_by_group": large_k_by_group,
            "group_ids": group_ids,
            "grouped_queries": grouped_queries,
            "candidate_name": candidate_name,
        }

    def classify_group_learnability(self, group_report: Dict[str, Any]) -> Dict[str, Any]:
        coverage = group_report.get("query_coverage", group_report)
        metric = group_report.get("metric_report", {})
        large_k = group_report.get("large_k_report", {})

        group_id = group_report["group_id"]
        original_query = group_report.get("original_query") or coverage.get("original_query", "")

        num_relevant = int(coverage.get("num_relevant_docs", 0))
        best_rank = coverage.get("best_relevant_rank")
        bm25_hit_at_1000 = bool(coverage.get("relevant_in_bm25_top1000", 0) > 0) or (
            best_rank is not None and best_rank <= 1000
        )
        qrels_sparse = bool(coverage.get("qrels_sparse", False))

        ndcg10_spread = float(metric.get("ndcg@10_spread", 0.0))
        recall50_spread = float(metric.get("recall@50_spread", 0.0))
        ndcg1000_spread = float(metric.get("ndcg@1000_spread", 0.0))
        recall1000_spread = float(metric.get("recall@1000_spread", 0.0))
        mrr1000_spread = float(metric.get("mrr@1000_spread", 0.0))
        largek_reward_spread = float(
            large_k.get("candidate_reward_spread", large_k.get("retrieval_quality_spread", 0.0))
        )

        unique_final_query_count = int(metric.get("unique_final_query_count", 4))
        avg_jaccard = float(metric.get("avg_pairwise_final_query_jaccard", 0.0))
        metric_blindness_type = metric.get("metric_blindness_type", "")

        any_k_spread = any(
            float(metric.get(f"{m}@{k}_spread", 0.0)) > EPS
            for m in ("ndcg", "recall", "mrr")
            for k in self.k_list
        )
        large_k_metric_spread = any(
            v > EPS for v in (ndcg1000_spread, recall1000_spread, mrr1000_spread, largek_reward_spread)
        )

        learnability_type = "ambiguous_keep_for_analysis"
        failure_reason = ""

        if num_relevant >= self.min_relevant_docs and not bm25_hit_at_1000:
            learnability_type = "bm25_retrieval_failure"
            failure_reason = FAILURE_REASONS["bm25_retrieval_failure"]
        elif (
            unique_final_query_count <= 1
            or metric_blindness_type == "strategy_query_too_similar"
            or (avg_jaccard > 0.85 and not any_k_spread)
        ):
            learnability_type = "strategy_collapse"
            failure_reason = FAILURE_REASONS["strategy_collapse"]
        elif metric_blindness_type == "qrels_sparse_all_k_blind" or (
            qrels_sparse and not any_k_spread
        ):
            learnability_type = "qrels_sparse_all_k_blind"
            failure_reason = FAILURE_REASONS["qrels_sparse_all_k_blind"]
        elif ndcg10_spread > EPS or recall50_spread > EPS:
            learnability_type = "learnable_small_k"
        elif bm25_hit_at_1000 and large_k_metric_spread:
            learnability_type = "learnable_large_k"
        elif bm25_hit_at_1000 and metric_blindness_type == "small_k_blind_large_k_signal":
            learnability_type = "learnable_large_k"

        keep_for_phase2 = learnability_type in {"learnable_large_k", "learnable_small_k"}
        replace_recommended = learnability_type in {
            "bm25_retrieval_failure",
            "qrels_sparse_all_k_blind",
        }

        return {
            "group_id": group_id,
            "original_query": original_query,
            "learnability_type": learnability_type,
            "keep_for_phase2": keep_for_phase2,
            "replace_recommended": replace_recommended,
            "num_relevant_docs": num_relevant,
            "best_relevant_rank": best_rank,
            "bm25_hit_at_1000": bm25_hit_at_1000,
            "largek_reward_spread": largek_reward_spread,
            "ndcg1000_spread": ndcg1000_spread,
            "recall1000_spread": recall1000_spread,
            "mrr1000_spread": mrr1000_spread,
            "ndcg10_spread": ndcg10_spread,
            "recall50_spread": recall50_spread,
            "unique_final_query_count": unique_final_query_count,
            "avg_pairwise_final_query_jaccard": avg_jaccard,
            "metric_blindness_type": metric_blindness_type,
            "failure_reason": failure_reason,
            "next_action": NEXT_ACTIONS[learnability_type],
        }

    def summarize_unlearnable_groups(self, group_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        counts: Dict[str, int] = {t: 0 for t in LEARNABILITY_TYPES}
        for row in group_reports:
            counts[row["learnability_type"]] = counts.get(row["learnability_type"], 0) + 1

        num_keep = sum(1 for r in group_reports if r["keep_for_phase2"])
        num_replace = sum(1 for r in group_reports if r["replace_recommended"])

        return {
            "num_input_groups": len(group_reports),
            "num_keep_for_phase2": num_keep,
            "num_replace_recommended": num_replace,
            "learnability_counts": counts,
            "bm25_failure_count": counts.get("bm25_retrieval_failure", 0),
            "qrels_sparse_all_k_blind_count": counts.get("qrels_sparse_all_k_blind", 0),
            "strategy_collapse_count": counts.get("strategy_collapse", 0),
        }

    def build_replacement_pool(
        self,
        candidate_samples: List[Dict[str, Any]],
        search_tool: Any,
        exclude_queries: Sequence[str],
    ) -> List[Dict[str, Any]]:
        exclude = {q.strip().lower() for q in exclude_queries if q}
        pool: List[Dict[str, Any]] = []

        for sample in candidate_samples:
            query = str(sample.get("user_query", "")).strip()
            if not query or query.lower() in exclude:
                continue

            target_items = [str(x) for x in sample.get("target_items", [])]
            if len(target_items) < self.min_relevant_docs:
                continue

            retrieved = search_tool.retrieved_ids(query, topk=1000)
            best_rank = _best_relevant_rank(retrieved, target_items)
            if best_rank is None or best_rank > 1000:
                continue

            scored = self.score_replacement_candidate(
                {
                    "group_id": sample.get("qid", sample.get("group_id", "unknown")),
                    "original_query": query,
                    "target_items": target_items,
                    "best_relevant_rank": best_rank,
                    "num_relevant_docs": len(target_items),
                    "relevant_in_bm25_top100": _count_relevant_in_topk(retrieved, target_items, 100),
                    "relevant_in_bm25_top1000": _count_relevant_in_topk(
                        retrieved, target_items, 1000
                    ),
                }
            )
            pool.append(scored)

        pool.sort(key=lambda r: r["candidate_score"], reverse=True)
        return pool

    def score_replacement_candidate(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        has_top100 = int(sample.get("relevant_in_bm25_top100", 0) > 0)
        has_top1000 = int(sample.get("relevant_in_bm25_top1000", 0) > 0)
        num_rel = int(sample.get("num_relevant_docs", 0))
        best_rank = sample.get("best_relevant_rank")

        is_too_easy_top1 = bool(
            best_rank == 1 and num_rel >= 8
        )

        candidate_score = (
            1.0 * has_top100
            + 0.5 * has_top1000
            + 0.2 * min(num_rel, 5) / 5.0
            - 0.1 * float(is_too_easy_top1)
        )

        return {
            **sample,
            "candidate_score": round(candidate_score, 4),
            "has_relevant_in_top100": bool(has_top100),
            "has_relevant_in_top1000": bool(has_top1000),
            "is_too_easy_top1": is_too_easy_top1,
            "bm25_hit_at_1000": True,
        }

    def select_replacements(self, pool: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
        seen_queries: set[str] = set()
        selected: List[Dict[str, Any]] = []
        for row in pool:
            q = row["original_query"].strip().lower()
            if q in seen_queries:
                continue
            seen_queries.add(q)
            selected.append(row)
            if len(selected) >= n:
                break
        return selected

    def export_clean_smoke_set(
        self,
        kept_groups: List[Dict[str, Any]],
        replacements: List[Dict[str, Any]],
        replace_targets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        clean_kept = [
            {
                "group_id": r["group_id"],
                "original_query": r["original_query"],
                "source": "kept_from_20_g4",
                "learnability_type": r["learnability_type"],
                "num_relevant_docs": r["num_relevant_docs"],
                "best_relevant_rank": r["best_relevant_rank"],
                "bm25_hit_at_1000": r["bm25_hit_at_1000"],
                "recommended_for_phase2": r["keep_for_phase2"],
                "next_action": r["next_action"],
            }
            for r in kept_groups
            if r["keep_for_phase2"]
        ]

        replacement_rows = []
        for target, repl in zip(replace_targets, replacements):
            replacement_rows.append(
                {
                    "group_id": repl.get("group_id", f"replacement_{len(replacement_rows)}"),
                    "original_query": repl["original_query"],
                    "source": "replacement_candidate",
                    "replaces_group_id": target["group_id"],
                    "learnability_type": "learnable_large_k",
                    "num_relevant_docs": repl["num_relevant_docs"],
                    "best_relevant_rank": repl["best_relevant_rank"],
                    "bm25_hit_at_1000": repl["bm25_hit_at_1000"],
                    "recommended_for_phase2": True,
                    "candidate_score": repl["candidate_score"],
                    "next_action": "keep_for_phase2",
                }
            )

        phase2_candidates = clean_kept + replacement_rows
        strategy_collapse_remaining = any(
            r["learnability_type"] == "strategy_collapse" for r in kept_groups
        )

        blocking_reason = None
        phase2_ready = False
        if strategy_collapse_remaining:
            blocking_reason = (
                "strategy_collapse remains; run Phase 1.18h before Phase 2 training."
            )
        elif len(phase2_candidates) < len(kept_groups) + len(replace_targets):
            blocking_reason = "Insufficient replacement candidates for unlearnable groups."
        else:
            phase2_ready = len(replacement_rows) == len(replace_targets) and not strategy_collapse_remaining

        return {
            "clean_smoke_groups": clean_kept,
            "replacement_rows": replacement_rows,
            "phase2_candidate_smoke_set": phase2_candidates,
            "phase2_candidate_set_size": len(phase2_candidates),
            "phase2_candidate_ready": phase2_ready,
            "blocking_reason": blocking_reason,
        }

    def run(
        self,
        rollout_path: str | Path,
        query_coverage_path: str | Path,
        group_metric_path: str | Path,
        large_k_group_report_path: str | Path,
        output_dir: str | Path,
        *,
        data_path: str | Path,
        replacement_pool_size: int = 200,
        target_clean_groups: int = 20,
        search_tool: Any = None,
        candidate_name: str = "reward_largek_mix_1000",
    ) -> Dict[str, Any]:
        inputs = self.load_existing_reports(
            rollout_path=rollout_path,
            query_coverage_path=query_coverage_path,
            group_metric_path=group_metric_path,
            large_k_group_report_path=large_k_group_report_path,
            candidate_name=candidate_name,
        )

        group_reports: List[Dict[str, Any]] = []
        for group_id in inputs["group_ids"]:
            merged = {
                "group_id": group_id,
                "original_query": inputs["grouped_queries"].get(
                    group_id,
                    inputs["coverage_by_group"].get(group_id, {}).get("original_query", ""),
                ),
                "query_coverage": inputs["coverage_by_group"].get(group_id, {}),
                "metric_report": inputs["metric_by_group"].get(group_id, {}),
                "large_k_report": inputs["large_k_by_group"].get(group_id, {}),
            }
            group_reports.append(self.classify_group_learnability(merged))

        summary_counts = self.summarize_unlearnable_groups(group_reports)
        replace_targets = [r for r in group_reports if r["replace_recommended"]]

        replacement_pool: List[Dict[str, Any]] = []
        replacement_candidates: List[Dict[str, Any]] = []
        if search_tool is not None:
            esci_samples = load_esci_samples(Path(data_path), replacement_pool_size + 50)
            exclude = list(inputs["grouped_queries"].values())
            replacement_pool = self.build_replacement_pool(
                esci_samples, search_tool, exclude_queries=exclude
            )
            replacement_candidates = self.select_replacements(
                replacement_pool, n=max(len(replace_targets), target_clean_groups)
            )

        export = self.export_clean_smoke_set(
            kept_groups=group_reports,
            replacements=replacement_candidates[: len(replace_targets)],
            replace_targets=replace_targets,
        )

        summary = {
            "phase": "1.18g",
            "mode": "bm25_failure_cleanup",
            **summary_counts,
            "replacement_pool_size": replacement_pool_size,
            "num_replacement_candidates": len(replacement_candidates),
            "num_replacements_applied": min(len(replace_targets), len(replacement_candidates)),
            "phase2_candidate_set_size": export["phase2_candidate_set_size"],
            "phase2_candidate_ready": export["phase2_candidate_ready"],
            "blocking_reason": export["blocking_reason"],
            "is_training": False,
            "input_rollout_path": str(rollout_path),
            "input_query_coverage_path": str(query_coverage_path),
            "input_group_metric_path": str(group_metric_path),
            "input_large_k_group_report_path": str(large_k_group_report_path),
        }

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self._write_jsonl(out / "group_cleanup_report.jsonl", group_reports)
        self._write_jsonl(out / "replacement_candidates.jsonl", replacement_candidates)
        self._write_jsonl(out / "clean_smoke_groups.jsonl", export["clean_smoke_groups"])
        self._write_jsonl(out / "phase2_candidate_smoke_set.jsonl", export["phase2_candidate_smoke_set"])

        cleanup_summary = {
            **summary_counts,
            "replace_target_group_ids": [r["group_id"] for r in replace_targets],
            "replacement_candidate_queries": [
                r["original_query"] for r in replacement_candidates[: len(replace_targets)]
            ],
        }
        (out / "bm25_failure_cleanup_summary.json").write_text(
            json.dumps(cleanup_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out / "bm25_failure_cleanup_report.md").write_text(
            build_bm25_failure_cleanup_report(summary, group_reports, export),
            encoding="utf-8",
        )
        readme = out / "README.md"
        if not readme.exists():
            readme.write_text(
                "# Phase 1.18g BM25 Failure / Unlearnable Sample Cleanup\n\n"
                "Diagnoses and marks BM25-only unlearnable queries before Phase 2 smoke training.\n",
                encoding="utf-8",
            )

        return {
            "summary": summary,
            "group_reports": group_reports,
            "replacement_pool": replacement_pool,
            "replacement_candidates": replacement_candidates,
            "export": export,
        }

    @staticmethod
    def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_bm25_failure_cleanup_report(
    summary: Dict[str, Any],
    group_reports: List[Dict[str, Any]],
    export: Dict[str, Any],
) -> str:
    lines = [
        "# Phase 1.18g BM25 Failure / Unlearnable Sample Cleanup Report",
        "",
        "## Summary",
        "",
        f"- num_input_groups: **{summary['num_input_groups']}**",
        f"- num_keep_for_phase2: **{summary['num_keep_for_phase2']}**",
        f"- num_replace_recommended: **{summary['num_replace_recommended']}**",
        f"- bm25_failure_count: **{summary['bm25_failure_count']}**",
        f"- qrels_sparse_all_k_blind_count: **{summary['qrels_sparse_all_k_blind_count']}**",
        f"- strategy_collapse_count: **{summary['strategy_collapse_count']}**",
        f"- phase2_candidate_set_size: **{summary['phase2_candidate_set_size']}**",
        f"- phase2_candidate_ready: **{summary['phase2_candidate_ready']}**",
        "",
        "## Learnability Counts",
        "",
    ]
    for k, v in summary["learnability_counts"].items():
        lines.append(f"- `{k}`: **{v}**")

    lines.extend(["", "## Per-Group Cleanup", ""])
    for row in group_reports:
        lines.append(
            f"- `{row['group_id']}`: **{row['learnability_type']}** "
            f"(keep={row['keep_for_phase2']}, action=`{row['next_action']}`)"
        )
        if row.get("failure_reason"):
            lines.append(f"  - {row['failure_reason']}")

    lines.extend(
        [
            "",
            "## Phase 2 Readiness",
            "",
            f"- blocking_reason: {summary.get('blocking_reason') or 'none'}",
            "",
            "## Next Steps",
            "",
        ]
    )
    if summary.get("strategy_collapse_count", 0) > 0:
        lines.append("1. Phase 1.18h — Strategy Prompt V2 for collapse cases")
    if summary.get("num_replace_recommended", 0) > 0:
        lines.append("2. Apply replacement candidates and re-run 20_g4 smoke set construction")
    lines.append("3. Phase 2.1 — Tiny GRPO smoke training only after cleanup + prompt fix")
    return "\n".join(lines) + "\n"
