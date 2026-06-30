#!/usr/bin/env python3
"""
Phase 1.18g: BM25 failure / unlearnable sample cleanup.

Marks BM25-only unlearnable queries and builds a cleaner Phase 2 smoke candidate set.
Does NOT train.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  python scripts/cleanup_bm25_failure_samples.py \
    --rollout-path experiments/phase119b_scale_gate_check/strategy_rollout_20_g4/rollout_records.jsonl \
    --query-coverage-path experiments/phase119b_scale_gate_check/qrels_metric_20_g4/query_relevance_coverage.jsonl \
    --group-metric-path experiments/phase119b_scale_gate_check/qrels_metric_20_g4/group_metric_spread_by_k.jsonl \
    --large-k-group-report-path experiments/phase119b_scale_gate_check/large_k_reward_20_g4/large_k_candidate_group_reports.jsonl \
    --output-dir experiments/phase118g_bm25_failure_cleanup_20_g4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REC_R1 = ROOT / "Rec-R1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REC_R1))

from src.agents.bm25_failure_cleanup import BM25FailureCleanup
from src.tools.bm25_tool import BM25SearchTool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.18g BM25 failure sample cleanup")
    parser.add_argument(
        "--rollout-path",
        type=Path,
        default=ROOT
        / "experiments/phase119b_scale_gate_check/strategy_rollout_20_g4/rollout_records.jsonl",
    )
    parser.add_argument(
        "--query-coverage-path",
        type=Path,
        default=ROOT
        / "experiments/phase119b_scale_gate_check/qrels_metric_20_g4/query_relevance_coverage.jsonl",
    )
    parser.add_argument(
        "--group-metric-path",
        type=Path,
        default=ROOT
        / "experiments/phase119b_scale_gate_check/qrels_metric_20_g4/group_metric_spread_by_k.jsonl",
    )
    parser.add_argument(
        "--large-k-group-report-path",
        type=Path,
        default=ROOT
        / "experiments/phase119b_scale_gate_check/large_k_reward_20_g4/large_k_candidate_group_reports.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/phase118g_bm25_failure_cleanup_20_g4",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "Rec-R1/data/esci/inst/sparse/subset/val.parquet",
    )
    parser.add_argument("--replacement-pool-size", type=int, default=200)
    parser.add_argument("--target-clean-groups", type=int, default=20)
    parser.add_argument("--k-list", type=int, nargs="+", default=[10, 50, 100, 1000])
    parser.add_argument("--candidate-name", type=str, default="reward_largek_mix_1000")
    parser.add_argument("--skip-bm25-pool", action="store_true", help="Skip BM25 replacement pool scan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cleanup = BM25FailureCleanup(k_list=args.k_list)
    search_tool = None if args.skip_bm25_pool else BM25SearchTool()

    result = cleanup.run(
        rollout_path=args.rollout_path,
        query_coverage_path=args.query_coverage_path,
        group_metric_path=args.group_metric_path,
        large_k_group_report_path=args.large_k_group_report_path,
        output_dir=args.output_dir,
        data_path=args.data_path,
        replacement_pool_size=args.replacement_pool_size,
        target_clean_groups=args.target_clean_groups,
        search_tool=search_tool,
        candidate_name=args.candidate_name,
    )

    summary = result["summary"]

    print("\n=== Phase 1.18g BM25 Failure / Unlearnable Sample Cleanup Summary ===")
    print(f"num_input_groups: {summary['num_input_groups']}")
    print(f"num_keep_for_phase2: {summary['num_keep_for_phase2']}")
    print(f"num_replace_recommended: {summary['num_replace_recommended']}")
    print(f"bm25_failure_count: {summary['bm25_failure_count']}")
    print(f"qrels_sparse_all_k_blind_count: {summary['qrels_sparse_all_k_blind_count']}")
    print(f"strategy_collapse_count: {summary['strategy_collapse_count']}")
    print(f"replacement_pool_size: {summary['replacement_pool_size']}")
    print(f"num_replacement_candidates: {summary['num_replacement_candidates']}")
    print(f"phase2_candidate_set_size: {summary['phase2_candidate_set_size']}")
    print(f"phase2_candidate_ready: {summary['phase2_candidate_ready']}")
    print(f"blocking_reason: {summary.get('blocking_reason')}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
