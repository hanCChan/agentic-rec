#!/usr/bin/env python3
"""
Phase 1.18e: Qrels / Metric Blindness analysis.

Diagnoses ESCI smoke qrels and BM25 retrieval metrics without training or reward changes.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  python scripts/analyze_qrels_metric_blindness.py \
    --rollout-path experiments/phase118d_strategy_rollout_5_g4/rollout_records.jsonl \
    --group-summary-path experiments/phase118d_strategy_rollout_5_g4/group_summaries.jsonl \
    --output-dir experiments/phase118e_qrels_metric_blindness_5_g4 \
    --k-list 10 50 100 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REC_R1 = ROOT / "Rec-R1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REC_R1))

from src.agents.qrels_metric_blindness import QrelsMetricBlindness, build_metric_blindness_report
from src.tools.bm25_tool import BM25SearchTool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.18e qrels / metric blindness analysis")
    parser.add_argument(
        "--rollout-path",
        type=Path,
        default=ROOT / "experiments/phase118d_strategy_rollout_5_g4/rollout_records.jsonl",
    )
    parser.add_argument(
        "--group-summary-path",
        type=Path,
        default=ROOT / "experiments/phase118d_strategy_rollout_5_g4/group_summaries.jsonl",
    )
    parser.add_argument(
        "--reward-source-path",
        type=Path,
        default=ROOT
        / "experiments/phase119a_strategy_reward_decomposition_5_g4/group_reward_source_report.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/phase118e_qrels_metric_blindness_5_g4",
    )
    parser.add_argument("--k-list", type=int, nargs="+", default=[10, 50, 100, 1000])
    parser.add_argument("--skip-bm25-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = QrelsMetricBlindness(k_list=args.k_list)
    inputs = analyzer.load_inputs(
        args.rollout_path,
        group_summary_path=args.group_summary_path,
        reward_source_path=args.reward_source_path if args.reward_source_path.exists() else None,
    )

    if args.skip_bm25_recompute:
        raise SystemExit("Phase 1.18e requires BM25 re-retrieval; do not use --skip-bm25-recompute.")

    search_tool = BM25SearchTool(rec_r1_root=REC_R1)
    analysis = analyzer.analyze_all(inputs, search_tool=search_tool)

    with (args.output_dir / "query_relevance_coverage.jsonl").open("w", encoding="utf-8") as fout:
        for row in analysis["query_coverage_rows"]:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (args.output_dir / "metric_by_k_diagnostics.jsonl").open("w", encoding="utf-8") as fout:
        for row in analysis["metric_by_k_rows"]:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (args.output_dir / "group_metric_spread_by_k.jsonl").open("w", encoding="utf-8") as fout:
        for row in analysis["group_spread_rows"]:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    (args.output_dir / "qrels_summary.json").write_text(
        json.dumps(analysis["qrels_summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        **analysis["summary"],
        "input_rollout_path": str(args.rollout_path),
        "input_group_summary_path": str(args.group_summary_path),
        "input_reward_source_path": str(args.reward_source_path)
        if args.reward_source_path.exists()
        else None,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_md = build_metric_blindness_report(
        analysis["query_coverage_rows"],
        analysis["group_spread_rows"],
        summary,
    )
    (args.output_dir / "metric_blindness_report.md").write_text(report_md, encoding="utf-8")

    readme = args.output_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Phase 1.18e Qrels / Metric Blindness\n\n"
            "Diagnostic analysis of ESCI qrels coverage and multi-K IR metric spread "
            "for Phase 1.18d strategy-controlled rollouts.\n",
            encoding="utf-8",
        )

    print("\n=== Phase 1.18e Qrels / Metric Blindness Summary ===")
    print(f"num_groups: {summary['num_groups']}")
    print(f"num_rollout_records: {summary['num_rollout_records']}")
    print(f"metric_has_quality_signal_group_rate: {summary['metric_has_quality_signal_group_rate']:.4f}")
    print(
        "small_k_blind_large_k_signal_group_rate: "
        f"{summary['small_k_blind_large_k_signal_group_rate']:.4f}"
    )
    print(f"qrels_sparse_all_k_blind_group_rate: {summary['qrels_sparse_all_k_blind_group_rate']:.4f}")
    print(f"bm25_retrieval_failure_group_rate: {summary['bm25_retrieval_failure_group_rate']:.4f}")
    print(f"strategy_query_too_similar_group_rate: {summary['strategy_query_too_similar_group_rate']:.4f}")
    print(f"mean_ndcg10_spread: {summary['mean_ndcg10_spread']:.4f}")
    print(f"mean_ndcg100_spread: {summary['mean_ndcg100_spread']:.4f}")
    print(f"mean_recall100_spread: {summary['mean_recall100_spread']:.4f}")
    print(f"qrels_sparse_query_rate: {summary['qrels_sparse_query_rate']:.4f}")
    print(f"recommended_next_phase: {summary['recommended_next_phase']}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
