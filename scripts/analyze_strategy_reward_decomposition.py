#!/usr/bin/env python3
"""
Phase 1.19a: Strategy Rollout Reward Decomposition.

Decomposes Phase 1.18d strategy-controlled rollout rewards to identify spread sources.
No training, no reward changes, no GRPO loss.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  python scripts/analyze_strategy_reward_decomposition.py \
    --rollout-path experiments/phase118d_strategy_rollout_5_g4/rollout_records.jsonl \
    --group-summary-path experiments/phase118d_strategy_rollout_5_g4/group_summaries.jsonl \
    --output-dir experiments/phase119a_strategy_reward_decomposition_5_g4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.strategy_reward_decomposition import StrategyRewardDecomposition, build_case_studies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.19a strategy reward decomposition")
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
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/phase119a_strategy_reward_decomposition_5_g4",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = StrategyRewardDecomposition()
    inputs = analyzer.load_inputs(args.rollout_path, args.group_summary_path)
    decomposed = analyzer.decompose_records(inputs["records"])
    analysis = analyzer.analyze_all(decomposed)

    with (args.output_dir / "strategy_reward_decomposition.jsonl").open("w", encoding="utf-8") as fout:
        for row in decomposed:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (args.output_dir / "group_reward_source_report.jsonl").open("w", encoding="utf-8") as fout:
        for report in analysis["group_reports"]:
            fout.write(json.dumps(report, ensure_ascii=False) + "\n")

    strategy_reward_summary = {
        "phase": "1.19a",
        "input_rollout_path": str(args.rollout_path),
        "input_group_summary_path": str(args.group_summary_path),
        "num_groups": analysis["num_groups"],
        "num_rollout_records": analysis["num_rollout_records"],
        "zero_std_group_rate_total_reward": analysis["zero_std_group_rate_total_reward"],
        "zero_std_group_rate_quality_only": analysis["zero_std_group_rate_quality_only"],
        "retrieval_quality_spread_group_rate": analysis["retrieval_quality_spread_group_rate"],
        "penalty_only_spread_group_rate": analysis["penalty_only_spread_group_rate"],
        "mixed_spread_group_rate": analysis["mixed_spread_group_rate"],
        "no_spread_group_rate": analysis["no_spread_group_rate"],
        "spread_source_counts": analysis["spread_source_counts"],
        "strategy_stats": analysis["strategy_stats"],
        "gate_evaluation": analysis["gate_evaluation"],
    }
    (args.output_dir / "strategy_reward_summary.json").write_text(
        json.dumps(strategy_reward_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    case_studies = build_case_studies(analysis)
    (args.output_dir / "case_studies.md").write_text(case_studies, encoding="utf-8")

    summary = {
        "phase": "1.19a",
        "num_groups": analysis["num_groups"],
        "num_rollout_records": analysis["num_rollout_records"],
        "zero_std_group_rate_total_reward": analysis["zero_std_group_rate_total_reward"],
        "zero_std_group_rate_quality_only": analysis["zero_std_group_rate_quality_only"],
        "retrieval_quality_spread_group_rate": analysis["retrieval_quality_spread_group_rate"],
        "penalty_only_spread_group_rate": analysis["penalty_only_spread_group_rate"],
        "mixed_spread_group_rate": analysis["mixed_spread_group_rate"],
        "no_spread_group_rate": analysis["no_spread_group_rate"],
        "strategy_stats": analysis["strategy_stats"],
        "gate_passed": analysis["gate_evaluation"]["gate_passed"],
        "recommendation": analysis["gate_evaluation"]["recommendation"],
        "is_training": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = args.output_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Phase 1.19a Strategy Rollout Reward Decomposition\n\n"
            "Reward spread source analysis for Phase 1.18d strategy-controlled rollouts.\n",
            encoding="utf-8",
        )

    print("\n=== Phase 1.19a Strategy Rollout Reward Decomposition Summary ===")
    print(f"num_groups: {summary['num_groups']}")
    print(f"num_rollout_records: {summary['num_rollout_records']}")
    print(f"zero_std_group_rate_total_reward: {summary['zero_std_group_rate_total_reward']:.4f}")
    print(f"zero_std_group_rate_quality_only: {summary['zero_std_group_rate_quality_only']:.4f}")
    print(f"retrieval_quality_spread_group_rate: {summary['retrieval_quality_spread_group_rate']:.4f}")
    print(f"penalty_only_spread_group_rate: {summary['penalty_only_spread_group_rate']:.4f}")
    print(f"mixed_spread_group_rate: {summary['mixed_spread_group_rate']:.4f}")
    print(f"no_spread_group_rate: {summary['no_spread_group_rate']:.4f}")
    print(f"gate_passed: {summary['gate_passed']}")
    print(f"recommendation: {summary['recommendation']}")
    print(f"output_dir: {args.output_dir}")

    for name, stats in analysis["strategy_stats"].items():
        print(
            f"  [{name}] mean_total={stats['strategy_mean_total_reward']:.4f} "
            f"mean_quality={stats['strategy_mean_quality_reward']:.4f} "
            f"mean_ndcg={stats['strategy_mean_ndcg_at_10']:.4f} "
            f"mean_penalty={stats['strategy_mean_penalty']:.4f} "
            f"mean_searches={stats['strategy_mean_search_calls']:.2f}"
        )


if __name__ == "__main__":
    main()
