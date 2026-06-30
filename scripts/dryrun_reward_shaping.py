#!/usr/bin/env python3
"""
Phase 1.18c: Reward Shaping Proposal + Dry-Run.

Offline evaluation of candidate reward formulas on Phase 1.17 multi-sample rollouts.
No training, no official reward changes.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  python scripts/dryrun_reward_shaping.py \
    --rollout-path experiments/phase117_multisample_rollout_5_g4/rollout_records.jsonl \
    --group-sensitivity-path experiments/phase118b_reward_sensitivity_5_g4/group_reward_sensitivity.jsonl \
    --query-metrics-path experiments/phase118b_reward_sensitivity_5_g4/query_metric_diagnostics.jsonl \
    --output-dir experiments/phase118c_reward_shaping_dryrun_5_g4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.reward_shaping_dryrun import (
    RewardShapingDryRun,
    build_candidate_comparison_md,
    build_reward_shaping_recommendations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.18c reward shaping dry-run")
    parser.add_argument(
        "--rollout-path",
        type=Path,
        default=ROOT / "experiments/phase117_multisample_rollout_5_g4/rollout_records.jsonl",
    )
    parser.add_argument(
        "--group-sensitivity-path",
        type=Path,
        default=ROOT / "experiments/phase118b_reward_sensitivity_5_g4/group_reward_sensitivity.jsonl",
    )
    parser.add_argument(
        "--query-metrics-path",
        type=Path,
        default=ROOT / "experiments/phase118b_reward_sensitivity_5_g4/query_metric_diagnostics.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/phase118c_reward_shaping_dryrun_5_g4",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dryrun = RewardShapingDryRun()
    result = dryrun.run(
        rollout_path=args.rollout_path,
        group_sensitivity_path=args.group_sensitivity_path,
        query_metrics_path=args.query_metrics_path,
    )

    comparison = result["comparison"]
    recommendation = result["recommendation"]
    summaries = {s["candidate_name"]: s for s in comparison["candidates"]}

    with (args.output_dir / "shaped_record_rewards.jsonl").open("w", encoding="utf-8") as fout:
        for row in result["shaped_records"]:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (args.output_dir / "candidate_group_reports.jsonl").open("w", encoding="utf-8") as fout:
        for row in result["candidate_group_reports"]:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    (args.output_dir / "candidate_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "candidate_comparison.md").write_text(
        build_candidate_comparison_md(comparison), encoding="utf-8"
    )
    (args.output_dir / "reward_shaping_recommendations.md").write_text(
        build_reward_shaping_recommendations(result), encoding="utf-8"
    )

    baseline_zero = comparison.get("baseline_zero_std_group_rate", 0.0)
    summary = {
        "phase": "1.18c",
        "num_groups": result["num_groups"],
        "num_rollout_records": result["num_rollout_records"],
        "baseline_zero_std_group_rate": baseline_zero,
        "best_candidate_name": comparison.get("best_candidate_by_reward_variance"),
        "best_candidate_zero_std_group_rate": comparison.get("best_candidate_zero_std_group_rate"),
        "reward_current_zero_std_group_rate": summaries["reward_current"]["zero_std_group_rate"],
        "reward_quality_only_zero_std_group_rate": summaries["reward_quality_only"]["zero_std_group_rate"],
        "reward_quality_best_step_zero_std_group_rate": summaries["reward_quality_best_step"]["zero_std_group_rate"],
        "reward_penalty_decoupled_zero_std_group_rate": summaries["reward_penalty_decoupled"]["zero_std_group_rate"],
        "reward_hit_depth_zero_std_group_rate": summaries["reward_hit_depth"]["zero_std_group_rate"],
        "reward_overlap_diagnostic_zero_std_group_rate": summaries["reward_overlap_diagnostic"]["zero_std_group_rate"],
        "safe_for_training_candidate": recommendation["safe_for_training_candidate"],
        "main_recommendation": recommendation["main_recommendation"],
        "next_phase": recommendation["next_phase"],
        "is_training": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = args.output_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Phase 1.18c Reward Shaping Proposal Dry-Run\n\n"
            "Offline comparison of candidate reward formulas on Phase 1.17 rollouts.\n",
            encoding="utf-8",
        )

    print("\n=== Phase 1.18c Reward Shaping Proposal Dry-Run Summary ===")
    print(f"num_groups: {summary['num_groups']}")
    print(f"num_rollout_records: {summary['num_rollout_records']}")
    print(f"baseline_zero_std_group_rate: {summary['baseline_zero_std_group_rate']:.4f}")
    print(f"reward_quality_only_zero_std_group_rate: {summary['reward_quality_only_zero_std_group_rate']:.4f}")
    print(
        "reward_quality_best_step_zero_std_group_rate: "
        f"{summary['reward_quality_best_step_zero_std_group_rate']:.4f}"
    )
    print(f"reward_hit_depth_zero_std_group_rate: {summary['reward_hit_depth_zero_std_group_rate']:.4f}")
    print(
        "reward_overlap_diagnostic_zero_std_group_rate: "
        f"{summary['reward_overlap_diagnostic_zero_std_group_rate']:.4f}"
    )
    print(f"best_candidate_name: {summary['best_candidate_name']}")
    print(f"best_candidate_zero_std_group_rate: {summary['best_candidate_zero_std_group_rate']:.4f}")
    print(f"safe_for_training_candidate: {summary['safe_for_training_candidate']}")
    print(f"main_recommendation: {summary['main_recommendation']}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
