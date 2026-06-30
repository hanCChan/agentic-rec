#!/usr/bin/env python3
"""
Phase 1.18a: Rollout Diversity / Reward Variance Diagnostics.

Analyzes Phase 1.17 real multi-sample rollout records without re-running rollouts.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  python scripts/analyze_multisample_rollout.py \
    --rollout-path experiments/phase117_multisample_rollout_5_g4/rollout_records.jsonl \
    --group-summary-path experiments/phase117_multisample_rollout_5_g4/group_summaries.jsonl \
    --output-dir experiments/phase118a_rollout_diagnostics_5_g4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.rollout_diagnostics import RolloutDiagnostics, build_case_studies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.18a rollout diversity diagnostics")
    parser.add_argument(
        "--rollout-path",
        type=Path,
        default=ROOT / "experiments/phase117_multisample_rollout_5_g4/rollout_records.jsonl",
    )
    parser.add_argument(
        "--group-summary-path",
        type=Path,
        default=ROOT / "experiments/phase117_multisample_rollout_5_g4/group_summaries.jsonl",
    )
    parser.add_argument(
        "--phase117-summary-path",
        type=Path,
        default=ROOT / "experiments/phase117_multisample_rollout_5_g4/summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/phase118a_rollout_diagnostics_5_g4",
    )
    return parser.parse_args()


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = RolloutDiagnostics()
    records = diagnostics.load_rollouts(args.rollout_path)
    analysis = diagnostics.analyze_all_groups(records)
    group_reports = analysis["group_reports"]
    classification = analysis["classification"]

    group_diag_path = args.output_dir / "group_diagnostics.jsonl"
    with group_diag_path.open("w", encoding="utf-8") as fout:
        for report in group_reports:
            fout.write(json.dumps(report, ensure_ascii=False) + "\n")

    diagnosis_summary = {
        "phase": "1.18a",
        "input_rollout_path": str(args.rollout_path),
        "input_group_summary_path": str(args.group_summary_path),
        "num_groups": analysis["num_groups"],
        "group_size": analysis["group_size"],
        "num_rollout_records": analysis["num_rollout_records"],
        **classification,
    }

    phase117_summary = load_json(args.phase117_summary_path)
    if phase117_summary:
        diagnosis_summary["phase117_reference"] = {
            "zero_std_group_rate": phase117_summary.get("zero_std_group_rate"),
            "mean_unique_trajectory_count": phase117_summary.get("mean_unique_trajectory_count"),
            "mean_unique_final_query_count": phase117_summary.get("mean_unique_final_query_count"),
        }

    case_studies = build_case_studies(group_reports)

    summary = {
        "phase": "1.18a",
        "num_groups": analysis["num_groups"],
        "group_size": analysis["group_size"],
        "num_rollout_records": analysis["num_rollout_records"],
        "zero_std_group_rate": classification["zero_std_group_rate"],
        "mean_group_reward_std": classification["mean_group_reward_std"],
        "mean_unique_trajectory_count": classification["mean_unique_trajectory_count"],
        "mean_unique_final_query_count": classification["mean_unique_final_query_count"],
        "category_counts": classification["category_counts"],
        "category_rates": classification["category_rates"],
        "avg_pairwise_final_query_jaccard": classification["avg_pairwise_final_query_jaccard"],
        "avg_pairwise_trajectory_jaccard": classification["avg_pairwise_trajectory_jaccard"],
        "main_diagnosis": classification["main_diagnosis"],
        "recommended_next_phase": classification["recommended_next_phase"],
        "is_training": False,
    }

    (args.output_dir / "diagnosis_summary.json").write_text(
        json.dumps(diagnosis_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "case_studies.md").write_text(case_studies, encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = args.output_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Phase 1.18a Rollout Diversity / Reward Variance Diagnostics\n\n"
            "Analysis of Phase 1.17 real multi-sample rollout records.\n"
            "No training or re-rollout was performed.\n",
            encoding="utf-8",
        )

    print("\n=== Phase 1.18a Rollout Diversity / Reward Variance Diagnostics Summary ===")
    print(f"num_groups: {summary['num_groups']}")
    print(f"group_size: {summary['group_size']}")
    print(f"num_rollout_records: {summary['num_rollout_records']}")
    print(f"zero_std_group_rate: {summary['zero_std_group_rate']:.4f}")
    print(f"mean_group_reward_std: {summary['mean_group_reward_std']:.6f}")
    print(f"mean_unique_trajectory_count: {summary['mean_unique_trajectory_count']:.4f}")
    print(f"mean_unique_final_query_count: {summary['mean_unique_final_query_count']:.4f}")
    print(
        "diverse_trajectory_reward_spread_rate: "
        f"{summary['category_rates']['diverse_trajectory_reward_spread']:.4f}"
    )
    print(
        "diverse_trajectory_zero_reward_rate: "
        f"{summary['category_rates']['diverse_trajectory_zero_reward']:.4f}"
    )
    print(
        "same_trajectory_zero_reward_rate: "
        f"{summary['category_rates']['same_trajectory_zero_reward']:.4f}"
    )
    print(f"avg_pairwise_final_query_jaccard: {summary['avg_pairwise_final_query_jaccard']:.4f}")
    print(f"avg_pairwise_trajectory_jaccard: {summary['avg_pairwise_trajectory_jaccard']:.4f}")
    print(f"main_diagnosis: {summary['main_diagnosis']}")
    print(f"recommended_next_phase: {summary['recommended_next_phase']}")
    print(f"output_dir: {args.output_dir}")
    print(f"\n[phase118a] wrote {group_diag_path}")
    print(f"[phase118a] wrote {args.output_dir / 'diagnosis_summary.json'}")
    print(f"[phase118a] wrote {args.output_dir / 'case_studies.md'}")


if __name__ == "__main__":
    main()
