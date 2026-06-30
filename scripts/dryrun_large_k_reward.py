#!/usr/bin/env python3
"""
Phase 1.18f: Large-K Reward Candidate Dry-Run.

Evaluates global large-K retrieval-quality reward candidates offline.
No training, no official reward changes, no GRPO loss.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  python scripts/dryrun_large_k_reward.py \
    --rollout-path experiments/phase118d_strategy_rollout_5_g4/rollout_records.jsonl \
    --metric-by-k-path experiments/phase118e_qrels_metric_blindness_5_g4/metric_by_k_diagnostics.jsonl \
    --group-metric-spread-path experiments/phase118e_qrels_metric_blindness_5_g4/group_metric_spread_by_k.jsonl \
    --decomposition-path experiments/phase119a_strategy_reward_decomposition_5_g4/group_reward_source_report.jsonl \
    --output-dir experiments/phase118f_large_k_reward_dryrun_5_g4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.large_k_reward_dryrun import (
    LargeKRewardDryRun,
    build_large_k_candidate_comparison_md,
    build_large_k_reward_recommendations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.18f large-K reward candidate dry-run")
    parser.add_argument(
        "--rollout-path",
        type=Path,
        default=ROOT / "experiments/phase118d_strategy_rollout_5_g4/rollout_records.jsonl",
    )
    parser.add_argument(
        "--metric-by-k-path",
        type=Path,
        default=ROOT / "experiments/phase118e_qrels_metric_blindness_5_g4/metric_by_k_diagnostics.jsonl",
    )
    parser.add_argument(
        "--group-metric-spread-path",
        type=Path,
        default=ROOT / "experiments/phase118e_qrels_metric_blindness_5_g4/group_metric_spread_by_k.jsonl",
    )
    parser.add_argument(
        "--decomposition-path",
        type=Path,
        default=ROOT
        / "experiments/phase119a_strategy_reward_decomposition_5_g4/group_reward_source_report.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/phase118f_large_k_reward_dryrun_5_g4",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    decomp_path = args.decomposition_path if args.decomposition_path.exists() else None
    dryrun = LargeKRewardDryRun()
    result = dryrun.run(
        rollout_path=args.rollout_path,
        metric_by_k_path=args.metric_by_k_path,
        group_metric_spread_path=args.group_metric_spread_path,
        decomposition_path=decomp_path,
    )

    with (args.output_dir / "large_k_shaped_record_rewards.jsonl").open("w", encoding="utf-8") as fout:
        for row in result["shaped_records"]:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (args.output_dir / "large_k_candidate_group_reports.jsonl").open("w", encoding="utf-8") as fout:
        for row in result["candidate_group_reports"]:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    comparison_payload = {
        "num_groups": result["num_groups"],
        "num_rollout_records": result["num_rollout_records"],
        "candidates": result["comparison"]["candidates"],
        "gate": result["gate"],
        "best_non_diagnostic_candidate": result["comparison"]["best_non_diagnostic_candidate"],
        "best_non_diagnostic_zero_std_group_rate": result[
            "comparison"
        ]["best_non_diagnostic_zero_std_group_rate"],
    }
    (args.output_dir / "large_k_candidate_comparison.json").write_text(
        json.dumps(comparison_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    comparison_md = build_large_k_candidate_comparison_md(result["comparison"], result["gate"])
    (args.output_dir / "large_k_candidate_comparison.md").write_text(comparison_md, encoding="utf-8")

    recommendations_md = build_large_k_reward_recommendations(result)
    (args.output_dir / "large_k_reward_recommendations.md").write_text(
        recommendations_md, encoding="utf-8"
    )

    summaries = {s["candidate_name"]: s for s in result["comparison"]["candidates"]}
    gate = result["gate"]

    if gate["gate_passed"]:
        main_recommendation = (
            f"Proceed to Phase 1.19 only with `{gate['recommended_candidate']}` as "
            "quality-only GRPO advantage; keep penalties outside advantage."
        )
    else:
        main_recommendation = (
            "Do not proceed to Phase 1.19. Expand smoke set or fix metric/qrels coverage first."
        )

    summary = {
        "phase": "1.18f",
        "num_groups": result["num_groups"],
        "num_rollout_records": result["num_rollout_records"],
        "baseline_zero_std_group_rate": summaries.get("reward_ndcg10", {}).get(
            "zero_std_group_rate", 0.8
        ),
        "reward_ndcg10_zero_std_group_rate": summaries.get("reward_ndcg10", {}).get(
            "zero_std_group_rate"
        ),
        "reward_ndcg100_zero_std_group_rate": summaries.get("reward_ndcg100", {}).get(
            "zero_std_group_rate"
        ),
        "reward_ndcg1000_zero_std_group_rate": summaries.get("reward_ndcg1000", {}).get(
            "zero_std_group_rate"
        ),
        "reward_largek_mix_100_zero_std_group_rate": summaries.get("reward_largek_mix_100", {}).get(
            "zero_std_group_rate"
        ),
        "reward_largek_mix_1000_zero_std_group_rate": summaries.get(
            "reward_largek_mix_1000", {}
        ).get("zero_std_group_rate"),
        "best_non_diagnostic_candidate": result["comparison"]["best_non_diagnostic_candidate"],
        "best_non_diagnostic_zero_std_group_rate": result[
            "comparison"
        ]["best_non_diagnostic_zero_std_group_rate"],
        "gate_passed": gate["gate_passed"],
        "safe_for_phase_119": gate["safe_for_phase_119"],
        "recommended_candidate": gate["recommended_candidate"],
        "main_recommendation": main_recommendation,
        "is_training": False,
        "input_rollout_path": str(args.rollout_path),
        "input_metric_by_k_path": str(args.metric_by_k_path),
    }

    for name, s in summaries.items():
        summary[f"{name}_retrieval_quality_spread_group_rate"] = s[
            "retrieval_quality_spread_group_rate"
        ]
        summary[f"{name}_penalty_only_spread_group_rate"] = s["penalty_only_spread_group_rate"]

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = args.output_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Phase 1.18f Large-K Reward Candidate Dry-Run\n\n"
            "Offline evaluation of large-K retrieval-quality reward candidates for "
            "Phase 1.18d strategy-controlled rollouts.\n",
            encoding="utf-8",
        )

    best = summaries.get(result["comparison"]["best_non_diagnostic_candidate"] or "", {})
    print("\n=== Phase 1.18f Large-K Reward Candidate Dry-Run Summary ===")
    print(f"num_groups: {result['num_groups']}")
    print(f"num_rollout_records: {result['num_rollout_records']}")
    print(
        f"reward_ndcg10_zero_std_group_rate: "
        f"{summaries.get('reward_ndcg10', {}).get('zero_std_group_rate', 0):.4f}"
    )
    print(
        f"reward_ndcg100_zero_std_group_rate: "
        f"{summaries.get('reward_ndcg100', {}).get('zero_std_group_rate', 0):.4f}"
    )
    print(
        f"reward_ndcg1000_zero_std_group_rate: "
        f"{summaries.get('reward_ndcg1000', {}).get('zero_std_group_rate', 0):.4f}"
    )
    print(
        f"reward_largek_mix_100_zero_std_group_rate: "
        f"{summaries.get('reward_largek_mix_100', {}).get('zero_std_group_rate', 0):.4f}"
    )
    print(
        f"reward_largek_mix_1000_zero_std_group_rate: "
        f"{summaries.get('reward_largek_mix_1000', {}).get('zero_std_group_rate', 0):.4f}"
    )
    print(f"best_non_diagnostic_candidate: {result['comparison']['best_non_diagnostic_candidate']}")
    print(
        f"best_non_diagnostic_zero_std_group_rate: "
        f"{result['comparison']['best_non_diagnostic_zero_std_group_rate']}"
    )
    if best:
        print(
            f"retrieval_quality_spread_group_rate: "
            f"{best.get('retrieval_quality_spread_group_rate', 0):.4f}"
        )
        print(
            f"penalty_only_spread_group_rate: "
            f"{best.get('penalty_only_spread_group_rate', 0):.4f}"
        )
    print(f"gate_passed: {gate['gate_passed']}")
    print(f"safe_for_phase_119: {gate['safe_for_phase_119']}")
    print(f"recommended_candidate: {gate['recommended_candidate']}")
    print(f"main_recommendation: {main_recommendation}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
