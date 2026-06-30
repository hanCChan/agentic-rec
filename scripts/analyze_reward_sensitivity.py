#!/usr/bin/env python3
"""
Phase 1.18b: Reward Sensitivity Diagnostics.

Analyzes why multi-sample GRPO groups lack reward spread without re-running rollouts.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  python scripts/analyze_reward_sensitivity.py \
    --rollout-path experiments/phase117_multisample_rollout_5_g4/rollout_records.jsonl \
    --group-diagnostics-path experiments/phase118a_rollout_diagnostics_5_g4/group_diagnostics.jsonl \
    --output-dir experiments/phase118b_reward_sensitivity_5_g4 \
    --topk-list 10 50 100
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

from src.agents.reward_sensitivity_diagnostics import (
    RewardSensitivityDiagnostics,
    build_reward_recommendations,
    load_group_diagnostics,
)
from src.tools.bm25_tool import BM25SearchTool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.18b reward sensitivity diagnostics")
    parser.add_argument(
        "--rollout-path",
        type=Path,
        default=ROOT / "experiments/phase117_multisample_rollout_5_g4/rollout_records.jsonl",
    )
    parser.add_argument(
        "--group-diagnostics-path",
        type=Path,
        default=ROOT / "experiments/phase118a_rollout_diagnostics_5_g4/group_diagnostics.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/phase118b_reward_sensitivity_5_g4",
    )
    parser.add_argument("--topk-list", type=int, nargs="+", default=[10, 50, 100])
    parser.add_argument("--skip-bm25-recompute", action="store_true")
    parser.add_argument("--max-groups", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = RewardSensitivityDiagnostics(topk_list=args.topk_list)
    records = diagnostics.load_rollouts(args.rollout_path)
    phase118a_reports = load_group_diagnostics(args.group_diagnostics_path)

    search_tool = None
    if not args.skip_bm25_recompute:
        search_tool = BM25SearchTool(rec_r1_root=REC_R1)

    analysis = diagnostics.analyze_all(
        records,
        search_tool=search_tool,
        skip_bm25_recompute=args.skip_bm25_recompute,
        phase118a_reports=phase118a_reports,
        max_groups=args.max_groups,
    )

    summary = analysis["summary"]
    group_reports = analysis["group_reports"]
    query_metrics = analysis["query_metrics"]

    with (args.output_dir / "group_reward_sensitivity.jsonl").open("w", encoding="utf-8") as fout:
        for report in group_reports:
            fout.write(json.dumps(report, ensure_ascii=False) + "\n")

    with (args.output_dir / "query_metric_diagnostics.jsonl").open("w", encoding="utf-8") as fout:
        for row in query_metrics:
            slim = {k: v for k, v in row.items() if k not in ("target_items", "rel_scores")}
            fout.write(json.dumps(slim, ensure_ascii=False) + "\n")

    reward_sensitivity_summary = {"phase": "1.18b", **summary}
    (args.output_dir / "reward_sensitivity_summary.json").write_text(
        json.dumps(reward_sensitivity_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    recommendations = build_reward_recommendations(analysis)
    (args.output_dir / "reward_recommendations.md").write_text(recommendations, encoding="utf-8")

    compact_summary = {
        "phase": "1.18b",
        "num_groups": summary["num_groups"],
        "group_size": summary["group_size"],
        "num_rollout_records": summary["num_rollout_records"],
        "current_zero_std_group_rate": summary["current_zero_std_group_rate"],
        "penalty_only_spread_rate": summary["penalty_only_spread_rate"],
        "current_reward_sensitive_rate": summary["current_reward_sensitive_rate"],
        "ndcg10_blind_but_recall_sensitive_rate": summary["ndcg10_blind_but_recall_sensitive_rate"],
        "category_counts": summary["category_counts"],
        "category_rates": summary["category_rates"],
        "mean_ndcg10_spread": summary["mean_ndcg10_spread"],
        "mean_recall50_spread": summary["mean_recall50_spread"],
        "mean_mrr50_spread": summary["mean_mrr50_spread"],
        "mean_top10_overlap": summary.get("mean_top10_overlap"),
        "mean_top50_overlap": summary.get("mean_top50_overlap"),
        "mean_top100_overlap": summary.get("mean_top100_overlap"),
        "main_conclusion": summary["main_conclusion"],
        "recommended_next_phase": summary["recommended_next_phase"],
        "is_training": False,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(compact_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = args.output_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Phase 1.18b Reward Sensitivity Diagnostics\n\n"
            "Diagnostic analysis of NDCG/Recall/MRR sensitivity and reward decomposition "
            "for Phase 1.17 multi-sample rollouts.\n",
            encoding="utf-8",
        )

    print("\n=== Phase 1.18b Reward Sensitivity Diagnostics Summary ===")
    print(f"num_groups: {summary['num_groups']}")
    print(f"current_zero_std_group_rate: {summary['current_zero_std_group_rate']:.4f}")
    print(f"penalty_only_spread_rate: {summary['penalty_only_spread_rate']:.4f}")
    print(f"current_reward_sensitive_rate: {summary['current_reward_sensitive_rate']:.4f}")
    print(
        "ndcg10_blind_but_recall_sensitive_rate: "
        f"{summary['ndcg10_blind_but_recall_sensitive_rate']:.4f}"
    )
    print(
        "retrieval_results_change_but_metric_blind_rate: "
        f"{summary['retrieval_results_change_but_metric_blind_rate']:.4f}"
    )
    print(f"query_too_similar_rate: {summary['query_too_similar_rate']:.4f}")
    print(f"label_sparse_or_all_zero_rate: {summary['label_sparse_or_all_zero_rate']:.4f}")
    print(f"mean_ndcg10_spread: {summary['mean_ndcg10_spread']:.4f}")
    print(f"mean_recall50_spread: {summary['mean_recall50_spread']:.4f}")
    print(f"mean_mrr50_spread: {summary['mean_mrr50_spread']:.4f}")
    top50 = summary.get("mean_top50_overlap")
    print(f"mean_top50_overlap: {top50:.4f}" if top50 is not None else "mean_top50_overlap: n/a")
    print(f"main_conclusion: {summary['main_conclusion']}")
    print(f"output_dir: {args.output_dir}")


if __name__ == "__main__":
    main()
