#!/usr/bin/env python3
"""
Phase 1.19: Real GRPO Loss Dry-Run with Large-K Quality Reward.

Uses Phase 1.18d strategy-controlled groups and Phase 1.18f reward_largek_mix_1000
for quality-only GRPO advantages + Phase 1.16 clipped loss dry-run.
No training, no optimizer.step.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  python scripts/smoke_real_grpo_loss_dryrun.py \
    --rollout-path experiments/phase118d_strategy_rollout_5_g4/rollout_records.jsonl \
    --shaped-reward-path experiments/phase118f_large_k_reward_dryrun_5_g4/large_k_shaped_record_rewards.jsonl \
    --candidate-name reward_largek_mix_1000 \
    --tokenizer-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
    --output-dir experiments/phase119_real_grpo_loss_dryrun_5_g4_largek1000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.real_grpo_loss_dryrun import (
    DEFAULT_CANDIDATE,
    RealGRPOLossDryRun,
    build_real_grpo_dryrun_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.19 real GRPO loss dry-run")
    parser.add_argument(
        "--rollout-path",
        type=Path,
        default=ROOT / "experiments/phase118d_strategy_rollout_5_g4/rollout_records.jsonl",
    )
    parser.add_argument(
        "--shaped-reward-path",
        type=Path,
        default=ROOT
        / "experiments/phase118f_large_k_reward_dryrun_5_g4/large_k_shaped_record_rewards.jsonl",
    )
    parser.add_argument(
        "--phase118f-summary-path",
        type=Path,
        default=ROOT / "experiments/phase118f_large_k_reward_dryrun_5_g4/summary.json",
    )
    parser.add_argument(
        "--candidate-name",
        type=str,
        default=DEFAULT_CANDIDATE,
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/data1/hcc/.hf_home/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/phase119_real_grpo_loss_dryrun_5_g4_largek1000",
    )
    parser.add_argument("--synthetic-logprob-delta", type=float, default=0.02)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument(
        "--loss-agg-mode",
        type=str,
        default="token-mean",
        choices=["token-mean", "seq-mean-token-sum", "seq-mean-token-mean"],
    )
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--max-total-length", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    phase118f_summary = args.phase118f_summary_path if args.phase118f_summary_path.exists() else None

    dryrun = RealGRPOLossDryRun(
        candidate_name=args.candidate_name,
        cliprange=args.cliprange,
        kl_coef=args.kl_coef,
        loss_agg_mode=args.loss_agg_mode,
        synthetic_logprob_delta=args.synthetic_logprob_delta,
        seed=args.seed,
    )

    result = dryrun.run(
        rollout_path=args.rollout_path,
        shaped_reward_path=args.shaped_reward_path,
        tokenizer_path=args.tokenizer_path,
        phase118f_summary_path=phase118f_summary,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        max_total_length=args.max_total_length,
    )

    summary = {
        **result["summary"],
        "input_rollout_path": str(args.rollout_path),
        "input_shaped_reward_path": str(args.shaped_reward_path),
        "input_phase118f_summary_path": str(phase118f_summary) if phase118f_summary else None,
    }

    if not summary["advantage_check_passed"]:
        raise SystemExit(f"advantage check failed: {result['adv_check']}")
    if not summary["loss_check_passed"]:
        raise SystemExit(f"loss check failed: {result['loss_check']}")

    with (args.output_dir / "quality_reward_alignment.jsonl").open("w", encoding="utf-8") as fout:
        for row in result["alignment_log"]:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (args.output_dir / "quality_reward_group_reports.jsonl").open("w", encoding="utf-8") as fout:
        for row in result["spread_stats"]["group_reports"]:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "grpo_loss_shapes.json").write_text(
        json.dumps(result["loss_shapes"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "grpo_loss_stats.json").write_text(
        json.dumps(result["loss_stats"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "real_grpo_loss_dryrun_report.md").write_text(
        build_real_grpo_dryrun_report(result),
        encoding="utf-8",
    )

    readme = args.output_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Phase 1.19 Real GRPO Loss Dry-Run (Large-K Quality Reward)\n\n"
            "Strategy-controlled real groups + reward_largek_mix_1000 quality-only "
            "advantages + GRPO clipped loss dry-run. No training.\n",
            encoding="utf-8",
        )

    print("\n=== Phase 1.19 Real GRPO Loss Dry-Run Summary ===")
    print(f"num_groups: {summary['num_groups']}")
    print(f"group_size: {summary['group_size']}")
    print(f"num_rollout_records: {summary['num_rollout_records']}")
    print(f"reward_candidate: {summary['reward_candidate']}")
    print(f"quality_only_advantage: {summary['quality_only_advantage']}")
    print(f"penalties_in_advantage: {summary['penalties_in_advantage']}")
    print(f"zero_std_group_rate: {summary['zero_std_group_rate']:.4f}")
    print(
        f"retrieval_quality_spread_group_rate: "
        f"{summary['retrieval_quality_spread_group_rate']:.4f}"
    )
    print(f"penalty_only_spread_group_rate: {summary['penalty_only_spread_group_rate']:.4f}")
    print(f"advantage_check_passed: {summary['advantage_check_passed']}")
    print(f"loss_check_passed: {summary['loss_check_passed']}")
    print(f"used_real_dataproto: {summary['used_real_dataproto']}")
    print(f"policy_loss_finite: {summary['policy_loss_finite']}")
    print(f"policy_loss_value: {summary['policy_loss_value']:.6f}")
    print(f"clipfrac: {summary['clipfrac']:.4f}")
    print(f"mean_valid_ratio: {summary['mean_valid_ratio']:.4f}")
    print(f"mean_valid_kl: {summary['mean_valid_kl']:.6f}")
    print(f"padding_loss_zero: {summary['padding_loss_zero']}")
    print(f"padding_ratio_zero: {summary['padding_ratio_zero']}")
    print(f"padding_kl_zero: {summary['padding_kl_zero']}")
    print(f"output_dir: {args.output_dir}")
    print(f"\n[phase119] {summary['dryrun_warning']}")


if __name__ == "__main__":
    main()
