#!/usr/bin/env python3
"""
Phase 1.20: No-update VERL Trainer Dry-Run.

Validates trainer-facing DataProto + VERL GRPO advantage + mini-batch split
+ no-update loss path. Does NOT train, call trainer.fit(), or optimizer.step().

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  python scripts/smoke_no_update_trainer_dryrun.py \
    --rollout-path experiments/phase119b_scale_gate_check/strategy_rollout_20_g4/rollout_records.jsonl \
    --shaped-reward-path experiments/phase119b_scale_gate_check/large_k_reward_20_g4/large_k_shaped_record_rewards.jsonl \
    --candidate-name reward_largek_mix_1000 \
    --tokenizer-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
    --output-dir experiments/phase120_no_update_trainer_dryrun_20_g4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REC_R1 = ROOT / "Rec-R1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(REC_R1) not in sys.path:
    sys.path.insert(0, str(REC_R1))

from src.agents.no_update_trainer_dryrun import (  # noqa: E402
    DEFAULT_CANDIDATE,
    NoUpdateTrainerDryRun,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.20 no-update VERL trainer dry-run")
    parser.add_argument(
        "--rollout-path",
        type=Path,
        default=ROOT
        / "experiments/phase119b_scale_gate_check/strategy_rollout_20_g4/rollout_records.jsonl",
    )
    parser.add_argument(
        "--shaped-reward-path",
        type=Path,
        default=ROOT
        / "experiments/phase119b_scale_gate_check/large_k_reward_20_g4/large_k_shaped_record_rewards.jsonl",
    )
    parser.add_argument("--candidate-name", type=str, default=DEFAULT_CANDIDATE)
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/data1/hcc/.hf_home/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/phase120_no_update_trainer_dryrun_20_g4",
    )
    parser.add_argument("--train-batch-size", type=int, default=20)
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=20)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument(
        "--loss-agg-mode",
        type=str,
        default="token-mean",
        choices=["token-mean", "seq-mean-token-sum", "seq-mean-token-mean"],
    )
    parser.add_argument("--synthetic-logprob-delta", type=float, default=0.02)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--max-total-length", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dryrun = NoUpdateTrainerDryRun(
        tokenizer_path=args.tokenizer_path,
        candidate_name=args.candidate_name,
        train_batch_size=args.train_batch_size,
        rollout_n=args.rollout_n,
        ppo_mini_batch_size=args.ppo_mini_batch_size,
        micro_batch_size=args.micro_batch_size,
        cliprange=args.cliprange,
        kl_coef=args.kl_coef,
        loss_agg_mode=args.loss_agg_mode,
        synthetic_logprob_delta=args.synthetic_logprob_delta,
        seed=args.seed,
    )

    result = dryrun.run(
        rollout_path=str(args.rollout_path),
        shaped_reward_path=str(args.shaped_reward_path),
        output_dir=str(args.output_dir),
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        max_total_length=args.max_total_length,
    )

    summary = {
        **result["summary"],
        "input_rollout_path": str(args.rollout_path),
        "input_shaped_reward_path": str(args.shaped_reward_path),
    }
    (args.output_dir / "summary.json").write_text(
        __import__("json").dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not summary["trainer_required_keys_passed"]:
        raise SystemExit(f"trainer required keys failed: {result['key_check']}")
    if not summary["advantage_check_passed"]:
        raise SystemExit(f"advantage check failed: {result['adv_check']}")
    if not summary["minibatch_check_passed"]:
        raise SystemExit(f"minibatch check failed: {result['minibatch_check']}")
    if not summary["loss_check_passed"]:
        raise SystemExit(f"loss check failed: {result['loss_check']}")
    if not summary["no_update_guard_passed"]:
        raise SystemExit(f"no-update guard failed: {result['guard_result']}")

    print("\n=== Phase 1.20 No-update VERL Trainer Dry-Run Summary ===")
    print(f"num_groups: {summary['num_groups']}")
    print(f"group_size: {summary['group_size']}")
    print(f"num_rollout_records: {summary['num_rollout_records']}")
    print(f"reward_candidate: {summary['reward_candidate']}")
    print(f"used_real_dataproto: {summary['used_real_dataproto']}")
    print(f"used_verl_compute_advantage: {summary['used_verl_compute_advantage']}")
    print(f"fallback_to_project_advantage: {summary['fallback_to_project_advantage']}")
    print(f"trainer_required_keys_passed: {summary['trainer_required_keys_passed']}")
    print(f"advantage_check_passed: {summary['advantage_check_passed']}")
    print(f"minibatch_check_passed: {summary['minibatch_check_passed']}")
    print(f"loss_check_passed: {summary['loss_check_passed']}")
    print(f"policy_loss_finite: {summary['policy_loss_finite']}")
    print(f"no_update_guard_passed: {summary['no_update_guard_passed']}")
    print(f"trainer_fit_called: {summary['trainer_fit_called']}")
    print(f"update_actor_called: {summary['update_actor_called']}")
    print(f"optimizer_step_called: {summary['optimizer_step_called']}")
    print(f"safe_for_phase2: {summary['safe_for_phase2']}")
    print(f"output_dir: {args.output_dir}")
    print(f"\n[phase120] {summary['dryrun_warning']}")


if __name__ == "__main__":
    main()
