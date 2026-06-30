#!/usr/bin/env python3
"""
Phase 1.16: GRPO Loss Dry-Run.

Computes PPO/GRPO-style clipped policy loss from mock log_probs and advantages.
No GRPO training, no actor.forward, no optimizer.step.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  python scripts/smoke_grpo_loss_dryrun.py \
    --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
    --output-dir experiments/phase116_grpo_loss_dryrun_10_g4 \
    --num-base-records 10 --group-size 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.dataproto_mock import DataProtoMock
from src.agents.grpo_advantage_mock import GRPOAdvantageMock
from src.agents.grpo_loss_dryrun import GRPOLossDryRun, LOSS_DRYRUN_WARNING
from src.agents.real_dataproto_adapter import RealDataProtoAdapter
from src.agents.verl_batch_builder import VerlBatchBuilder, check_batch_shapes
from src.agents.verl_training_field_builder import VerlTrainingFieldBuilder, check_training_fields


def load_rollout_records(path: Path, num_records: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                if len(records) >= num_records:
                    break
    return records


def attach_group_metadata(fields: Dict[str, Any], grouped_records: List[Dict[str, Any]], group_size: int) -> None:
    fields["group_ids"] = [r["group_id"] for r in grouped_records]
    fields["group_indices"] = [r["group_index"] for r in grouped_records]
    fields["group_size"] = [group_size] * len(grouped_records)


def build_dataproto_pipeline(
    grouped_records: List[Dict[str, Any]],
    tokenizer_path: str,
    group_size: int,
    max_prompt_length: int,
    max_response_length: int,
    max_total_length: int,
) -> tuple[Any, bool]:
    batch_builder = VerlBatchBuilder(
        tokenizer_path=tokenizer_path,
        max_prompt_length=max_prompt_length,
        max_response_length=max_response_length,
        max_total_length=max_total_length,
    )
    batch = batch_builder.build_batch(grouped_records)
    check_batch_shapes(batch)

    pad_token_id = batch_builder.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = batch_builder.tokenizer.eos_token_id

    field_builder = VerlTrainingFieldBuilder(pad_token_id=pad_token_id)
    fields = field_builder.build_training_fields(batch)
    check_training_fields(fields)
    attach_group_metadata(fields, grouped_records, group_size)

    mock_proto = DataProtoMock.from_fields(fields)
    mock_validate = mock_proto.validate()
    if not mock_validate["passed"]:
        raise SystemExit(f"DataProtoMock validate failed: {mock_validate['errors']}")

    adapter = RealDataProtoAdapter()
    convert_result = adapter.to_real_dataproto(mock_proto)
    data_proto = convert_result["real_proto"] if convert_result["used_real_dataproto"] else mock_proto
    return data_proto, convert_result["used_real_dataproto"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.16 GRPO loss dry-run")
    parser.add_argument("--rollout-path", type=str, required=True)
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/data1/hcc/.hf_home/Qwen2.5-3B-Instruct",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-base-records", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--synthetic-reward-jitter", type=float, default=0.02)
    parser.add_argument("--no-synthetic-jitter", action="store_true")
    parser.add_argument("--normalize-by-std", action="store_true", default=True)
    parser.add_argument("--no-normalize-by-std", action="store_true")
    parser.add_argument("--synthetic-logprob-delta", type=float, default=0.02)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument(
        "--loss-agg-mode",
        type=str,
        default="token-mean",
        choices=["token-mean", "seq-mean-token-sum", "seq-mean-token-mean"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--max-total-length", type=int, default=3072)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jitter = 0.0 if args.no_synthetic_jitter else args.synthetic_reward_jitter
    normalize_by_std = not args.no_normalize_by_std

    base_records = load_rollout_records(Path(args.rollout_path), args.num_base_records)
    if not base_records:
        raise SystemExit(f"no records in {args.rollout_path}")

    grpo_mock = GRPOAdvantageMock(
        group_size=args.group_size,
        normalize_by_std=normalize_by_std,
        synthetic_reward_jitter=jitter,
        seed=args.seed,
    )
    grouped_records = grpo_mock.build_grouped_records(base_records)

    data_proto, used_real_dataproto = build_dataproto_pipeline(
        grouped_records,
        args.tokenizer_path,
        args.group_size,
        args.max_prompt_length,
        args.max_response_length,
        args.max_total_length,
    )

    adv_output = grpo_mock.compute_group_advantages(data_proto)
    adv_check = grpo_mock.check_group_advantages(data_proto, adv_output)
    if not adv_check["advantage_check_passed"]:
        raise SystemExit(f"advantage check failed: {adv_check}")

    loss_dryrun = GRPOLossDryRun(
        cliprange=args.cliprange,
        kl_coef=args.kl_coef,
        loss_agg_mode=args.loss_agg_mode,
        synthetic_logprob_delta=args.synthetic_logprob_delta,
        seed=args.seed,
    )
    loss_inputs = loss_dryrun.build_mock_logprob_inputs(data_proto, adv_output)
    loss_output = loss_dryrun.compute_policy_loss(data_proto, loss_inputs)
    loss_check = loss_dryrun.check_loss_output(data_proto, loss_inputs, loss_output)

    if not loss_check["loss_check_passed"]:
        raise SystemExit(f"loss check failed: {loss_check}")

    summary = {
        "phase": "1.16",
        "num_base_records": len(base_records),
        "group_size": args.group_size,
        "num_grouped_records": len(grouped_records),
        "used_real_dataproto": used_real_dataproto,
        "advantage_check_passed": adv_check["advantage_check_passed"],
        "loss_check_passed": loss_check["loss_check_passed"],
        "loss_agg_mode": args.loss_agg_mode,
        "cliprange": args.cliprange,
        "kl_coef": args.kl_coef,
        "synthetic_logprob_delta": args.synthetic_logprob_delta,
        "policy_loss_value": loss_output["policy_loss_value"],
        "policy_loss_finite": loss_check["policy_loss_finite"],
        "policy_loss_mat_shape": loss_check["policy_loss_mat_shape"],
        "ratio_shape": loss_check["ratio_shape"],
        "token_kl_shape": loss_check["token_kl_shape"],
        "kl_penalty_shape": loss_check["kl_penalty_shape"],
        "clipfrac": loss_output["clipfrac"],
        "mean_valid_ratio": loss_output["mean_valid_ratio"],
        "min_valid_ratio": loss_output["min_valid_ratio"],
        "max_valid_ratio": loss_output["max_valid_ratio"],
        "mean_valid_kl": loss_output["mean_valid_kl"],
        "mean_valid_kl_penalty": loss_output["mean_valid_kl_penalty"],
        "zero_std_group_rate": adv_check["zero_std_group_rate"],
        "mean_abs_sequence_advantage": adv_check["mean_abs_sequence_advantage"],
        "padding_loss_zero": loss_check["padding_loss_zero"],
        "padding_ratio_zero": loss_check["padding_ratio_zero"],
        "padding_kl_zero": loss_check["padding_kl_zero"],
        "is_dryrun": True,
        "dryrun_warning": LOSS_DRYRUN_WARNING,
        "config": {
            "rollout_path": args.rollout_path,
            "no_synthetic_jitter": args.no_synthetic_jitter,
        },
    }

    grpo_loss_shapes = {
        "policy_loss_mat": loss_check["policy_loss_mat_shape"],
        "ratio": loss_check["ratio_shape"],
        "token_kl": loss_check["token_kl_shape"],
        "kl_penalty": loss_check["kl_penalty_shape"],
    }
    grpo_loss_stats = {
        "policy_loss_value": loss_output["policy_loss_value"],
        "clipfrac": loss_output["clipfrac"],
        "mean_valid_ratio": loss_output["mean_valid_ratio"],
        "min_valid_ratio": loss_output["min_valid_ratio"],
        "max_valid_ratio": loss_output["max_valid_ratio"],
        "mean_valid_kl": loss_output["mean_valid_kl"],
        "mean_valid_kl_penalty": loss_output["mean_valid_kl_penalty"],
        "zero_std_group_rate": adv_check["zero_std_group_rate"],
        "mean_abs_sequence_advantage": adv_check["mean_abs_sequence_advantage"],
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (output_dir / "grpo_loss_shapes.json").open("w", encoding="utf-8") as f:
        json.dump(grpo_loss_shapes, f, indent=2, ensure_ascii=False)
    with (output_dir / "grpo_loss_stats.json").open("w", encoding="utf-8") as f:
        json.dump(grpo_loss_stats, f, indent=2, ensure_ascii=False)

    print("\n=== Phase 1.16 GRPO Loss Dry-Run Summary ===")
    print(f"num_base_records: {summary['num_base_records']}")
    print(f"group_size: {summary['group_size']}")
    print(f"num_grouped_records: {summary['num_grouped_records']}")
    print(f"used_real_dataproto: {summary['used_real_dataproto']}")
    print(f"advantage_check_passed: {summary['advantage_check_passed']}")
    print(f"loss_check_passed: {summary['loss_check_passed']}")
    print(f"loss_agg_mode: {summary['loss_agg_mode']}")
    print(f"policy_loss_value: {summary['policy_loss_value']:.6f}")
    print(f"clipfrac: {summary['clipfrac']:.4f}")
    print(f"mean_valid_ratio: {summary['mean_valid_ratio']:.4f}")
    print(f"mean_valid_kl: {summary['mean_valid_kl']:.6f}")
    print(f"mean_valid_kl_penalty: {summary['mean_valid_kl_penalty']:.6f}")
    print(f"zero_std_group_rate: {summary['zero_std_group_rate']:.4f}")
    print(f"padding_loss_zero: {summary['padding_loss_zero']}")
    print(f"output_dir: {output_dir}")
    print(f"\n[phase116] {LOSS_DRYRUN_WARNING}")
    print(f"[phase116] wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
