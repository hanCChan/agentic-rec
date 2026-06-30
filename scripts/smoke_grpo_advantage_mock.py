#!/usr/bin/env python3
"""
Phase 1.15: GRPO Advantage Mock / Grouped Reward Dry-Run.

Builds synthetic grouped records and computes GRPO-style advantages.
No GRPO training, no actor.forward, no optimizer.step.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  python scripts/smoke_grpo_advantage_mock.py \
    --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
    --output-dir experiments/phase115_grpo_advantage_mock_10_g4 \
    --num-base-records 10 \
    --group-size 4
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
from src.agents.grpo_advantage_mock import GRPOAdvantageMock, MOCK_GROUP_WARNING
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.15 GRPO advantage mock")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--max-total-length", type=int, default=3072)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_path = Path(args.rollout_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jitter = 0.0 if args.no_synthetic_jitter else args.synthetic_reward_jitter
    normalize_by_std = not args.no_normalize_by_std

    base_records = load_rollout_records(rollout_path, args.num_base_records)
    if not base_records:
        raise SystemExit(f"no records in {rollout_path}")

    grpo_mock = GRPOAdvantageMock(
        group_size=args.group_size,
        normalize_by_std=normalize_by_std,
        synthetic_reward_jitter=jitter,
        seed=args.seed,
    )
    grouped_records = grpo_mock.build_grouped_records(base_records)

    batch_builder = VerlBatchBuilder(
        tokenizer_path=args.tokenizer_path,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        max_total_length=args.max_total_length,
    )
    batch = batch_builder.build_batch(grouped_records)
    check_batch_shapes(batch)

    pad_token_id = batch_builder.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = batch_builder.tokenizer.eos_token_id

    field_builder = VerlTrainingFieldBuilder(pad_token_id=pad_token_id)
    fields = field_builder.build_training_fields(batch)
    check_training_fields(fields)
    attach_group_metadata(fields, grouped_records, args.group_size)

    mock_proto = DataProtoMock.from_fields(fields)
    mock_validate = mock_proto.validate()
    if not mock_validate["passed"]:
        raise SystemExit(f"DataProtoMock validate failed: {mock_validate['errors']}")

    adapter = RealDataProtoAdapter()
    convert_result = adapter.to_real_dataproto(mock_proto)
    data_proto = convert_result["real_proto"] if convert_result["used_real_dataproto"] else mock_proto
    used_real_dataproto = convert_result["used_real_dataproto"]

    adv_output = grpo_mock.compute_group_advantages(data_proto)
    check_result = grpo_mock.check_group_advantages(data_proto, adv_output)

    if not check_result["advantage_check_passed"]:
        raise SystemExit(f"advantage check failed: {check_result}")

    summary = {
        "phase": "1.15",
        "num_base_records": len(base_records),
        "group_size": args.group_size,
        "num_grouped_records": len(grouped_records),
        "used_real_dataproto": used_real_dataproto,
        "normalize_by_std": normalize_by_std,
        "synthetic_reward_jitter": jitter,
        "advantage_check_passed": check_result["advantage_check_passed"],
        "sequence_rewards_shape": check_result["sequence_rewards_shape"],
        "sequence_advantages_shape": check_result["sequence_advantages_shape"],
        "token_level_advantages_shape": check_result["token_level_advantages_shape"],
        "num_groups": check_result["num_groups"],
        "zero_std_group_count": check_result["zero_std_group_count"],
        "zero_std_group_rate": check_result["zero_std_group_rate"],
        "mean_group_reward_std": check_result["mean_group_reward_std"],
        "mean_abs_sequence_advantage": check_result["mean_abs_sequence_advantage"],
        "zero_advantage_token_rate": check_result["zero_advantage_token_rate"],
        "group_mean_advantage_close_to_zero": check_result["group_mean_advantage_close_to_zero"],
        "padding_advantages_zero": check_result["padding_advantages_zero"],
        "is_mock": True,
        "mock_warning": MOCK_GROUP_WARNING,
        "config": {
            "rollout_path": str(rollout_path),
            "tokenizer_path": args.tokenizer_path,
            "num_base_records": args.num_base_records,
            "no_synthetic_jitter": args.no_synthetic_jitter,
        },
    }

    group_advantage_shapes = {
        "sequence_rewards": check_result["sequence_rewards_shape"],
        "sequence_advantages": check_result["sequence_advantages_shape"],
        "token_level_advantages": check_result["token_level_advantages_shape"],
    }
    group_advantage_stats = {
        "num_groups": check_result["num_groups"],
        "group_size": check_result["group_size"],
        "zero_std_group_rate": check_result["zero_std_group_rate"],
        "mean_group_reward_std": check_result["mean_group_reward_std"],
        "min_group_reward_std": check_result["min_group_reward_std"],
        "max_group_reward_std": check_result["max_group_reward_std"],
        "mean_abs_sequence_advantage": check_result["mean_abs_sequence_advantage"],
        "zero_advantage_token_rate": check_result["zero_advantage_token_rate"],
        "group_mean_advantage_close_to_zero": check_result["group_mean_advantage_close_to_zero"],
        "padding_advantages_zero": check_result["padding_advantages_zero"],
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (output_dir / "group_advantage_shapes.json").open("w", encoding="utf-8") as f:
        json.dump(group_advantage_shapes, f, indent=2, ensure_ascii=False)
    with (output_dir / "group_advantage_stats.json").open("w", encoding="utf-8") as f:
        json.dump(group_advantage_stats, f, indent=2, ensure_ascii=False)

    print("\n=== Phase 1.15 GRPO Advantage Mock Summary ===")
    print(f"num_base_records: {summary['num_base_records']}")
    print(f"group_size: {summary['group_size']}")
    print(f"num_grouped_records: {summary['num_grouped_records']}")
    print(f"used_real_dataproto: {summary['used_real_dataproto']}")
    print(f"advantage_check_passed: {summary['advantage_check_passed']}")
    print(f"sequence_advantages_shape: {summary['sequence_advantages_shape']}")
    print(f"token_level_advantages_shape: {summary['token_level_advantages_shape']}")
    print(f"num_groups: {summary['num_groups']}")
    print(f"zero_std_group_rate: {summary['zero_std_group_rate']:.4f}")
    print(f"mean_group_reward_std: {summary['mean_group_reward_std']:.6f}")
    print(f"mean_abs_sequence_advantage: {summary['mean_abs_sequence_advantage']:.4f}")
    print(f"padding_advantages_zero: {summary['padding_advantages_zero']}")
    print(f"output_dir: {output_dir}")
    print(f"\n[phase115] {MOCK_GROUP_WARNING}")
    print(f"[phase115] wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
