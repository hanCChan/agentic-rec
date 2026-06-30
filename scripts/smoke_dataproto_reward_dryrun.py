#!/usr/bin/env python3
"""
Phase 1.10: DataProto / Reward Function Dry-Run.

Maps Phase 1.9 training fields to DataProtoMock, validates payload,
checks actor input fields, and runs CommerceRewardFn dry-run.
No GRPO training, no VERL trainer, no actor.forward.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  python scripts/smoke_dataproto_reward_dryrun.py \
    --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
    --output-dir experiments/phase110_dataproto_reward_dryrun_10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.dataproto_mock import (
    DRY_RUN_WARNING,
    DataProtoMock,
    build_dataproto_shapes,
    check_actor_inputs,
)
from src.agents.verl_batch_builder import VerlBatchBuilder, check_batch_shapes
from src.agents.verl_training_field_builder import VerlTrainingFieldBuilder, check_training_fields
from src.reward.commerce_reward_fn import CommerceRewardFn


def load_rollout_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.10 DataProto reward dry-run")
    parser.add_argument("--rollout-path", type=str, required=True)
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/data1/hcc/.hf_home/Qwen2.5-3B-Instruct",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--max-total-length", type=int, default=3072)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_path = Path(args.rollout_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_rollout_records(rollout_path)
    if not records:
        raise SystemExit(f"no records in {rollout_path}")

    batch_builder = VerlBatchBuilder(
        tokenizer_path=args.tokenizer_path,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        max_total_length=args.max_total_length,
    )
    batch = batch_builder.build_batch(records)
    check_batch_shapes(batch)

    pad_token_id = batch_builder.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = batch_builder.tokenizer.eos_token_id

    field_builder = VerlTrainingFieldBuilder(pad_token_id=pad_token_id)
    fields = field_builder.build_training_fields(batch)
    check_training_fields(fields)

    data_proto = DataProtoMock.from_fields(fields)
    validate_result = data_proto.validate()
    if not validate_result["passed"]:
        raise SystemExit(f"DataProtoMock validate failed: {validate_result['errors']}")

    actor_check = check_actor_inputs(data_proto)
    if not actor_check["actor_input_check_passed"]:
        raise SystemExit(f"actor input check failed: missing {actor_check['missing_keys']}")

    reward_fn = CommerceRewardFn()
    reward_output = reward_fn(data_proto)
    if not reward_output["check_passed"]:
        raise SystemExit("reward_fn check failed: num_nonzero_token_rewards != batch_size")

    summary = {
        "phase": "1.10",
        "num_records": len(records),
        "batch_size": data_proto.batch_size(),
        "dataproto_validate_passed": validate_result["passed"],
        "actor_input_check_passed": actor_check["actor_input_check_passed"],
        "reward_fn_check_passed": reward_output["check_passed"],
        "input_ids_shape": list(data_proto.batch["input_ids"].shape),
        "responses_shape": list(data_proto.batch["responses"].shape),
        "token_level_rewards_shape": list(data_proto.batch["token_level_rewards"].shape),
        "sequence_rewards_shape": list(data_proto.batch["sequence_rewards"].shape),
        "reward_mean": reward_output["metrics"]["reward_mean"],
        "reward_min": reward_output["metrics"]["reward_min"],
        "reward_max": reward_output["metrics"]["reward_max"],
        "num_nonzero_token_rewards": reward_output["metrics"]["num_nonzero_token_rewards"],
        "missing_actor_keys": actor_check["missing_keys"],
        "mock_warning": DRY_RUN_WARNING,
        "config": {
            "rollout_path": str(rollout_path),
            "tokenizer_path": args.tokenizer_path,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "max_total_length": args.max_total_length,
        },
    }

    dataproto_shapes = build_dataproto_shapes(data_proto)
    reward_fn_output = {
        "metrics": reward_output["metrics"],
        "sequence_rewards_shape": list(reward_output["sequence_rewards"].shape),
        "token_level_rewards_shape": list(reward_output["token_level_rewards"].shape),
        "check_passed": reward_output["check_passed"],
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (output_dir / "dataproto_shapes.json").open("w", encoding="utf-8") as f:
        json.dump(dataproto_shapes, f, indent=2, ensure_ascii=False)
    with (output_dir / "reward_fn_output.json").open("w", encoding="utf-8") as f:
        json.dump(reward_fn_output, f, indent=2, ensure_ascii=False)

    print("\n=== Phase 1.10 DataProto / Reward Dry-Run Summary ===")
    print(f"num_records: {summary['num_records']}")
    print(f"batch_size: {summary['batch_size']}")
    print(f"dataproto_validate_passed: {summary['dataproto_validate_passed']}")
    print(f"actor_input_check_passed: {summary['actor_input_check_passed']}")
    print(f"reward_fn_check_passed: {summary['reward_fn_check_passed']}")
    print(f"input_ids_shape: {summary['input_ids_shape']}")
    print(f"responses_shape: {summary['responses_shape']}")
    print(f"token_level_rewards_shape: {summary['token_level_rewards_shape']}")
    print(f"reward_mean: {summary['reward_mean']:.4f}")
    print(f"num_nonzero_token_rewards: {summary['num_nonzero_token_rewards']}")
    print(f"output_dir: {output_dir}")
    print(f"\n[phase110] {DRY_RUN_WARNING}")
    print(f"[phase110] wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
