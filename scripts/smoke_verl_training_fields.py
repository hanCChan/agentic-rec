#!/usr/bin/env python3
"""
Phase 1.9: VERL Training Fields Mock / Field Alignment.

Reads Phase 1.7 rollout_records.jsonl, builds Phase 1.8 batch, then adds VERL-like
training fields with mock logprob/advantage placeholders. No GRPO training.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  python scripts/smoke_verl_training_fields.py \
    --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
    --output-dir experiments/phase19_verl_training_fields_mock_10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.verl_batch_builder import VerlBatchBuilder, check_batch_shapes
from src.agents.verl_training_field_builder import (
    MOCK_FIELDS_WARNING,
    VerlTrainingFieldBuilder,
    check_training_fields,
)


def load_rollout_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def count_nonzero_token_rewards(fields: Dict[str, Any]) -> int:
    per_row = (fields["token_level_rewards"].abs().sum(dim=1) > 0).sum().item()
    return int(per_row)


def build_training_fields_shapes(fields: Dict[str, Any]) -> Dict[str, Any]:
    def shape_dtype(key: str) -> Dict[str, Any]:
        tensor = fields[key]
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}

    return {
        "input_ids": shape_dtype("input_ids"),
        "attention_mask": shape_dtype("attention_mask"),
        "position_ids": shape_dtype("position_ids"),
        "prompts": shape_dtype("prompts"),
        "responses": shape_dtype("responses"),
        "response_attention_mask": shape_dtype("response_attention_mask"),
        "response_mask": shape_dtype("response_mask"),
        "sequence_rewards": shape_dtype("sequence_rewards"),
        "token_level_rewards": shape_dtype("token_level_rewards"),
        "mock_old_log_probs": shape_dtype("mock_old_log_probs"),
        "mock_advantages": shape_dtype("mock_advantages"),
        "mock_returns": shape_dtype("mock_returns"),
        "prompt_lengths": shape_dtype("prompt_lengths"),
        "response_lengths": shape_dtype("response_lengths"),
    }


def summarize_fields(
    records: List[Dict[str, Any]],
    fields: Dict[str, Any],
    shape_check_passed: bool,
) -> Dict[str, Any]:
    return {
        "phase": "1.9",
        "num_records": len(records),
        "batch_size": int(fields["input_ids"].shape[0]),
        "seq_len": int(fields["seq_len"]),
        "prompt_width": int(fields["prompt_width"]),
        "response_width": int(fields["response_width"]),
        "input_ids_shape": list(fields["input_ids"].shape),
        "position_ids_shape": list(fields["position_ids"].shape),
        "prompts_shape": list(fields["prompts"].shape),
        "responses_shape": list(fields["responses"].shape),
        "response_attention_mask_shape": list(fields["response_attention_mask"].shape),
        "sequence_rewards_shape": list(fields["sequence_rewards"].shape),
        "token_level_rewards_shape": list(fields["token_level_rewards"].shape),
        "mock_old_log_probs_shape": list(fields["mock_old_log_probs"].shape),
        "mock_advantages_shape": list(fields["mock_advantages"].shape),
        "mock_returns_shape": list(fields["mock_returns"].shape),
        "reward_mean": float(fields["sequence_rewards"].mean().item()),
        "reward_min": float(fields["sequence_rewards"].min().item()),
        "reward_max": float(fields["sequence_rewards"].max().item()),
        "num_nonzero_token_rewards": count_nonzero_token_rewards(fields),
        "shape_check_passed": shape_check_passed,
        "mock_fields_warning": MOCK_FIELDS_WARNING,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.9 VERL training fields mock")
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
    parser.add_argument("--save-pt", action="store_true", help="Save mock_training_fields.pt")
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

    shape_check_passed = True
    try:
        check_training_fields(fields)
    except AssertionError as exc:
        shape_check_passed = False
        raise SystemExit(f"training fields shape check failed: {exc}") from exc

    summary = summarize_fields(records, fields, shape_check_passed)
    summary["config"] = {
        "rollout_path": str(rollout_path),
        "tokenizer_path": args.tokenizer_path,
        "max_prompt_length": args.max_prompt_length,
        "max_response_length": args.max_response_length,
        "max_total_length": args.max_total_length,
    }

    shapes = build_training_fields_shapes(fields)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (output_dir / "training_fields_shapes.json").open("w", encoding="utf-8") as f:
        json.dump(shapes, f, indent=2, ensure_ascii=False)

    if args.save_pt:
        import torch

        torch.save(
            {
                "input_ids": fields["input_ids"],
                "attention_mask": fields["attention_mask"],
                "position_ids": fields["position_ids"],
                "prompts": fields["prompts"],
                "responses": fields["responses"],
                "response_attention_mask": fields["response_attention_mask"],
                "response_mask": fields["response_mask"],
                "sequence_rewards": fields["sequence_rewards"],
                "token_level_rewards": fields["token_level_rewards"],
                "mock_old_log_probs": fields["mock_old_log_probs"],
                "mock_advantages": fields["mock_advantages"],
                "mock_returns": fields["mock_returns"],
                "prompt_lengths": fields["prompt_lengths"],
                "response_lengths": fields["response_lengths"],
            },
            output_dir / "mock_training_fields.pt",
        )

    print("\n=== Phase 1.9 VERL Training Fields Mock Summary ===")
    print(f"num_records: {summary['num_records']}")
    print(f"batch_size: {summary['batch_size']}")
    print(f"input_ids_shape: {summary['input_ids_shape']}")
    print(f"prompts_shape: {summary['prompts_shape']}")
    print(f"responses_shape: {summary['responses_shape']}")
    print(f"token_level_rewards_shape: {summary['token_level_rewards_shape']}")
    print(f"mock_old_log_probs_shape: {summary['mock_old_log_probs_shape']}")
    print(f"reward_mean: {summary['reward_mean']:.4f}")
    print(f"num_nonzero_token_rewards: {summary['num_nonzero_token_rewards']}")
    print(f"shape_check_passed: {summary['shape_check_passed']}")
    print(f"output_dir: {output_dir}")
    print(f"\n[phase19] {MOCK_FIELDS_WARNING}")
    print(f"[phase19] wrote {output_dir / 'summary.json'}")
    print(f"[phase19] wrote {output_dir / 'training_fields_shapes.json'}")


if __name__ == "__main__":
    main()
