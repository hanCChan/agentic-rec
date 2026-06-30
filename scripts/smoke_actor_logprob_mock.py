#!/usr/bin/env python3
"""
Phase 1.12: Actor LogProb Interface Mock / Field Mapping.

Inspects actor logprob input fields, builds request payload, emits mock logprobs.
No actor.forward, no real logprobs, no GRPO training.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  python scripts/smoke_actor_logprob_mock.py \
    --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
    --output-dir experiments/phase112_actor_logprob_mock_10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.actor_logprob_mock import (
    MOCK_LOGPROB_WARNING,
    ActorLogProbInterfaceMock,
    tensor_shape_report,
)
from src.agents.dataproto_mock import DataProtoMock
from src.agents.real_dataproto_adapter import RealDataProtoAdapter
from src.agents.verl_batch_builder import VerlBatchBuilder, check_batch_shapes
from src.agents.verl_training_field_builder import VerlTrainingFieldBuilder, check_training_fields


def load_rollout_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.12 actor logprob interface mock")
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

    mock_proto = DataProtoMock.from_fields(fields)
    mock_validate = mock_proto.validate()
    if not mock_validate["passed"]:
        raise SystemExit(f"DataProtoMock validate failed: {mock_validate['errors']}")

    adapter = RealDataProtoAdapter()
    convert_result = adapter.to_real_dataproto(mock_proto)
    data_proto = convert_result["real_proto"] if convert_result["used_real_dataproto"] else mock_proto
    used_real_dataproto = convert_result["used_real_dataproto"]

    actor_mock = ActorLogProbInterfaceMock()
    inspect_result = actor_mock.inspect_required_fields(data_proto)
    if not inspect_result["actor_input_check_passed"]:
        raise SystemExit(f"actor input check failed: missing {inspect_result['missing_keys']}")

    actor_request = actor_mock.build_actor_logprob_request(data_proto)
    logprob_output = actor_mock.mock_compute_log_prob(actor_request)
    logprob_check = actor_mock.check_logprob_output(data_proto, logprob_output)

    request_shapes = tensor_shape_report(actor_request)
    output_shapes = tensor_shape_report(
        {
            "old_log_probs": logprob_output["old_log_probs"],
            "entropys": logprob_output["entropys"],
        }
    )

    summary = {
        "phase": "1.12",
        "num_records": len(records),
        "used_real_dataproto": used_real_dataproto,
        "actor_input_check_passed": inspect_result["actor_input_check_passed"],
        "logprob_shape_check_passed": logprob_check["logprob_shape_check_passed"],
        "input_ids_shape": inspect_result.get("input_ids_shape"),
        "attention_mask_shape": inspect_result.get("attention_mask_shape"),
        "position_ids_shape": inspect_result.get("position_ids_shape"),
        "responses_shape": inspect_result.get("responses_shape"),
        "response_attention_mask_shape": inspect_result.get("response_attention_mask_shape"),
        "old_log_probs_shape": logprob_check["old_log_probs_shape"],
        "entropys_shape": logprob_check["entropys_shape"],
        "missing_actor_keys": inspect_result["missing_keys"],
        "verl_compute_log_prob_keys": inspect_result["verl_compute_log_prob_keys"],
        "is_mock": True,
        "mock_warning": MOCK_LOGPROB_WARNING,
        "config": {
            "rollout_path": str(rollout_path),
            "tokenizer_path": args.tokenizer_path,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "max_total_length": args.max_total_length,
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (output_dir / "actor_logprob_request_shapes.json").open("w", encoding="utf-8") as f:
        json.dump(request_shapes, f, indent=2, ensure_ascii=False)
    with (output_dir / "mock_logprob_output_shapes.json").open("w", encoding="utf-8") as f:
        json.dump(output_shapes, f, indent=2, ensure_ascii=False)

    print("\n=== Phase 1.12 Actor LogProb Interface Mock Summary ===")
    print(f"num_records: {summary['num_records']}")
    print(f"used_real_dataproto: {summary['used_real_dataproto']}")
    print(f"actor_input_check_passed: {summary['actor_input_check_passed']}")
    print(f"logprob_shape_check_passed: {summary['logprob_shape_check_passed']}")
    print(f"input_ids_shape: {summary['input_ids_shape']}")
    print(f"responses_shape: {summary['responses_shape']}")
    print(f"old_log_probs_shape: {summary['old_log_probs_shape']}")
    print(f"entropys_shape: {summary['entropys_shape']}")
    print(f"missing_actor_keys: {summary['missing_actor_keys']}")
    print(f"is_mock: {summary['is_mock']}")
    print(f"output_dir: {output_dir}")
    print(f"\n[phase112] {MOCK_LOGPROB_WARNING}")
    print(f"[phase112] wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
