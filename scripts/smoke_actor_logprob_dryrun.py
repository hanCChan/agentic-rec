#!/usr/bin/env python3
"""
Phase 1.13: Real Actor LogProb Dry-Run.

Uses HuggingFace AutoModelForCausalLM under torch.no_grad() to compute real
response-token logprobs for a small batch. No GRPO training, no VERL trainer.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  CUDA_VISIBLE_DEVICES=2 python scripts/smoke_actor_logprob_dryrun.py \
    --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
    --output-dir experiments/phase113_actor_logprob_dryrun_2 \
    --num-records 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.actor_logprob_dryrun import DRYRUN_WARNING, ActorLogProbDryRun
from src.agents.dataproto_mock import DataProtoMock
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.13 real actor logprob dry-run")
    parser.add_argument("--rollout-path", type=str, required=True)
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data1/hcc/.hf_home/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/data1/hcc/.hf_home/Qwen2.5-3B-Instruct",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-records", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--max-total-length", type=int, default=3072)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_path = Path(args.rollout_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_rollout_records(rollout_path, args.num_records)
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

    dryrun = ActorLogProbDryRun(
        model_path=args.model_path,
        dtype=args.dtype,
        device=args.device,
        max_batch_size=args.max_batch_size,
    )
    dryrun.load_model()
    logprob_output = dryrun.compute_response_log_probs(data_proto)
    check_result = dryrun.check_real_logprob_output(data_proto, logprob_output)

    if not check_result["real_logprob_check_passed"]:
        raise SystemExit(f"real logprob check failed: {check_result}")

    summary = {
        "phase": "1.13",
        "num_records": len(records),
        "used_real_dataproto": used_real_dataproto,
        "model_path": args.model_path,
        "dtype": args.dtype,
        "device": args.device,
        "real_logprob_check_passed": check_result["real_logprob_check_passed"],
        "input_ids_shape": list(data_proto.batch["input_ids"].shape),
        "responses_shape": list(data_proto.batch["responses"].shape),
        "real_old_log_probs_shape": check_result["real_old_log_probs_shape"],
        "finite_valid_logprobs": check_result["finite_valid_logprobs"],
        "padding_logprobs_zero": check_result["padding_logprobs_zero"],
        "mean_valid_logprob": check_result["mean_valid_logprob"],
        "min_valid_logprob": check_result["min_valid_logprob"],
        "max_valid_logprob": check_result["max_valid_logprob"],
        "is_real": True,
        "dryrun_warning": DRYRUN_WARNING,
        "config": {
            "rollout_path": str(rollout_path),
            "tokenizer_path": args.tokenizer_path,
            "num_records": args.num_records,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "max_total_length": args.max_total_length,
        },
    }

    real_logprob_shapes = {
        "input_ids": list(data_proto.batch["input_ids"].shape),
        "responses": list(data_proto.batch["responses"].shape),
        "real_old_log_probs": check_result["real_old_log_probs_shape"],
        "real_entropys": list(logprob_output["real_entropys"].shape),
    }
    real_logprob_stats = {
        "finite_valid_logprobs": check_result["finite_valid_logprobs"],
        "padding_logprobs_zero": check_result["padding_logprobs_zero"],
        "mean_valid_logprob": check_result["mean_valid_logprob"],
        "min_valid_logprob": check_result["min_valid_logprob"],
        "max_valid_logprob": check_result["max_valid_logprob"],
        "debug_slices": logprob_output.get("debug_slices", []),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (output_dir / "real_logprob_shapes.json").open("w", encoding="utf-8") as f:
        json.dump(real_logprob_shapes, f, indent=2, ensure_ascii=False)
    with (output_dir / "real_logprob_stats.json").open("w", encoding="utf-8") as f:
        json.dump(real_logprob_stats, f, indent=2, ensure_ascii=False)

    print("\n=== Phase 1.13 Real Actor LogProb Dry-Run Summary ===")
    print(f"num_records: {summary['num_records']}")
    print(f"used_real_dataproto: {summary['used_real_dataproto']}")
    print(f"model_path: {summary['model_path']}")
    print(f"dtype: {summary['dtype']}")
    print(f"device: {summary['device']}")
    print(f"real_logprob_check_passed: {summary['real_logprob_check_passed']}")
    print(f"input_ids_shape: {summary['input_ids_shape']}")
    print(f"responses_shape: {summary['responses_shape']}")
    print(f"real_old_log_probs_shape: {summary['real_old_log_probs_shape']}")
    print(f"finite_valid_logprobs: {summary['finite_valid_logprobs']}")
    print(f"padding_logprobs_zero: {summary['padding_logprobs_zero']}")
    print(f"mean_valid_logprob: {summary['mean_valid_logprob']:.4f}")
    print(f"output_dir: {output_dir}")
    print(f"\n[phase113] {DRYRUN_WARNING}")
    print(f"[phase113] wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
