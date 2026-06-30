#!/usr/bin/env python3
"""
Phase 1.14: Reference LogProb / KL Dry-Run.

Computes actor/old/ref logprobs and token-level KL diagnostics under torch.no_grad().
No GRPO training, no VERL trainer, no optimizer.step.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  CUDA_VISIBLE_DEVICES=2 python scripts/smoke_ref_kl_dryrun.py \
    --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
    --output-dir experiments/phase114_ref_kl_dryrun_2 \
    --num-records 2 \
    --shared-ref
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
from src.agents.real_dataproto_adapter import RealDataProtoAdapter
from src.agents.ref_kl_dryrun import KL_DRYRUN_WARNING, ReferenceKLDryRun
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
    parser = argparse.ArgumentParser(description="Phase 1.14 reference logprob / KL dry-run")
    parser.add_argument("--rollout-path", type=str, required=True)
    parser.add_argument(
        "--actor-model-path",
        type=str,
        default="/data1/hcc/.hf_home/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--ref-model-path",
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
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--shared-ref", action="store_true", default=True)
    parser.add_argument("--load-separate-ref", action="store_true", help="Load separate ref model")
    parser.add_argument("--max-batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_path = Path(args.rollout_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_ref = not args.load_separate_ref

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

    kl_dryrun = ReferenceKLDryRun(
        actor_model_path=args.actor_model_path,
        ref_model_path=args.ref_model_path,
        dtype=args.dtype,
        device=args.device,
        kl_coef=args.kl_coef,
        shared_ref=shared_ref,
        max_batch_size=args.max_batch_size,
    )
    kl_dryrun.load_actor()
    if not shared_ref:
        kl_dryrun.load_ref()

    actor_output = kl_dryrun.compute_actor_log_probs(data_proto)
    ref_output = kl_dryrun.compute_ref_log_probs(data_proto, actor_output=actor_output)
    kl_output = kl_dryrun.compute_kl(data_proto, actor_output, ref_output)
    check_result = kl_dryrun.check_kl_output(data_proto, kl_output)

    if not check_result["kl_check_passed"]:
        raise SystemExit(f"KL check failed: {check_result}")

    summary = {
        "phase": "1.14",
        "num_records": len(records),
        "used_real_dataproto": used_real_dataproto,
        "actor_model_path": args.actor_model_path,
        "ref_model_path": args.ref_model_path,
        "shared_ref": shared_ref,
        "kl_coef": args.kl_coef,
        "dtype": args.dtype,
        "device": args.device,
        "kl_check_passed": check_result["kl_check_passed"],
        "actor_log_probs_shape": check_result["actor_log_probs_shape"],
        "old_log_probs_shape": check_result["old_log_probs_shape"],
        "ref_log_probs_shape": check_result["ref_log_probs_shape"],
        "token_kl_shape": check_result["token_kl_shape"],
        "ratio_shape": check_result["ratio_shape"],
        "finite_actor_log_probs": check_result["finite_actor_log_probs"],
        "finite_ref_log_probs": check_result["finite_ref_log_probs"],
        "finite_token_kl": check_result["finite_token_kl"],
        "finite_ratio": check_result["finite_ratio"],
        "padding_zero_check": check_result["padding_zero_check"],
        "mean_valid_actor_logprob": check_result["mean_valid_actor_logprob"],
        "mean_valid_ref_logprob": check_result["mean_valid_ref_logprob"],
        "mean_valid_kl": check_result["mean_valid_kl"],
        "mean_valid_abs_kl": check_result["mean_valid_abs_kl"],
        "mean_valid_ratio": check_result["mean_valid_ratio"],
        "min_valid_ratio": check_result["min_valid_ratio"],
        "max_valid_ratio": check_result["max_valid_ratio"],
        "is_dryrun": True,
        "dryrun_warning": KL_DRYRUN_WARNING,
        "config": {
            "rollout_path": str(rollout_path),
            "tokenizer_path": args.tokenizer_path,
            "num_records": args.num_records,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "max_total_length": args.max_total_length,
        },
    }

    kl_shapes = {
        "actor_log_probs": check_result["actor_log_probs_shape"],
        "old_log_probs": check_result["old_log_probs_shape"],
        "ref_log_probs": check_result["ref_log_probs_shape"],
        "token_kl": check_result["token_kl_shape"],
        "ratio": check_result["ratio_shape"],
    }
    kl_stats = {
        "mean_valid_actor_logprob": check_result["mean_valid_actor_logprob"],
        "mean_valid_ref_logprob": check_result["mean_valid_ref_logprob"],
        "mean_valid_kl": check_result["mean_valid_kl"],
        "mean_valid_abs_kl": check_result["mean_valid_abs_kl"],
        "mean_valid_ratio": check_result["mean_valid_ratio"],
        "min_valid_ratio": check_result["min_valid_ratio"],
        "max_valid_ratio": check_result["max_valid_ratio"],
        "padding_zero_check": check_result["padding_zero_check"],
        "shared_ref": shared_ref,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (output_dir / "kl_shapes.json").open("w", encoding="utf-8") as f:
        json.dump(kl_shapes, f, indent=2, ensure_ascii=False)
    with (output_dir / "kl_stats.json").open("w", encoding="utf-8") as f:
        json.dump(kl_stats, f, indent=2, ensure_ascii=False)

    print("\n=== Phase 1.14 Reference LogProb / KL Dry-Run Summary ===")
    print(f"num_records: {summary['num_records']}")
    print(f"used_real_dataproto: {summary['used_real_dataproto']}")
    print(f"shared_ref: {summary['shared_ref']}")
    print(f"kl_coef: {summary['kl_coef']}")
    print(f"kl_check_passed: {summary['kl_check_passed']}")
    print(f"actor_log_probs_shape: {summary['actor_log_probs_shape']}")
    print(f"ref_log_probs_shape: {summary['ref_log_probs_shape']}")
    print(f"token_kl_shape: {summary['token_kl_shape']}")
    print(f"ratio_shape: {summary['ratio_shape']}")
    print(f"mean_valid_actor_logprob: {summary['mean_valid_actor_logprob']:.4f}")
    print(f"mean_valid_ref_logprob: {summary['mean_valid_ref_logprob']:.4f}")
    print(f"mean_valid_kl: {summary['mean_valid_kl']:.6f}")
    print(f"mean_valid_abs_kl: {summary['mean_valid_abs_kl']:.6f}")
    print(f"mean_valid_ratio: {summary['mean_valid_ratio']:.6f}")
    print(f"padding_zero_check: {summary['padding_zero_check']}")
    print(f"output_dir: {output_dir}")
    print(f"\n[phase114] {KL_DRYRUN_WARNING}")
    print(f"[phase114] wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
