#!/usr/bin/env python3
"""
Phase 1.8: VERL Batch Mock / Shape Check.

Reads Phase 1.7 rollout_records.jsonl, tokenizes prompt/response, builds mock batch,
and validates tensor shapes. No GRPO training, no vLLM, no env/BM25.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  python scripts/smoke_verl_batch_mock.py \
    --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
    --output-dir experiments/phase18_verl_batch_mock_10
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


def load_rollout_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_batch(
    records: List[Dict[str, Any]],
    batch: Dict[str, Any],
    shape_check_passed: bool,
) -> Dict[str, Any]:
    rewards = batch["rewards"]
    prompt_lengths = batch["prompt_lengths"].float()
    response_lengths = batch["response_lengths"].float()
    token_meta = batch.get("token_meta", [])

    return {
        "phase": "1.8",
        "num_records": len(records),
        "batch_size": int(batch["input_ids"].shape[0]),
        "seq_len": int(batch["seq_len"]),
        "input_ids_shape": list(batch["input_ids"].shape),
        "attention_mask_shape": list(batch["attention_mask"].shape),
        "response_mask_shape": list(batch["response_mask"].shape),
        "rewards_shape": list(rewards.shape),
        "reward_mean": float(rewards.mean().item()),
        "reward_min": float(rewards.min().item()),
        "reward_max": float(rewards.max().item()),
        "avg_prompt_length": float(prompt_lengths.mean().item()),
        "avg_response_length": float(response_lengths.mean().item()),
        "num_prompt_truncated": sum(1 for m in token_meta if m.get("prompt_truncated")),
        "num_response_truncated": sum(1 for m in token_meta if m.get("response_truncated")),
        "num_total_truncated": sum(1 for m in token_meta if m.get("total_truncated")),
        "shape_check_passed": shape_check_passed,
    }


def build_batch_meta(records: List[Dict[str, Any]], batch: Dict[str, Any]) -> Dict[str, Any]:
    per_record = []
    for i, record in enumerate(records):
        per_record.append(
            {
                "sample_id": batch["sample_ids"][i],
                "reward": float(batch["rewards"][i].item()),
                "prompt_length": int(batch["prompt_lengths"][i].item()),
                "response_length": int(batch["response_lengths"][i].item()),
                "token_meta": batch["token_meta"][i],
                "metrics": batch["metrics"][i],
            }
        )
    return {
        "num_records": len(records),
        "sample_ids": batch["sample_ids"],
        "per_record": per_record,
    }


def build_batch_shapes(batch: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "input_ids": {
            "shape": list(batch["input_ids"].shape),
            "dtype": str(batch["input_ids"].dtype),
        },
        "attention_mask": {
            "shape": list(batch["attention_mask"].shape),
            "dtype": str(batch["attention_mask"].dtype),
        },
        "response_mask": {
            "shape": list(batch["response_mask"].shape),
            "dtype": str(batch["response_mask"].dtype),
        },
        "rewards": {
            "shape": list(batch["rewards"].shape),
            "dtype": str(batch["rewards"].dtype),
        },
        "prompt_lengths": {
            "shape": list(batch["prompt_lengths"].shape),
            "dtype": str(batch["prompt_lengths"].dtype),
        },
        "response_lengths": {
            "shape": list(batch["response_lengths"].shape),
            "dtype": str(batch["response_lengths"].dtype),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.8 VERL batch mock shape check")
    parser.add_argument(
        "--rollout-path",
        type=str,
        required=True,
        help="Path to Phase 1.7 rollout_records.jsonl",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/data1/hcc/.hf_home/Qwen2.5-3B-Instruct",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--max-total-length", type=int, default=3072)
    parser.add_argument("--save-pt", action="store_true", help="Save mock_batch.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollout_path = Path(args.rollout_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_rollout_records(rollout_path)
    if not records:
        raise SystemExit(f"no records in {rollout_path}")

    builder = VerlBatchBuilder(
        tokenizer_path=args.tokenizer_path,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        max_total_length=args.max_total_length,
    )
    batch = builder.build_batch(records)

    shape_check_passed = True
    try:
        check_batch_shapes(batch)
    except AssertionError as exc:
        shape_check_passed = False
        raise SystemExit(f"shape check failed: {exc}") from exc

    summary = summarize_batch(records, batch, shape_check_passed)
    summary["config"] = {
        "rollout_path": str(rollout_path),
        "tokenizer_path": args.tokenizer_path,
        "max_prompt_length": args.max_prompt_length,
        "max_response_length": args.max_response_length,
        "max_total_length": args.max_total_length,
    }

    batch_meta = build_batch_meta(records, batch)
    batch_shapes = build_batch_shapes(batch)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (output_dir / "batch_meta.json").open("w", encoding="utf-8") as f:
        json.dump(batch_meta, f, indent=2, ensure_ascii=False)
    with (output_dir / "batch_shapes.json").open("w", encoding="utf-8") as f:
        json.dump(batch_shapes, f, indent=2, ensure_ascii=False)

    if args.save_pt:
        import torch

        torch.save(
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "response_mask": batch["response_mask"],
                "rewards": batch["rewards"],
                "prompt_lengths": batch["prompt_lengths"],
                "response_lengths": batch["response_lengths"],
            },
            output_dir / "mock_batch.pt",
        )

    print("\n=== Phase 1.8 VERL Batch Mock Summary ===")
    print(f"num_records: {summary['num_records']}")
    print(f"batch_size: {summary['batch_size']}")
    print(f"input_ids_shape: {summary['input_ids_shape']}")
    print(f"attention_mask_shape: {summary['attention_mask_shape']}")
    print(f"response_mask_shape: {summary['response_mask_shape']}")
    print(f"rewards_shape: {summary['rewards_shape']}")
    print(f"reward_mean: {summary['reward_mean']:.4f}")
    print(f"avg_prompt_length: {summary['avg_prompt_length']:.1f}")
    print(f"avg_response_length: {summary['avg_response_length']:.1f}")
    print(f"num_response_truncated: {summary['num_response_truncated']}")
    print(f"shape_check_passed: {summary['shape_check_passed']}")
    print(f"output_dir: {output_dir}")
    print(f"\n[phase18] wrote {output_dir / 'summary.json'}")
    print(f"[phase18] wrote {output_dir / 'batch_shapes.json'}")


if __name__ == "__main__":
    main()
