#!/usr/bin/env python3
"""
Phase 1.11: Real verl.DataProto Compatibility Check.

Converts DataProtoMock to real verl.protocol.DataProto when available.
Gracefully falls back to DataProtoMock on import/conversion failure.
No GRPO training, no actor.forward.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  python scripts/smoke_real_dataproto_compat.py \
    --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
    --output-dir experiments/phase111_real_dataproto_compat_10
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
    parser = argparse.ArgumentParser(description="Phase 1.11 real DataProto compatibility check")
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
    import_report = adapter._import_report
    convert_result = adapter.to_real_dataproto(mock_proto)

    inspect_report: Dict[str, Any] = {"real_dataproto_check_passed": False}
    if convert_result["used_real_dataproto"] and convert_result["real_proto"] is not None:
        inspect_report = adapter.inspect_real_dataproto(convert_result["real_proto"], mock_proto)

    summary = {
        "phase": "1.11",
        "num_records": len(records),
        "mock_validate_passed": mock_validate["passed"],
        "verl_import_ok": import_report.get("verl_import_ok", False),
        "tensordict_import_ok": import_report.get("tensordict_import_ok", False),
        "used_real_dataproto": convert_result["used_real_dataproto"],
        "fallback_to_mock": convert_result["fallback_to_mock"],
        "real_dataproto_check_passed": inspect_report.get("real_dataproto_check_passed", False),
        "batch_size": mock_proto.batch_size(),
        "input_ids_shape": list(mock_proto.batch["input_ids"].shape),
        "responses_shape": list(mock_proto.batch["responses"].shape),
        "error": convert_result.get("error"),
        "config": {
            "rollout_path": str(rollout_path),
            "tokenizer_path": args.tokenizer_path,
            "max_prompt_length": args.max_prompt_length,
            "max_response_length": args.max_response_length,
            "max_total_length": args.max_total_length,
        },
    }

    compatibility_report = {
        "import_report": import_report,
        "convert_result": {
            "used_real_dataproto": convert_result["used_real_dataproto"],
            "fallback_to_mock": convert_result["fallback_to_mock"],
            "error": convert_result.get("error"),
        },
        "inspect_report": inspect_report,
        "mock_validate": mock_validate,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (output_dir / "compatibility_report.json").open("w", encoding="utf-8") as f:
        json.dump(compatibility_report, f, indent=2, ensure_ascii=False)

    print("\n=== Phase 1.11 Real DataProto Compatibility Summary ===")
    print(f"num_records: {summary['num_records']}")
    print(f"mock_validate_passed: {summary['mock_validate_passed']}")
    print(f"verl_import_ok: {summary['verl_import_ok']}")
    print(f"tensordict_import_ok: {summary['tensordict_import_ok']}")
    print(f"used_real_dataproto: {summary['used_real_dataproto']}")
    print(f"fallback_to_mock: {summary['fallback_to_mock']}")
    print(f"real_dataproto_check_passed: {summary['real_dataproto_check_passed']}")
    print(f"input_ids_shape: {summary['input_ids_shape']}")
    print(f"responses_shape: {summary['responses_shape']}")
    if summary["error"]:
        print(f"error: {summary['error']}")
    print(f"output_dir: {output_dir}")
    print(f"\n[phase111] wrote {output_dir / 'summary.json'}")
    print(f"[phase111] wrote {output_dir / 'compatibility_report.json'}")


if __name__ == "__main__":
    main()
