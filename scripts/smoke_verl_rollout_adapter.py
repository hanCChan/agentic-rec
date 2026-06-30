#!/usr/bin/env python3
"""
Phase 1.7: VERL Rollout Adapter smoke test.

Converts CommerceAgentEnv episodes to VERL-like rollout_records.jsonl.
No GRPO training, no VERL trainer.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  CUDA_VISIBLE_DEVICES=2 python scripts/smoke_verl_rollout_adapter.py --num-samples 10
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
REC_R1 = ROOT / "Rec-R1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REC_R1))

from src.agents.episode_runner import load_esci_samples
from src.agents.commerce_agent_env import CommerceAgentEnv
from src.agents.qwen_rollout_policy import QwenRolloutPolicy
from src.agents.verl_rollout_adapter import VerlRolloutAdapter
from src.tools.bm25_tool import BM25SearchTool


REQUIRED_RECORD_KEYS = ("sample_id", "prompt", "response", "reward", "trajectory", "metrics", "extra_info")


def summarize_rollout_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"num_rollout_records": 0}

    def avg_metric(key: str) -> float:
        vals = [float(r["metrics"][key]) for r in records if r.get("metrics", {}).get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def avg_reward() -> float:
        return sum(float(r["reward"]) for r in records) / len(records)

    def rate_metric(key: str) -> float:
        return sum(1 for r in records if r.get("metrics", {}).get(key)) / len(records)

    total_policy_steps = sum(r["extra_info"].get("total_policy_steps", 0) for r in records)
    parse_ok_steps = sum(r["extra_info"].get("parse_ok_steps", 0) for r in records)
    total_invalid = sum(r["metrics"].get("num_invalid_actions", 0) for r in records)
    total_env_steps = sum(len(r["trajectory"].get("steps", [])) for r in records)
    avg_response_chars = sum(len(r.get("response", "")) for r in records) / len(records)

    return {
        "num_samples": len(records),
        "num_rollout_records": len(records),
        "avg_reward": avg_reward(),
        "avg_ndcg_at_10": avg_metric("ndcg_at_10"),
        "avg_recall_at_10": avg_metric("recall_at_10"),
        "finish_rate": rate_metric("finished"),
        "llm_finish_rate": rate_metric("llm_finished"),
        "auto_finish_rate": rate_metric("auto_finished"),
        "parse_success_rate": parse_ok_steps / total_policy_steps if total_policy_steps else 1.0,
        "invalid_action_rate": total_invalid / total_env_steps if total_env_steps else 0.0,
        "avg_search_calls": avg_metric("num_search_calls"),
        "avg_response_chars": avg_response_chars,
    }


def validate_record(record: Dict[str, Any]) -> None:
    for key in REQUIRED_RECORD_KEYS:
        if key not in record:
            raise ValueError(f"missing required key: {key}")
    if not isinstance(record["reward"], (int, float)):
        raise TypeError("reward must be float")
    if not isinstance(record["trajectory"], dict):
        raise TypeError("trajectory must be dict")
    if "steps" not in record["trajectory"]:
        raise ValueError("trajectory.steps missing")


def print_summary(summary: Dict[str, Any], output_dir: Path) -> None:
    keys = [
        "num_samples",
        "num_rollout_records",
        "avg_reward",
        "avg_ndcg_at_10",
        "finish_rate",
        "llm_finish_rate",
        "auto_finish_rate",
        "invalid_action_rate",
        "avg_search_calls",
    ]
    print("\n=== Phase 1.7 VERL Adapter Summary ===")
    for key in keys:
        val = summary.get(key)
        if isinstance(val, float):
            print(f"{key}: {val:.4f}")
        else:
            print(f"{key}: {val}")
    print(f"output_dir: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.7 VERL rollout adapter smoke")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "Rec-R1/data/esci/inst/sparse/subset_smoke/val.parquet",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data1/hcc/.hf_home/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/phase17_verl_adapter_smoke",
    )
    args = parser.parse_args()

    samples = load_esci_samples(args.data, args.num_samples)
    print(f"[phase17] loaded {len(samples)} samples from {args.data}")

    search_tool = BM25SearchTool(rec_r1_root=REC_R1)
    env = CommerceAgentEnv(
        search_tool=search_tool,
        max_steps=args.max_steps,
        default_topk=args.topk,
    )
    policy = QwenRolloutPolicy(
        model_path=str(args.model_path),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    adapter = VerlRolloutAdapter(
        env=env,
        policy=policy,
        max_steps=args.max_steps,
        topk=args.topk,
    )

    rollout_records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for sample in samples:
        try:
            record = adapter.run_one(sample)
            validate_record(record)
            rollout_records.append(record)
            m = record["metrics"]
            print(
                f"[phase17] {record['sample_id']} reward={record['reward']:.4f} "
                f"ndcg={m['ndcg_at_10']:.4f} finished={m['finished']}"
            )
        except Exception as exc:
            failures.append({"qid": sample["qid"], "error": str(exc), "trace": traceback.format_exc()})
            print(f"[phase17] ERROR {sample['qid']}: {exc}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = args.output_dir / "rollout_records.jsonl"
    with rollout_path.open("w", encoding="utf-8") as fout:
        for rec in rollout_records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = summarize_rollout_records(rollout_records)
    summary["phase"] = "1.7"
    summary["config"] = {
        "num_samples": args.num_samples,
        "max_steps": args.max_steps,
        "topk": args.topk,
        "model_path": str(args.model_path),
        "data": str(args.data),
    }
    summary["failures"] = failures

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(summary, args.output_dir)
    print(f"\n[phase17] wrote {rollout_path}")
    print(f"[phase17] wrote {summary_path}")


if __name__ == "__main__":
    main()
