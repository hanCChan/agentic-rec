#!/usr/bin/env python3
"""
Phase 1.6: Finish-Aware Qwen rollout smoke.

Extends Phase 1.5 with:
  - finish-aware prompt (remaining steps + best_query_by_ndcg)
  - last-step auto-finalize
  - llm_finish_rate / auto_finish_rate metrics

No GRPO, no VERL training.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  CUDA_VISIBLE_DEVICES=2 python scripts/smoke_qwen_rollout.py --num-samples 10
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

from src.agents.commerce_agent_env import CommerceAgentEnv
from src.agents.episode_runner import load_esci_samples, run_finish_aware_episode
from src.agents.qwen_rollout_policy import QwenRolloutPolicy
from src.tools.bm25_tool import BM25SearchTool


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}

    def avg(key: str) -> float:
        vals = [float(r[key]) for r in records if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def rate(key: str) -> float:
        if not records:
            return 0.0
        return sum(1 for r in records if r.get(key)) / len(records)

    total_policy_steps = sum(r.get("total_policy_steps", 0) for r in records)
    parse_ok_steps = sum(r.get("parse_ok_steps", 0) for r in records)
    total_invalid = sum(r.get("num_invalid_actions", 0) for r in records)
    total_steps_env = sum(len(r.get("steps", [])) for r in records)

    return {
        "num_samples": len(records),
        "parse_success_rate": parse_ok_steps / total_policy_steps if total_policy_steps else 1.0,
        "invalid_action_rate": total_invalid / total_steps_env if total_steps_env else 0.0,
        "finish_rate": rate("finished"),
        "llm_finish_rate": rate("llm_finished"),
        "auto_finish_rate": rate("auto_finished"),
        "avg_search_calls": avg("num_search_calls"),
        "avg_repeated_queries": avg("num_repeated_queries"),
        "avg_ndcg_at_10": avg("final_ndcg_at_10"),
        "avg_recall_at_10": avg("final_recall_at_10"),
        "avg_total_reward": avg("total_reward"),
        "avg_output_tokens": avg("output_tokens"),
    }


def print_summary(summary: Dict[str, Any]) -> None:
    keys = [
        "num_samples",
        "parse_success_rate",
        "invalid_action_rate",
        "finish_rate",
        "llm_finish_rate",
        "auto_finish_rate",
        "avg_search_calls",
        "avg_repeated_queries",
        "avg_ndcg_at_10",
        "avg_recall_at_10",
        "avg_total_reward",
        "avg_output_tokens",
    ]
    for key in keys:
        val = summary.get(key)
        if isinstance(val, float):
            print(f"{key}: {val:.4f}")
        else:
            print(f"{key}: {val}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.6 Finish-Aware Qwen rollout smoke")
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
        default=ROOT / "experiments/phase16_finish_aware_smoke",
    )
    args = parser.parse_args()

    samples = load_esci_samples(args.data, args.num_samples)
    print(f"[phase16] loaded {len(samples)} samples from {args.data}")
    print(f"[phase16] model={args.model_path}")

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

    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for sample in samples:
        try:
            record = run_finish_aware_episode(env, policy, sample)
            records.append(record)
            print(
                f"[phase16] {sample['qid']} finished={record['finished']} "
                f"llm={record['llm_finished']} auto={record['auto_finished']} "
                f"ndcg={record['final_ndcg_at_10']:.4f} reward={record['total_reward']:.4f}"
            )
        except Exception as exc:
            failures.append({"qid": sample["qid"], "error": str(exc), "trace": traceback.format_exc()})
            print(f"[phase16] ERROR {sample['qid']}: {exc}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    traj_path = args.output_dir / "trajectory.jsonl"
    with traj_path.open("w", encoding="utf-8") as fout:
        for rec in records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = summarize(records)
    summary["phase"] = "1.6"
    summary["config"] = {
        "num_samples": args.num_samples,
        "max_steps": args.max_steps,
        "topk": args.topk,
        "model_path": str(args.model_path),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "data": str(args.data),
        "finish_aware": True,
        "auto_finalize_last_step": True,
    }
    summary["failures"] = failures

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Phase 1.6 Summary ===")
    print_summary(summary)
    print(f"\n[phase16] wrote {traj_path}")
    print(f"[phase16] wrote {summary_path}")


if __name__ == "__main__":
    main()
