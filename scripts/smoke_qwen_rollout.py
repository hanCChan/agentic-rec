#!/usr/bin/env python3
"""
Phase 1.5: Qwen2.5-3B rollout smoke — LLM JSON policy + CommerceAgentEnv.

No GRPO, no VERL training.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  CUDA_VISIBLE_DEVICES=2 python scripts/smoke_qwen_rollout.py --num-samples 5
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REC_R1 = ROOT / "Rec-R1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REC_R1))

from src.agents.commerce_agent_env import CommerceAgentEnv
from src.agents.qwen_rollout_policy import QwenRolloutPolicy
from src.tools.bm25_tool import BM25SearchTool


def load_esci_samples(parquet_path: Path, num_samples: int) -> List[Dict[str, Any]]:
    df = pd.read_parquet(parquet_path).head(num_samples)
    samples = []
    for idx, row in df.iterrows():
        targets = [str(x) for x in row["item_id"]]
        samples.append(
            {
                "qid": f"{row.get('data_source', 'esci')}_{idx}",
                "user_query": str(row["query"]),
                "target_items": targets,
            }
        )
    return samples


def _step_penalties(env_step, parse_ok: bool) -> Dict[str, float]:
    penalties = {"search_cost": 0.0, "repeat": 0.0, "invalid": 0.0}
    step = env_step
    if step.invalid:
        if step.invalid_reason == "repeated_query":
            penalties["repeat"] = step.penalty
        else:
            penalties["invalid"] = step.penalty
    elif step.penalty and step.action.get("tool") == "bm25_search":
        penalties["search_cost"] = step.penalty
    if not parse_ok:
        penalties["invalid"] = max(penalties["invalid"], env_step.penalty or 0.0)
    return penalties


def run_episode(
    env: CommerceAgentEnv,
    policy: QwenRolloutPolicy,
    sample: Dict[str, Any],
    default_topk: int,
) -> Dict[str, Any]:
    user_query = sample["user_query"]
    env.reset(
        qid=sample["qid"],
        original_query=user_query,
        target_items=sample["target_items"],
        mode="qwen_rollout",
    )

    search_history: List[Dict[str, Any]] = []
    observation = ""
    rollout_steps: List[Dict[str, Any]] = []
    total_output_tokens = 0
    parse_ok_steps = 0
    total_steps = 0

    while not env.state.done and env.state.step_count < env.max_steps:
        policy_out = policy.act(
            user_query=user_query,
            search_history=search_history,
            observation=observation,
            max_steps=env.max_steps,
        )
        total_output_tokens += policy.count_output_tokens(policy_out["raw_output"])
        total_steps += 1

        if policy_out["parse_ok"]:
            parse_ok_steps += 1
            action_for_env = policy_out["action"]
        else:
            action_for_env = policy_out["raw_output"]

        try:
            result = env.step(action_for_env)
        except Exception as exc:
            rollout_steps.append(
                {
                    "step_id": env.state.step_count,
                    "prompt": policy_out["prompt"],
                    "raw_output": policy_out["raw_output"],
                    "parse_ok": policy_out["parse_ok"],
                    "action": policy_out["action"],
                    "invalid": True,
                    "error": policy_out["error"],
                    "env_error": str(exc),
                }
            )
            break

        step = result["step"]
        penalties = _step_penalties(step, policy_out["parse_ok"])

        step_record = {
            "step_id": step.step_idx,
            "prompt": policy_out["prompt"],
            "raw_output": policy_out["raw_output"],
            "parse_ok": policy_out["parse_ok"],
            "action": step.action if not step.invalid else policy_out["action"],
            "invalid": step.invalid or not policy_out["parse_ok"],
            "invalid_reason": step.invalid_reason or policy_out["error"],
            "observation": step.observation,
            "ndcg_at_10": step.ndcg,
            "recall_at_10": step.recall,
            "delta_ndcg": step.delta_ndcg,
            "step_reward": step.step_reward,
            "penalties": penalties,
        }
        rollout_steps.append(step_record)

        if step.action.get("tool") == "bm25_search" and not step.invalid:
            search_history.append(
                {
                    "query": step.action["query"],
                    "ndcg_at_10": step.ndcg,
                }
            )

        observation = result.get("observation") or step.observation or ""
        if result["done"]:
            break

    traj = env.build_trajectory()
    return {
        "qid": sample["qid"],
        "user_query": user_query,
        "target_items": sample["target_items"],
        "steps": rollout_steps,
        "final_ndcg_at_10": traj.final_ndcg,
        "final_recall_at_10": traj.final_recall,
        "total_reward": traj.total_reward,
        "num_search_calls": traj.num_search_calls,
        "num_invalid_actions": env.state.invalid_count,
        "num_repeated_queries": env.state.repeat_count,
        "finished": env.state.has_final_answer,
        "terminated_reason": traj.terminated_reason,
        "output_tokens": total_output_tokens,
        "parse_ok_steps": parse_ok_steps,
        "total_policy_steps": total_steps,
    }


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}

    def avg(key: str) -> float:
        vals = [r[key] for r in records if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    total_policy_steps = sum(r.get("total_policy_steps", 0) for r in records)
    parse_ok_steps = sum(r.get("parse_ok_steps", 0) for r in records)
    total_invalid = sum(r.get("num_invalid_actions", 0) for r in records)
    total_steps_env = sum(len(r.get("steps", [])) for r in records)

    return {
        "num_samples": len(records),
        "parse_success_rate": parse_ok_steps / total_policy_steps if total_policy_steps else 0.0,
        "invalid_action_rate": total_invalid / total_steps_env if total_steps_env else 0.0,
        "finish_rate": avg("finished"),
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
    parser = argparse.ArgumentParser(description="Phase 1.5 Qwen rollout smoke")
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
        default=ROOT / "experiments/phase15_qwen_rollout_smoke",
    )
    args = parser.parse_args()

    samples = load_esci_samples(args.data, args.num_samples)
    print(f"[phase15] loaded {len(samples)} samples from {args.data}")
    print(f"[phase15] model={args.model_path}")

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
            record = run_episode(env, policy, sample, default_topk=args.topk)
            records.append(record)
            print(
                f"[phase15] {sample['qid']} finished={record['finished']} "
                f"ndcg={record['final_ndcg_at_10']:.4f} reward={record['total_reward']:.4f}"
            )
        except Exception as exc:
            failures.append({"qid": sample["qid"], "error": str(exc), "trace": traceback.format_exc()})
            print(f"[phase15] ERROR {sample['qid']}: {exc}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    traj_path = args.output_dir / "trajectory.jsonl"
    with traj_path.open("w", encoding="utf-8") as fout:
        for rec in records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = summarize(records)
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
    }
    summary["failures"] = failures

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Phase 1.5 Summary ===")
    print_summary(summary)
    print(f"\n[phase15] wrote {traj_path}")
    print(f"[phase15] wrote {summary_path}")


if __name__ == "__main__":
    main()
