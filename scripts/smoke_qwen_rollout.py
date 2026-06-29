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
from typing import Any, Dict, List, Optional, Tuple

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


def _make_auto_final_action(best_query: str) -> Dict[str, Any]:
    return {
        "tool": "final_answer",
        "final_query": best_query,
        "reason": "auto-finalize on last step using best query by ndcg_at_10",
    }


def _run_env_step(
    env: CommerceAgentEnv,
    action: Any,
    policy_out: Optional[Dict[str, Any]],
    *,
    auto_finish: bool = False,
    llm_called: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Execute env.step and build step record. Returns (result, step_record) or (None, error_record)."""
    try:
        result = env.step(action)
    except Exception as exc:
        record = {
            "step_id": env.state.step_count if env.state else 0,
            "prompt": (policy_out or {}).get("prompt", ""),
            "raw_output": (policy_out or {}).get("raw_output", json.dumps(action)),
            "parse_ok": False if policy_out else True,
            "action": action if isinstance(action, dict) else None,
            "invalid": True,
            "auto_finish": auto_finish,
            "llm_called": llm_called,
            "env_error": str(exc),
        }
        return None, record

    step = result["step"]
    parse_ok = True if auto_finish else bool((policy_out or {}).get("parse_ok"))
    penalties = _step_penalties(step, parse_ok)

    record = {
        "step_id": step.step_idx,
        "prompt": (policy_out or {}).get("prompt", "(auto-finalize; LLM not called on last step)"),
        "raw_output": (policy_out or {}).get("raw_output", json.dumps(action, ensure_ascii=False)),
        "parse_ok": parse_ok,
        "action": step.action if not step.invalid else (policy_out or {}).get("action", action),
        "invalid": step.invalid or (not parse_ok and not auto_finish),
        "invalid_reason": step.invalid_reason or (policy_out or {}).get("error"),
        "observation": step.observation,
        "ndcg_at_10": step.ndcg,
        "recall_at_10": step.recall,
        "delta_ndcg": step.delta_ndcg,
        "step_reward": step.step_reward,
        "penalties": penalties,
        "auto_finish": auto_finish,
        "llm_called": llm_called,
    }
    return result, record


def run_episode(
    env: CommerceAgentEnv,
    policy: QwenRolloutPolicy,
    sample: Dict[str, Any],
) -> Dict[str, Any]:
    user_query = sample["user_query"]
    env.reset(
        qid=sample["qid"],
        original_query=user_query,
        target_items=sample["target_items"],
        mode="qwen_rollout_finish_aware",
    )

    init_metrics = env.evaluate_query(user_query)
    best_query_by_ndcg = user_query
    best_ndcg_at_10 = float(init_metrics["ndcg"])

    search_history: List[Dict[str, Any]] = []
    observation = ""
    rollout_steps: List[Dict[str, Any]] = []
    total_output_tokens = 0
    parse_ok_steps = 0
    total_policy_steps = 0
    llm_finished = False
    auto_finished = False

    while not env.state.done and env.state.step_count < env.max_steps:
        remaining = env.max_steps - env.state.step_count
        current_step = env.state.step_count + 1

        if remaining <= 1:
            auto_action = _make_auto_final_action(best_query_by_ndcg)
            result, step_record = _run_env_step(
                env,
                auto_action,
                policy_out=None,
                auto_finish=True,
                llm_called=False,
            )
            rollout_steps.append(step_record)
            auto_finished = True
            if result is not None:
                break
            continue

        policy_out = policy.act(
            user_query=user_query,
            search_history=search_history,
            observation=observation,
            max_steps=env.max_steps,
            current_step=current_step,
            remaining_steps=remaining,
            best_query_by_ndcg=best_query_by_ndcg,
            best_ndcg_at_10=best_ndcg_at_10,
        )
        total_output_tokens += policy.count_output_tokens(policy_out["raw_output"])
        total_policy_steps += 1

        if policy_out["parse_ok"]:
            parse_ok_steps += 1
            action_for_env = policy_out["action"]
        else:
            action_for_env = policy_out["raw_output"]

        if (
            policy_out["parse_ok"]
            and policy_out["action"]
            and policy_out["action"].get("tool") == "final_answer"
        ):
            llm_finished = True

        result, step_record = _run_env_step(
            env,
            action_for_env,
            policy_out,
            auto_finish=False,
            llm_called=True,
        )
        if result is None:
            rollout_steps.append(step_record)
            break

        rollout_steps.append(step_record)
        step = result["step"]

        if env.state.best_ndcg >= best_ndcg_at_10:
            best_ndcg_at_10 = float(env.state.best_ndcg)
            best_query_by_ndcg = env.state.best_query

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
    finished = bool(env.state.has_final_answer)

    return {
        "qid": sample["qid"],
        "user_query": user_query,
        "target_items": sample["target_items"],
        "best_query_by_ndcg": best_query_by_ndcg,
        "best_ndcg_at_10": best_ndcg_at_10,
        "steps": rollout_steps,
        "final_ndcg_at_10": traj.final_ndcg,
        "final_recall_at_10": traj.final_recall,
        "total_reward": traj.total_reward,
        "num_search_calls": traj.num_search_calls,
        "num_invalid_actions": env.state.invalid_count,
        "num_repeated_queries": env.state.repeat_count,
        "finished": finished,
        "llm_finished": llm_finished,
        "auto_finished": auto_finished,
        "terminated_reason": traj.terminated_reason,
        "output_tokens": total_output_tokens,
        "parse_ok_steps": parse_ok_steps,
        "total_policy_steps": total_policy_steps,
    }


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
            record = run_episode(env, policy, sample)
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
