#!/usr/bin/env python3
"""
Phase 1 smoke test: CommerceAgentEnv + BM25 + process reward.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  python scripts/smoke_agent_env.py --num-samples 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REC_R1 = ROOT / "Rec-R1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REC_R1))

from src.agents.commerce_agent_env import CommerceAgentEnv
from src.agents.rule_policy import RuleSearchPolicy, baseline_original_query_action
from src.agents.trajectory import EpisodeTrajectory, save_trajectories_jsonl
from src.reward.outcome_reward import compute_ndcg
from src.tools.bm25_tool import BM25SearchTool


def load_esci_samples(parquet_path: Path, num_samples: int) -> List[Dict[str, Any]]:
    df = pd.read_parquet(parquet_path).head(num_samples)
    samples = []
    for idx, row in df.iterrows():
        targets = [str(x) for x in row["item_id"]]
        samples.append(
            {
                "qid": f"{row.get('data_source', 'esci')}_{idx}",
                "original_query": str(row["query"]),
                "target_items": targets,
            }
        )
    return samples


def run_baseline(env: CommerceAgentEnv, sample: Dict[str, Any]) -> EpisodeTrajectory:
    env.reset(
        qid=sample["qid"],
        original_query=sample["original_query"],
        target_items=sample["target_items"],
        mode="baseline_single_shot",
    )
    metrics = env.evaluate_query(sample["original_query"])
    baseline_ndcg = metrics["ndcg"]
    env.step(baseline_original_query_action(sample["original_query"]))
    env.step(
        {
            "tool": "final_answer",
            "final_query": sample["original_query"],
            "reason": "baseline uses original query",
        }
    )
    traj = env.build_trajectory(baseline_ndcg=baseline_ndcg)
    traj.baseline_ndcg = baseline_ndcg
    return traj


def run_rule_policy(env: CommerceAgentEnv, sample: Dict[str, Any], baseline_ndcg: float) -> EpisodeTrajectory:
    env.reset(
        qid=sample["qid"],
        original_query=sample["original_query"],
        target_items=sample["target_items"],
        mode="rule_multi_step",
    )
    policy = RuleSearchPolicy(max_steps=env.max_steps, default_topk=env.default_topk)
    policy.reset(sample["original_query"])

    while True:
        remaining = env.max_steps - env.state.step_count
        action = policy.next_action(sample["original_query"], env.state.step_count + 1, remaining)
        result = env.step(action)
        step = result["step"]
        if step.ndcg is not None and action.get("tool") == "bm25_search":
            policy.observe(query=action["query"], ndcg=step.ndcg)
        if result["done"]:
            break

    traj = env.build_trajectory(baseline_ndcg=baseline_ndcg)
    return traj


def summarize(trajectories: List[EpisodeTrajectory]) -> Dict[str, Any]:
    if not trajectories:
        return {}

    def avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "count": len(trajectories),
        "avg_final_ndcg": avg([t.final_ndcg for t in trajectories]),
        "avg_total_reward": avg([t.total_reward for t in trajectories]),
        "avg_search_calls": avg([t.num_search_calls for t in trajectories]),
        "avg_process_reward_sum": avg([t.process_reward_sum for t in trajectories]),
        "avg_total_penalty": avg([t.total_penalty for t in trajectories]),
        "final_answer_rate": avg([1.0 if t.terminated_reason == "final_answer" else 0.0 for t in trajectories]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CommerceAgentEnv Phase 1 smoke test")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "Rec-R1/data/esci/inst/sparse/subset_smoke/val.parquet",
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/phase1_env_smoke",
    )
    args = parser.parse_args()

    samples = load_esci_samples(args.data, args.num_samples)
    search_tool = BM25SearchTool(rec_r1_root=REC_R1)
    env = CommerceAgentEnv(search_tool=search_tool, max_steps=args.max_steps)

    baseline_trajs: List[EpisodeTrajectory] = []
    rule_trajs: List[EpisodeTrajectory] = []

    print(f"[smoke] loaded {len(samples)} samples from {args.data}")

    for sample in samples:
        baseline = run_baseline(env, sample)
        baseline_trajs.append(baseline)
        rule = run_rule_policy(env, sample, baseline_ndcg=baseline.final_ndcg)
        rule_trajs.append(rule)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = out_dir / "baseline_trajectory.jsonl"
    rule_path = out_dir / "rule_policy_trajectory.jsonl"
    save_trajectories_jsonl(baseline_trajs, baseline_path)
    save_trajectories_jsonl(rule_trajs, rule_path)

    summary = {
        "config": {
            "num_samples": args.num_samples,
            "max_steps": args.max_steps,
            "data": str(args.data),
        },
        "baseline_single_shot": summarize(baseline_trajs),
        "rule_multi_step": summarize(rule_trajs),
        "delta_final_ndcg": summarize(rule_trajs)["avg_final_ndcg"] - summarize(baseline_trajs)["avg_final_ndcg"],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[smoke] wrote {baseline_path}")
    print(f"[smoke] wrote {rule_path}")
    print(f"[smoke] wrote {summary_path}")


if __name__ == "__main__":
    main()
