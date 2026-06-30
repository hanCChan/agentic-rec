"""
Phase 1.17 real multi-sample rollout runner.

For each base query/sample, run G independent Qwen tool-use episodes,
collect real BM25 rewards, and return grouped rollout records.

This class does NOT train and does NOT connect to GRPO trainer.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Optional

from src.agents.episode_runner import run_finish_aware_episode
from src.agents.verl_rollout_adapter import build_actor_prompt, build_multistep_response


def _extract_search_queries(trajectory: Dict[str, Any]) -> List[str]:
    queries: List[str] = []
    for step in trajectory.get("steps", []):
        action = step.get("action") or {}
        if action.get("tool") == "bm25_search":
            query = action.get("query", "")
            if query:
                queries.append(str(query))
    return queries


def _trajectory_fingerprint(trajectory: Dict[str, Any]) -> str:
    parts: List[str] = []
    for step in trajectory.get("steps", []):
        action = step.get("action") or {}
        tool = action.get("tool", "")
        if tool == "bm25_search":
            parts.append(f"search:{action.get('query', '')}")
        elif tool == "final_answer":
            parts.append(f"final:{action.get('final_query', '')}")
        elif tool == "invalid":
            parts.append("invalid")
        else:
            parts.append(str(tool or "none"))
    return "|".join(parts)


def trajectory_to_rollout_record(
    trajectory: Dict[str, Any],
    *,
    group_id: str,
    group_index: int,
    group_size: int,
    sample_seed: int,
    sampling_temperature: float,
    sampling_top_p: float,
    max_steps: int = 3,
) -> Dict[str, Any]:
    """Convert a finish-aware trajectory to a Phase 1.7-compatible rollout record."""
    response = build_multistep_response(trajectory["steps"])
    prompt = build_actor_prompt(trajectory["user_query"], max_steps=max_steps)
    reward = float(trajectory["total_reward"])
    final_query = trajectory.get("final_query") or trajectory["best_query_by_ndcg"]
    best_query = trajectory["best_query_by_ndcg"]

    return {
        "sample_id": trajectory["qid"],
        "prompt": prompt,
        "response": response,
        "reward": reward,
        "trajectory": trajectory,
        "metrics": {
            "ndcg_at_10": float(trajectory["final_ndcg_at_10"]),
            "recall_at_10": float(trajectory["final_recall_at_10"]),
            "total_reward": reward,
            "num_search_calls": int(trajectory["num_search_calls"]),
            "num_invalid_actions": int(trajectory["num_invalid_actions"]),
            "num_repeated_queries": int(trajectory["num_repeated_queries"]),
            "finished": bool(trajectory["finished"]),
            "llm_finished": bool(trajectory["llm_finished"]),
            "auto_finished": bool(trajectory["auto_finished"]),
        },
        "extra_info": {
            "original_query": trajectory["user_query"],
            "best_query_by_ndcg": best_query,
            "best_ndcg_at_10": float(trajectory["best_ndcg_at_10"]),
            "final_query": final_query,
            "final_query_is_best": final_query.strip().lower() == best_query.strip().lower(),
            "terminated_reason": trajectory["terminated_reason"],
            "parse_ok_steps": int(trajectory["parse_ok_steps"]),
            "total_policy_steps": int(trajectory["total_policy_steps"]),
        },
        "group_id": group_id,
        "group_index": group_index,
        "group_size": group_size,
        "is_real_multisample": True,
        "is_synthetic_group_member": False,
        "sample_seed": sample_seed,
        "sampling_temperature": sampling_temperature,
        "sampling_top_p": sampling_top_p,
    }


def compute_group_metrics(records: List[Dict[str, Any]], eps: float = 1e-6) -> Dict[str, Any]:
    """Aggregate diversity and reward statistics for one GRPO group."""
    rewards = [float(r["reward"]) for r in records]
    ndcgs = [float(r["metrics"]["ndcg_at_10"]) for r in records]

    reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
    reward_std = (
        (sum((x - reward_mean) ** 2 for x in rewards) / len(rewards)) ** 0.5 if len(rewards) > 1 else 0.0
    )
    ndcg_mean = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
    ndcg_std = (
        (sum((x - ndcg_mean) ** 2 for x in ndcgs) / len(ndcgs)) ** 0.5 if len(ndcgs) > 1 else 0.0
    )

    all_search_queries: List[str] = []
    final_queries: List[str] = []
    fingerprints: List[str] = []
    total_policy_steps = 0
    parse_ok_steps = 0
    total_invalid = 0
    total_env_steps = 0

    for record in records:
        traj = record["trajectory"]
        all_search_queries.extend(_extract_search_queries(traj))
        final_query = record["extra_info"].get("final_query") or traj.get("final_query") or ""
        final_queries.append(str(final_query))
        fingerprints.append(_trajectory_fingerprint(traj))
        total_policy_steps += int(traj.get("total_policy_steps", 0))
        parse_ok_steps += int(traj.get("parse_ok_steps", 0))
        total_invalid += int(traj.get("num_invalid_actions", 0))
        total_env_steps += len(traj.get("steps", []))

    unique_search_query_count = len(set(all_search_queries))
    unique_final_query_count = len(set(final_queries))
    unique_trajectory_count = len(set(fingerprints))

    return {
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "reward_min": min(rewards) if rewards else 0.0,
        "reward_max": max(rewards) if rewards else 0.0,
        "ndcg_mean": ndcg_mean,
        "ndcg_std": ndcg_std,
        "finish_rate": sum(1 for r in records if r["metrics"]["finished"]) / len(records),
        "llm_finish_rate": sum(1 for r in records if r["metrics"]["llm_finished"]) / len(records),
        "auto_finish_rate": sum(1 for r in records if r["metrics"]["auto_finished"]) / len(records),
        "invalid_action_rate": total_invalid / total_env_steps if total_env_steps else 0.0,
        "json_parse_success_rate": parse_ok_steps / total_policy_steps if total_policy_steps else 1.0,
        "unique_search_query_count": unique_search_query_count,
        "unique_final_query_count": unique_final_query_count,
        "unique_trajectory_count": unique_trajectory_count,
        "zero_std_reward": reward_std < eps,
        "all_same_final_query": unique_final_query_count == 1,
        "all_same_trajectory": unique_trajectory_count == 1,
    }


class MultiSampleEpisodeRunner:
    """
    Phase 1.17 real multi-sample rollout runner.

    For each base query/sample, run G independent Qwen tool-use episodes,
    collect real BM25 rewards, and return grouped rollout records.

    This class does NOT train and does NOT connect to GRPO trainer.
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        policy: Any,
        group_size: int = 4,
        max_steps: int = 3,
        topk: int = 20,
        base_seed: int = 42,
        sampling_temperature: float = 0.7,
        sampling_top_p: float = 0.95,
    ):
        self.env_factory = env_factory
        self.policy = policy
        self.group_size = group_size
        self.max_steps = max_steps
        self.topk = topk
        self.base_seed = base_seed
        self.sampling_temperature = sampling_temperature
        self.sampling_top_p = sampling_top_p

    def _member_seed(self, group_id: str, group_index: int) -> int:
        gid_hash = int(hashlib.md5(group_id.encode("utf-8")).hexdigest()[:8], 16)
        return self.base_seed + group_index * 1000 + (gid_hash % 1000)

    def run_group(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Run G independent episodes for one base sample and return grouped records."""
        group_id = sample.get("qid") or sample.get("sample_id") or "unknown"
        original_query = sample.get("query") or sample.get("user_query") or ""
        records: List[Dict[str, Any]] = []

        for group_index in range(self.group_size):
            sample_seed = self._member_seed(group_id, group_index)
            episode_sample = {**sample, "qid": f"{group_id}#g{group_index}"}
            env = self.env_factory()
            trajectory = run_finish_aware_episode(
                env,
                self.policy,
                episode_sample,
                sample_seed=sample_seed,
            )
            record = trajectory_to_rollout_record(
                trajectory,
                group_id=group_id,
                group_index=group_index,
                group_size=self.group_size,
                sample_seed=sample_seed,
                sampling_temperature=self.sampling_temperature,
                sampling_top_p=self.sampling_top_p,
                max_steps=self.max_steps,
            )
            records.append(record)

        group_metrics = compute_group_metrics(records)
        return {
            "group_id": group_id,
            "original_query": original_query,
            "group_size": self.group_size,
            "records": records,
            "group_metrics": group_metrics,
        }

    def run_groups(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.run_group(sample) for sample in samples]
