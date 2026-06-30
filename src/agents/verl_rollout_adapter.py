"""
Phase 1.7 VERL Rollout Adapter.

This adapter does not launch GRPO training.
It only converts CommerceAgentEnv episodes into VERL-like rollout records:
prompt, response, reward, trajectory, metrics, extra_info.

The next phase will connect this adapter to the real VERL rollout worker.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.agents.commerce_agent_env import CommerceAgentEnv
from src.agents.episode_runner import run_finish_aware_episode
from src.agents.qwen_rollout_policy import QwenRolloutPolicy


def build_actor_prompt(user_query: str, max_steps: int = 3) -> str:
    """Initial prompt for actor generation (VERL-like input)."""
    return (
        "You are an e-commerce search agent. Improve BM25 product search for the user query.\n"
        f"You may take at most {max_steps} tool steps (bm25_search / final_answer).\n"
        "Output one JSON action per step.\n\n"
        f"User shopping query:\n{user_query}"
    )


def build_multistep_response(steps: List[Dict[str, Any]]) -> str:
    """Concatenate multi-step raw outputs and observations for VERL-like response field."""
    parts: List[str] = []
    for step in steps:
        sid = step.get("step_id", "?")
        raw = step.get("raw_output", "")
        parts.append(f"<step_{sid}>\n{raw}\n</step_{sid}>")
        obs = step.get("observation")
        if obs:
            parts.append(f"<observation_{sid}>\n{obs}\n</observation_{sid}>")
    return "\n".join(parts)


class VerlRolloutAdapter:
    """Convert CommerceAgentEnv + QwenRolloutPolicy episodes to VERL-like rollout records."""

    def __init__(
        self,
        env: CommerceAgentEnv,
        policy: QwenRolloutPolicy,
        max_steps: int = 3,
        topk: int = 20,
    ):
        self.env = env
        self.policy = policy
        self.max_steps = max_steps
        self.topk = topk

    def run_one(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Run one episode and return a VERL-like rollout record."""
        trajectory = run_finish_aware_episode(self.env, self.policy, sample)
        response = build_multistep_response(trajectory["steps"])
        prompt = build_actor_prompt(trajectory["user_query"], max_steps=self.max_steps)
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
        }

    def run_batch(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.run_one(sample) for sample in samples]
