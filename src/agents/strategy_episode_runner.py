"""
Phase 1.18d strategy-controlled rollout runner.

For each base query, run one episode per predefined search strategy.
Does NOT train and does NOT change reward.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Dict, List, Optional

from src.agents.episode_runner import run_finish_aware_episode
from src.agents.multisample_episode_runner import compute_group_metrics, trajectory_to_rollout_record
from src.agents.rollout_diagnostics import token_jaccard
from src.agents.search_strategy_prompts import DEFAULT_STRATEGY_ORDER, get_strategy, validate_strategies


def _trajectory_fingerprint(trajectory: Dict[str, Any]) -> str:
    parts: List[str] = []
    for step in trajectory.get("steps", []):
        action = step.get("action") or {}
        tool = action.get("tool", "")
        if tool == "bm25_search":
            parts.append(f"search:{action.get('query', '')}")
        elif tool == "final_answer":
            parts.append(f"final:{action.get('final_query', '')}")
    return "|".join(parts)


def _pairwise_average(values: List[str], metric_fn) -> float:
    if len(values) < 2:
        return 1.0 if values else 0.0
    scores = [metric_fn(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values))]
    return sum(scores) / len(scores) if scores else 0.0


def strategy_trajectory_to_record(
    trajectory: Dict[str, Any],
    *,
    group_id: str,
    group_index: int,
    group_size: int,
    strategy_name: str,
    strategy_instruction: str,
    sample_seed: int,
    sampling_temperature: float,
    sampling_top_p: float,
    max_steps: int = 3,
    strategy_version: str = "v1",
) -> Dict[str, Any]:
    record = trajectory_to_rollout_record(
        trajectory,
        group_id=group_id,
        group_index=group_index,
        group_size=group_size,
        sample_seed=sample_seed,
        sampling_temperature=sampling_temperature,
        sampling_top_p=sampling_top_p,
        max_steps=max_steps,
    )
    record.update(
        {
            "strategy_name": strategy_name,
            "strategy_instruction": strategy_instruction,
            "strategy_version": strategy_version,
            "is_strategy_controlled": True,
            "is_real_multisample": True,
        }
    )
    record["extra_info"]["strategy_name"] = strategy_name
    record["extra_info"]["strategy_version"] = strategy_version
    return record


def compute_strategy_group_metrics(
    records: List[Dict[str, Any]],
    strategies: List[str],
    eps: float = 1e-6,
) -> Dict[str, Any]:
    base_metrics = compute_group_metrics(records, eps=eps)

    strategy_rewards: Dict[str, float] = {}
    strategy_ndcg: Dict[str, float] = {}
    for record in records:
        name = record["strategy_name"]
        strategy_rewards[name] = float(record["reward"])
        strategy_ndcg[name] = float(record["metrics"]["ndcg_at_10"])

    rewards_by_strategy = [strategy_rewards.get(s, 0.0) for s in strategies if s in strategy_rewards]
    reward_values = list(strategy_rewards.values())
    reward_std = base_metrics["reward_std"]

    best_strategy_by_reward = max(strategy_rewards, key=strategy_rewards.get) if strategy_rewards else None
    worst_strategy_by_reward = min(strategy_rewards, key=strategy_rewards.get) if strategy_rewards else None

    final_queries = [
        str(r["extra_info"].get("final_query") or r["trajectory"].get("final_query") or "") for r in records
    ]
    fingerprints = [_trajectory_fingerprint(r["trajectory"]) for r in records]

    unique_strategy_final_queries = {r["strategy_name"]: final_queries[i] for i, r in enumerate(records)}
    unique_strategy_final_query_count = len(set(unique_strategy_final_queries.values()))

    return {
        **base_metrics,
        "strategy_names": strategies,
        "strategy_rewards": strategy_rewards,
        "strategy_ndcg_at_10": strategy_ndcg,
        "best_strategy_by_reward": best_strategy_by_reward,
        "worst_strategy_by_reward": worst_strategy_by_reward,
        "unique_strategy_final_query_count": unique_strategy_final_query_count,
        "strategy_reward_std": reward_std,
        "avg_pairwise_final_query_jaccard": _pairwise_average(final_queries, token_jaccard),
        "avg_pairwise_trajectory_jaccard": _pairwise_average(
            fingerprints,
            lambda a, b: token_jaccard(a.replace("|", " "), b.replace("|", " ")),
        ),
    }


class StrategyEpisodeRunner:
    """
    Phase 1.18d strategy-controlled rollout runner.

    For each base query, run one episode per predefined search strategy.
    It does NOT train and does NOT change reward.
    """

    def __init__(
        self,
        env_factory: Callable[[], Any],
        policy: Any,
        strategies: List[str] | None = None,
        max_steps: int = 3,
        topk: int = 20,
        base_seed: int = 42,
        sampling_temperature: float = 0.7,
        sampling_top_p: float = 0.95,
        strategy_getter: Callable[[str], Dict[str, str]] | None = None,
        strategy_version: str = "v1",
    ):
        self.env_factory = env_factory
        self.policy = policy
        self.strategies = list(strategies or DEFAULT_STRATEGY_ORDER)
        validate_strategies(self.strategies)
        self.strategy_getter = strategy_getter or get_strategy
        self.strategy_version = strategy_version
        self.group_size = len(self.strategies)
        self.max_steps = max_steps
        self.topk = topk
        self.base_seed = base_seed
        self.sampling_temperature = sampling_temperature
        self.sampling_top_p = sampling_top_p

    def _member_seed(self, group_id: str, group_index: int) -> int:
        gid_hash = int(hashlib.md5(group_id.encode("utf-8")).hexdigest()[:8], 16)
        return self.base_seed + group_index * 1000 + (gid_hash % 1000)

    def run_strategy_episode(
        self,
        sample: Dict[str, Any],
        strategy_name: str,
        group_index: int,
    ) -> Dict[str, Any]:
        group_id = sample.get("qid") or sample.get("sample_id") or "unknown"
        strategy = self.strategy_getter(strategy_name)
        sample_seed = self._member_seed(group_id, group_index)
        episode_sample = {**sample, "qid": f"{group_id}#{strategy_name}"}

        env = self.env_factory()
        trajectory = run_finish_aware_episode(
            env,
            self.policy,
            episode_sample,
            sample_seed=sample_seed,
            strategy_name=strategy_name,
            strategy_instruction=strategy["instruction"],
        )
        trajectory["strategy_name"] = strategy_name

        return strategy_trajectory_to_record(
            trajectory,
            group_id=group_id,
            group_index=group_index,
            group_size=self.group_size,
            strategy_name=strategy_name,
            strategy_instruction=strategy["instruction"],
            sample_seed=sample_seed,
            sampling_temperature=self.sampling_temperature,
            sampling_top_p=self.sampling_top_p,
            max_steps=self.max_steps,
            strategy_version=self.strategy_version,
        )

    def run_group(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        group_id = sample.get("qid") or sample.get("sample_id") or "unknown"
        original_query = sample.get("query") or sample.get("user_query") or ""
        records: List[Dict[str, Any]] = []

        for group_index, strategy_name in enumerate(self.strategies):
            records.append(self.run_strategy_episode(sample, strategy_name, group_index))

        group_metrics = compute_strategy_group_metrics(records, self.strategies)
        return {
            "group_id": group_id,
            "original_query": original_query,
            "group_size": self.group_size,
            "strategies": self.strategies,
            "records": records,
            "group_metrics": group_metrics,
        }

    def run_groups(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.run_group(sample) for sample in samples]
