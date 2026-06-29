"""Process + outcome reward aggregation for CommerceAgentEnv."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RewardConfig:
    ndcg_k: int = 10
    lambda_process: float = 0.5
    search_cost: float = 0.05
    repeat_penalty: float = 0.10
    invalid_penalty: float = 0.20
    no_final_penalty: float = 0.10


@dataclass
class EpisodeRewardBreakdown:
    final_reward: float = 0.0
    process_reward_sum: float = 0.0
    penalties: Dict[str, float] = field(default_factory=dict)
    total_penalty: float = 0.0
    total_reward: float = 0.0


def compute_episode_reward(
    *,
    final_ndcg: float,
    delta_ndcg_list: List[float],
    num_search_calls: int,
    num_invalid: int,
    num_repeated: int,
    has_final_answer: bool,
    config: Optional[RewardConfig] = None,
) -> EpisodeRewardBreakdown:
    cfg = config or RewardConfig()

    penalties: Dict[str, float] = {}
    if num_search_calls > 0:
        penalties["search_cost"] = cfg.search_cost * num_search_calls
    if num_repeated > 0:
        penalties["repeated_query"] = cfg.repeat_penalty * num_repeated
    if num_invalid > 0:
        penalties["invalid_action"] = cfg.invalid_penalty * num_invalid
    if not has_final_answer:
        penalties["no_final_answer"] = cfg.no_final_penalty

    total_penalty = sum(penalties.values())
    process_reward_sum = sum(delta_ndcg_list)
    final_reward = final_ndcg
    total_reward = final_reward + cfg.lambda_process * process_reward_sum - total_penalty

    return EpisodeRewardBreakdown(
        final_reward=final_reward,
        process_reward_sum=process_reward_sum,
        penalties=penalties,
        total_penalty=total_penalty,
        total_reward=total_reward,
    )
