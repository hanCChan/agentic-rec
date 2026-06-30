"""
Phase 1.10 Commerce Reward Function (dry-run).

Reads rewards from DataProtoMock batch fields populated during rollout.
Does not re-run env/BM25 or recompute NDCG.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from src.agents.dataproto_mock import DataProtoMock


class CommerceRewardFn:
    """Dry-run reward function that reads precomputed sequence/token rewards."""

    def __init__(self, reward_key: str = "sequence_rewards"):
        self.reward_key = reward_key

    def __call__(self, data_proto: DataProtoMock) -> Dict[str, Any]:
        """
        Return token/sequence rewards and summary metrics from DataProtoMock batch.

        Does not recompute NDCG; rollout stage already computed episode rewards.
        """
        if self.reward_key not in data_proto.batch:
            raise KeyError(f"missing reward key in batch: {self.reward_key}")

        sequence_rewards = data_proto.batch[self.reward_key].float()
        token_level_rewards = data_proto.batch["token_level_rewards"].float()
        batch_size = sequence_rewards.shape[0]

        if sequence_rewards.ndim != 1 or sequence_rewards.shape[0] != batch_size:
            raise ValueError(f"sequence_rewards shape invalid: {sequence_rewards.shape}")
        if token_level_rewards.shape != data_proto.batch["responses"].shape:
            raise ValueError(
                f"token_level_rewards shape {token_level_rewards.shape} != "
                f"responses shape {data_proto.batch['responses'].shape}"
            )

        nonzero = int((token_level_rewards != 0).sum().item())
        metrics = {
            "reward_mean": float(sequence_rewards.mean().item()),
            "reward_min": float(sequence_rewards.min().item()),
            "reward_max": float(sequence_rewards.max().item()),
            "num_nonzero_token_rewards": nonzero,
            "batch_size": batch_size,
        }

        return {
            "token_level_rewards": token_level_rewards,
            "sequence_rewards": sequence_rewards,
            "metrics": metrics,
            "check_passed": nonzero == batch_size,
        }
