"""
Phase 1.15 GRPO Advantage Mock / Grouped Reward Dry-Run.

Builds synthetic grouped rollout structures and computes GRPO-style group-normalized
advantages. Does NOT train or connect to VERL trainer.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Union

import torch

MOCK_GROUP_WARNING = (
    "Synthetic grouped records are for GRPO advantage dry-run only "
    "and must not be used for real training."
)


def _get_batch(data_proto: Any) -> Dict[str, torch.Tensor]:
    batch = data_proto.batch
    if hasattr(batch, "keys") and not isinstance(batch, dict):
        return {key: batch[key] for key in batch.keys()}
    return batch


def _get_non_tensor_list(data_proto: Any, key: str) -> List[Any]:
    val = data_proto.non_tensor_batch[key]
    if isinstance(val, (list, tuple)):
        return list(val)
    if hasattr(val, "tolist"):
        return val.tolist()
    return list(val)


class GRPOAdvantageMock:
    """
    Phase 1.15 GRPO advantage mock / grouped reward dry-run.

    Builds grouped rollout structures and computes group-normalized advantages.
    Does NOT train, does NOT call actor.forward.
    """

    def __init__(
        self,
        group_size: int = 4,
        normalize_by_std: bool = True,
        eps: float = 1e-6,
        synthetic_reward_jitter: float = 0.02,
        seed: int = 42,
    ):
        self.group_size = group_size
        self.normalize_by_std = normalize_by_std
        self.eps = eps
        self.synthetic_reward_jitter = synthetic_reward_jitter
        self.seed = seed
        self.use_jitter = synthetic_reward_jitter > 0

    def _reward_offsets(self) -> torch.Tensor:
        if not self.use_jitter:
            return torch.zeros(self.group_size, dtype=torch.float32)
        return torch.linspace(
            -self.synthetic_reward_jitter,
            self.synthetic_reward_jitter,
            steps=self.group_size,
            dtype=torch.float32,
        )

    def build_grouped_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expand one-response records into synthetic GRPO groups."""
        offsets = self._reward_offsets()
        grouped: List[Dict[str, Any]] = []

        for record in records:
            base_reward = float(record.get("reward", 0.0))
            group_id = record.get("sample_id", "unknown")

            for group_index in range(self.group_size):
                member = copy.deepcopy(record)
                synthetic_reward = base_reward + float(offsets[group_index].item())
                member["sample_id"] = f"{group_id}#g{group_index}"
                member["reward"] = synthetic_reward
                member["group_id"] = group_id
                member["group_index"] = group_index
                member["group_size"] = self.group_size
                member["is_synthetic_group_member"] = True
                member["base_reward"] = base_reward
                member["synthetic_reward"] = synthetic_reward

                metrics = member.setdefault("metrics", {})
                metrics["total_reward"] = synthetic_reward
                metrics["is_synthetic_group_member"] = True

                extra = member.setdefault("extra_info", {})
                extra["group_id"] = group_id
                extra["group_index"] = group_index
                extra["group_size"] = self.group_size
                extra["base_reward"] = base_reward
                extra["synthetic_reward"] = synthetic_reward

                grouped.append(member)

        return grouped

    def compute_group_advantages(self, data_proto: Any) -> Dict[str, Any]:
        """Compute GRPO-style group-normalized sequence and token-level advantages."""
        batch = _get_batch(data_proto)
        sequence_rewards = batch["sequence_rewards"].float()
        response_attention_mask = batch["response_attention_mask"]
        token_level_rewards = batch["token_level_rewards"].float()

        group_ids = _get_non_tensor_list(data_proto, "group_ids")
        group_indices = _get_non_tensor_list(data_proto, "group_indices")
        batch_size = sequence_rewards.shape[0]

        if len(group_ids) != batch_size:
            raise ValueError(f"group_ids length {len(group_ids)} != batch_size {batch_size}")

        unique_group_ids = sorted(set(group_ids))
        num_groups = len(unique_group_ids)

        sequence_advantages = torch.zeros(batch_size, dtype=torch.float32)
        token_level_advantages = torch.zeros_like(token_level_rewards)
        group_reward_mean = torch.zeros(num_groups, dtype=torch.float32)
        group_reward_std = torch.zeros(num_groups, dtype=torch.float32)
        group_sizes = torch.zeros(num_groups, dtype=torch.long)
        zero_std_group_mask = torch.zeros(num_groups, dtype=torch.bool)

        for gi, gid in enumerate(unique_group_ids):
            member_indices = [i for i, g in enumerate(group_ids) if g == gid]
            idx_tensor = torch.tensor(member_indices, dtype=torch.long)
            group_rewards = sequence_rewards[idx_tensor]
            mean = group_rewards.mean()
            std = group_rewards.std(unbiased=False)

            group_reward_mean[gi] = mean
            group_reward_std[gi] = std
            group_sizes[gi] = len(member_indices)
            zero_std_group_mask[gi] = std < self.eps

            if self.normalize_by_std:
                advantages = (group_rewards - mean) / (std + self.eps)
            else:
                advantages = group_rewards - mean

            sequence_advantages[idx_tensor] = advantages

        for i in range(batch_size):
            rlen = int(response_attention_mask[i].sum().item())
            if rlen > 0:
                token_level_advantages[i, :rlen] = sequence_advantages[i]

        return {
            "sequence_rewards": sequence_rewards,
            "sequence_advantages": sequence_advantages,
            "token_level_advantages": token_level_advantages,
            "group_reward_mean": group_reward_mean,
            "group_reward_std": group_reward_std,
            "group_sizes": group_sizes,
            "zero_std_group_mask": zero_std_group_mask,
            "group_ids": unique_group_ids,
            "group_indices": group_indices,
            "normalize_by_std": self.normalize_by_std,
            "is_mock": True,
        }

    def check_group_advantages(
        self,
        data_proto: Any,
        adv_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate advantage shapes, masks, and group structure."""
        batch = _get_batch(data_proto)
        responses = batch["responses"]
        mask = batch["response_attention_mask"].bool()
        sequence_rewards = adv_output["sequence_rewards"]
        sequence_advantages = adv_output["sequence_advantages"]
        token_level_advantages = adv_output["token_level_advantages"]
        group_reward_std = adv_output["group_reward_std"]
        zero_std_group_mask = adv_output["zero_std_group_mask"]

        group_ids = _get_non_tensor_list(data_proto, "group_ids")
        batch_size = sequence_rewards.shape[0]

        assert sequence_advantages.shape == sequence_rewards.shape
        assert token_level_advantages.shape == responses.shape
        assert torch.isfinite(sequence_advantages).all()
        assert torch.isfinite(token_level_advantages[mask]).all()
        padding_advantages_zero = (token_level_advantages[~mask] == 0).all().item()

        group_member_counts: Dict[str, int] = {}
        for gid in group_ids:
            group_member_counts[gid] = group_member_counts.get(gid, 0) + 1
        every_group_full = all(count == self.group_size for count in group_member_counts.values())

        group_mean_adv_close = True
        unique_group_ids = sorted(set(group_ids))
        for gid in unique_group_ids:
            idx = [i for i, g in enumerate(group_ids) if g == gid]
            idx_tensor = torch.tensor(idx, dtype=torch.long)
            group_mean = sequence_advantages[idx_tensor].mean().item()
            if abs(group_mean) > 0.05:
                group_mean_adv_close = False

        token_broadcast_ok = True
        for i in range(batch_size):
            rlen = int(mask[i].sum().item())
            if rlen > 0:
                valid_adv = token_level_advantages[i, :rlen]
                if not torch.allclose(valid_adv, torch.full_like(valid_adv, sequence_advantages[i].item())):
                    token_broadcast_ok = False

        zero_std_group_count = int(zero_std_group_mask.sum().item())
        num_groups = len(unique_group_ids)
        zero_std_group_rate = zero_std_group_count / num_groups if num_groups else 0.0

        mean_abs_sequence_advantage = float(sequence_advantages.abs().mean().item())
        zero_advantage_token_rate = float(
            (token_level_advantages[mask].abs() < self.eps).float().mean().item()
        ) if mask.any() else 0.0

        passed = all(
            [
                padding_advantages_zero,
                every_group_full,
                group_mean_adv_close,
                token_broadcast_ok,
                torch.isfinite(sequence_advantages).all().item(),
            ]
        )

        return {
            "advantage_check_passed": bool(passed),
            "sequence_advantages_shape": list(sequence_advantages.shape),
            "token_level_advantages_shape": list(token_level_advantages.shape),
            "sequence_rewards_shape": list(sequence_rewards.shape),
            "num_groups": num_groups,
            "group_size": self.group_size,
            "zero_std_group_count": zero_std_group_count,
            "zero_std_group_rate": zero_std_group_rate,
            "mean_group_reward_std": float(group_reward_std.mean().item()),
            "min_group_reward_std": float(group_reward_std.min().item()) if num_groups else 0.0,
            "max_group_reward_std": float(group_reward_std.max().item()) if num_groups else 0.0,
            "mean_abs_sequence_advantage": mean_abs_sequence_advantage,
            "zero_advantage_token_rate": zero_advantage_token_rate,
            "padding_advantages_zero": bool(padding_advantages_zero),
            "group_mean_advantage_close_to_zero": bool(group_mean_adv_close),
            "every_group_has_group_size_members": bool(every_group_full),
            "token_broadcast_ok": bool(token_broadcast_ok),
        }
