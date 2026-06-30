"""
Phase 1.9 VERL Training Field Builder (mock).

This module does not launch GRPO training.
It extends Phase 1.8 tokenized batches with VERL-like training fields:
position_ids, prompts, responses, token_level_rewards, and mock logprob/advantage placeholders.

mock_old_log_probs / mock_advantages / mock_returns are shape placeholders only
and must not be used for training.
"""

from __future__ import annotations

from typing import Any, Dict

import torch


MOCK_FIELDS_WARNING = (
    "mock_old_log_probs/mock_advantages/mock_returns are placeholders "
    "and must not be used for training"
)


class VerlTrainingFieldBuilder:
    """Build VERL-like training field mock batch from Phase 1.8 batch."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def build_training_fields(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:
            Phase 1.8 batch from VerlBatchBuilder.

        Output:
            VERL-like training field mock batch.
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        response_mask = batch["response_mask"]
        sequence_rewards = batch["rewards"].float()
        prompt_lengths = batch["prompt_lengths"]
        response_lengths = batch["response_lengths"]

        batch_size, _ = input_ids.shape
        prompt_width = int(prompt_lengths.max().item())
        response_width = int(response_lengths.max().item())

        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids = position_ids.clamp(min=0)
        position_ids = position_ids * attention_mask

        prompts = torch.full(
            (batch_size, prompt_width), self.pad_token_id, dtype=torch.long
        )
        responses = torch.full(
            (batch_size, response_width), self.pad_token_id, dtype=torch.long
        )
        response_attention_mask = torch.zeros(
            (batch_size, response_width), dtype=torch.long
        )
        token_level_rewards = torch.zeros((batch_size, response_width), dtype=torch.float32)
        mock_old_log_probs = torch.zeros((batch_size, response_width), dtype=torch.float32)
        mock_advantages = torch.zeros((batch_size, response_width), dtype=torch.float32)
        mock_returns = torch.zeros((batch_size, response_width), dtype=torch.float32)

        for i in range(batch_size):
            plen = int(prompt_lengths[i].item())
            rlen = int(response_lengths[i].item())

            prompts[i, :plen] = input_ids[i, :plen]
            if rlen > 0:
                responses[i, :rlen] = input_ids[i, plen : plen + rlen]
                response_attention_mask[i, :rlen] = 1
                token_level_rewards[i, rlen - 1] = sequence_rewards[i]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompts": prompts,
            "responses": responses,
            "response_attention_mask": response_attention_mask,
            "response_mask": response_mask,
            "sequence_rewards": sequence_rewards,
            "token_level_rewards": token_level_rewards,
            "mock_old_log_probs": mock_old_log_probs,
            "mock_advantages": mock_advantages,
            "mock_returns": mock_returns,
            "prompt_lengths": prompt_lengths,
            "response_lengths": response_lengths,
            "sample_ids": batch["sample_ids"],
            "metrics": batch["metrics"],
            "extra_info": batch["extra_info"],
            "prompt_width": prompt_width,
            "response_width": response_width,
            "seq_len": batch.get("seq_len", input_ids.shape[1]),
        }


def check_training_fields(fields: Dict[str, Any]) -> None:
    """Assert VERL-like training field mock shapes, dtypes, and mask consistency."""
    input_ids = fields["input_ids"]
    attention_mask = fields["attention_mask"]
    position_ids = fields["position_ids"]
    prompts = fields["prompts"]
    responses = fields["responses"]
    response_attention_mask = fields["response_attention_mask"]
    response_mask = fields["response_mask"]
    sequence_rewards = fields["sequence_rewards"]
    token_level_rewards = fields["token_level_rewards"]
    mock_old_log_probs = fields["mock_old_log_probs"]
    mock_advantages = fields["mock_advantages"]
    mock_returns = fields["mock_returns"]
    prompt_lengths = fields["prompt_lengths"]
    response_lengths = fields["response_lengths"]

    assert input_ids.ndim == 2
    assert attention_mask.shape == input_ids.shape
    assert position_ids.shape == input_ids.shape
    assert response_mask.shape == input_ids.shape

    assert prompts.ndim == 2
    assert responses.ndim == 2
    assert response_attention_mask.shape == responses.shape

    assert sequence_rewards.ndim == 1
    assert sequence_rewards.shape[0] == input_ids.shape[0]

    assert token_level_rewards.shape == responses.shape
    assert mock_old_log_probs.shape == responses.shape
    assert mock_advantages.shape == responses.shape
    assert mock_returns.shape == responses.shape

    assert response_attention_mask.sum(dim=1).min().item() > 0
    assert token_level_rewards.abs().sum(dim=1).numel() == input_ids.shape[0]

    assert sequence_rewards.dtype == torch.float32
    assert token_level_rewards.dtype == torch.float32
    assert mock_old_log_probs.dtype == torch.float32
    assert mock_advantages.dtype == torch.float32
    assert mock_returns.dtype == torch.float32
    assert input_ids.dtype == torch.long
    assert prompts.dtype == torch.long
    assert responses.dtype == torch.long

    batch_size = input_ids.shape[0]
    for i in range(batch_size):
        plen = int(prompt_lengths[i].item())
        rlen = int(response_lengths[i].item())
        valid_len = int(attention_mask[i].sum().item())

        assert plen + rlen <= valid_len, "prompt + response length exceeds valid attention tokens"

        nonzero = token_level_rewards[i].nonzero(as_tuple=False).squeeze(-1)
        if nonzero.numel() > 0:
            assert (response_attention_mask[i, nonzero] == 1).all().item(), (
                "token_level_rewards nonzero positions must be inside response_attention_mask"
            )

        pad_positions = attention_mask[i] == 0
        if pad_positions.any():
            assert (position_ids[i, pad_positions] == 0).all().item(), (
                "position_ids at padding positions must be 0"
            )

        assert response_mask[i, :plen].sum().item() == 0
        assert response_mask[i, plen : plen + rlen].sum().item() == rlen
