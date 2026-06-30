"""
Phase 1.8 VERL Batch Builder (mock).

This module does not launch GRPO training.
It converts Phase 1.7 VERL-like rollout records into tokenized training-like batches:
input_ids, attention_mask, response_mask, rewards.

The next phase will align these fields with Rec-R1/VERL trainer inputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoTokenizer


def _truncate_left(ids: List[int], max_len: int) -> Tuple[List[int], bool]:
    """Keep the leftmost tokens; truncate from the right if too long."""
    if len(ids) <= max_len:
        return ids, False
    return ids[:max_len], True


class VerlBatchBuilder:
    """Build VERL-like mock batches from rollout records via HuggingFace tokenization."""

    def __init__(
        self,
        tokenizer_path: str,
        max_prompt_length: int = 1024,
        max_response_length: int = 2048,
        max_total_length: int = 3072,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.max_total_length = max_total_length

    def _encode_text(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def build_one(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize one rollout record into tensors and per-record metadata."""
        prompt = record.get("prompt", "")
        response = record.get("response", "")
        reward = float(record.get("reward", 0.0))

        prompt_ids, prompt_truncated = _truncate_left(
            self._encode_text(prompt), self.max_prompt_length
        )
        response_ids, response_truncated = _truncate_left(
            self._encode_text(response), self.max_response_length
        )

        input_ids = prompt_ids + response_ids
        total_truncated = False
        if len(input_ids) > self.max_total_length:
            input_ids = input_ids[: self.max_total_length]
            total_truncated = True
            if len(input_ids) < len(prompt_ids):
                prompt_len = len(input_ids)
                response_len = 0
            else:
                prompt_len = len(prompt_ids)
                response_len = len(input_ids) - prompt_len
        else:
            prompt_len = len(prompt_ids)
            response_len = len(response_ids)

        if total_truncated and prompt_len == len(prompt_ids) and len(input_ids) < len(prompt_ids) + len(response_ids):
            response_truncated = True

        attention_mask = [1] * len(input_ids)
        response_mask = [0] * prompt_len + [1] * response_len

        token_meta = {
            "prompt_truncated": prompt_truncated,
            "response_truncated": response_truncated,
            "total_truncated": total_truncated,
            "prompt_length": prompt_len,
            "response_length": response_len,
            "total_length": len(input_ids),
        }

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "response_mask": torch.tensor(response_mask, dtype=torch.long),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "prompt_length": prompt_len,
            "response_length": response_len,
            "sample_id": record.get("sample_id", ""),
            "metrics": record.get("metrics", {}),
            "extra_info": record.get("extra_info", {}),
            "token_meta": token_meta,
        }

    def build_batch(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pad and stack per-record tensors into a batch."""
        items = [self.build_one(record) for record in records]
        if not items:
            raise ValueError("records must not be empty")

        max_seq_len = min(max(item["input_ids"].shape[0] for item in items), self.max_total_length)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        batch_input_ids: List[torch.Tensor] = []
        batch_attention: List[torch.Tensor] = []
        batch_response_mask: List[torch.Tensor] = []
        rewards: List[torch.Tensor] = []
        prompt_lengths: List[int] = []
        response_lengths: List[int] = []
        sample_ids: List[str] = []
        metrics: List[Dict[str, Any]] = []
        extra_info: List[Dict[str, Any]] = []
        token_meta: List[Dict[str, Any]] = []

        for item in items:
            seq_len = item["input_ids"].shape[0]
            pad_len = max_seq_len - seq_len

            input_ids = torch.cat(
                [item["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]
            )
            attention_mask = torch.cat(
                [item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]
            )
            response_mask = torch.cat(
                [item["response_mask"], torch.zeros(pad_len, dtype=torch.long)]
            )

            batch_input_ids.append(input_ids)
            batch_attention.append(attention_mask)
            batch_response_mask.append(response_mask)
            rewards.append(item["reward"])
            prompt_lengths.append(item["prompt_length"])
            response_lengths.append(item["response_length"])
            sample_ids.append(item["sample_id"])
            metrics.append(item["metrics"])
            extra_info.append(item["extra_info"])
            token_meta.append(item["token_meta"])

        return {
            "input_ids": torch.stack(batch_input_ids, dim=0),
            "attention_mask": torch.stack(batch_attention, dim=0),
            "response_mask": torch.stack(batch_response_mask, dim=0),
            "rewards": torch.stack(rewards, dim=0),
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            "response_lengths": torch.tensor(response_lengths, dtype=torch.long),
            "sample_ids": sample_ids,
            "metrics": metrics,
            "extra_info": extra_info,
            "token_meta": token_meta,
            "seq_len": max_seq_len,
        }


def check_batch_shapes(batch: Dict[str, Any]) -> None:
    """Assert VERL-like mock batch tensor shapes and dtypes."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    response_mask = batch["response_mask"]
    rewards = batch["rewards"]
    sample_ids = batch["sample_ids"]
    prompt_lengths = batch["prompt_lengths"]

    assert input_ids.ndim == 2, f"input_ids.ndim={input_ids.ndim}"
    assert attention_mask.shape == input_ids.shape
    assert response_mask.shape == input_ids.shape
    assert rewards.ndim == 1
    assert rewards.shape[0] == input_ids.shape[0]
    assert len(sample_ids) == input_ids.shape[0]
    assert response_mask.sum(dim=1).min().item() > 0
    assert attention_mask.sum(dim=1).min().item() > 0

    assert rewards.dtype == torch.float32
    assert input_ids.dtype == torch.long
    assert attention_mask.dtype == torch.long
    assert response_mask.dtype == torch.long

    for i in range(input_ids.shape[0]):
        plen = int(prompt_lengths[i].item())
        assert response_mask[i, :plen].sum().item() == 0, "response_mask must not cover prompt tokens"
        valid_len = int(attention_mask[i].sum().item())
        if valid_len < input_ids.shape[1]:
            assert response_mask[i, valid_len:].sum().item() == 0, "padding response_mask must be 0"
