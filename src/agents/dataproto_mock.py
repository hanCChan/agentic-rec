"""
Phase 1.10 DataProto Mock.

This module does not launch GRPO training or VERL trainer.
It maps Phase 1.9 training fields into a DataProto-like payload for dry-run validation.

Use real verl.protocol.DataProto in Phase 1.11 after compatibility check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch

from src.agents.verl_training_field_builder import MOCK_FIELDS_WARNING

DRY_RUN_WARNING = "This is a dry-run DataProtoMock. It must not be used for optimizer.step."

ACTOR_REQUIRED_KEYS = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "responses",
    "response_attention_mask",
)


@dataclass
class DataProtoMock:
    """DataProto-like container: tensor batch + non-tensor batch + meta_info."""

    batch: Dict[str, torch.Tensor]
    non_tensor_batch: Dict[str, Any] = field(default_factory=dict)
    meta_info: Dict[str, Any] = field(default_factory=dict)

    def batch_size(self) -> int:
        return int(self.batch["input_ids"].shape[0])

    def keys(self) -> List[str]:
        return list(self.batch.keys())

    @classmethod
    def from_fields(cls, fields: Dict[str, Any]) -> "DataProtoMock":
        """Build DataProtoMock from Phase 1.9 training fields."""
        batch = {
            "input_ids": fields["input_ids"],
            "attention_mask": fields["attention_mask"],
            "position_ids": fields["position_ids"],
            "prompts": fields["prompts"],
            "responses": fields["responses"],
            "response_attention_mask": fields["response_attention_mask"],
            "response_mask": fields["response_mask"],
            "token_level_rewards": fields["token_level_rewards"],
            "sequence_rewards": fields["sequence_rewards"],
        }
        non_tensor_batch = {
            "sample_ids": fields["sample_ids"],
            "metrics": fields["metrics"],
            "extra_info": fields["extra_info"],
        }
        meta_info = {
            "phase": "1.10",
            "source": "agentic-rec phase19 training fields mock",
            "mock_old_log_probs_shape": list(fields["mock_old_log_probs"].shape),
            "mock_advantages_shape": list(fields["mock_advantages"].shape),
            "mock_returns_shape": list(fields["mock_returns"].shape),
            "warning": MOCK_FIELDS_WARNING,
            "dry_run_warning": DRY_RUN_WARNING,
        }
        return cls(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    def validate(self) -> Dict[str, Any]:
        """Validate batch shapes, dtypes, and non-tensor alignment."""
        checks: Dict[str, Any] = {"passed": True, "errors": []}

        def fail(msg: str) -> None:
            checks["passed"] = False
            checks["errors"].append(msg)

        try:
            batch_size = self.batch["input_ids"].shape[0]

            assert self.batch["input_ids"].ndim == 2
            assert self.batch["attention_mask"].shape == self.batch["input_ids"].shape
            assert self.batch["position_ids"].shape == self.batch["input_ids"].shape
            assert self.batch["response_mask"].shape == self.batch["input_ids"].shape

            assert self.batch["responses"].ndim == 2
            assert self.batch["response_attention_mask"].shape == self.batch["responses"].shape
            assert self.batch["token_level_rewards"].shape == self.batch["responses"].shape

            assert self.batch["sequence_rewards"].shape == (batch_size,)

            assert len(self.non_tensor_batch["sample_ids"]) == batch_size
            assert len(self.non_tensor_batch["metrics"]) == batch_size
            assert len(self.non_tensor_batch["extra_info"]) == batch_size

            for key, tensor in self.batch.items():
                if tensor.shape[0] != batch_size:
                    fail(f"batch[{key}] batch dim {tensor.shape[0]} != {batch_size}")

            if self.batch["input_ids"].dtype != torch.long:
                fail(f"input_ids dtype must be long, got {self.batch['input_ids'].dtype}")
            if self.batch["prompts"].dtype != torch.long:
                fail(f"prompts dtype must be long, got {self.batch['prompts'].dtype}")
            if self.batch["responses"].dtype != torch.long:
                fail(f"responses dtype must be long, got {self.batch['responses'].dtype}")

            if self.batch["sequence_rewards"].dtype != torch.float32:
                fail(
                    f"sequence_rewards dtype must be float32, got {self.batch['sequence_rewards'].dtype}"
                )
            if self.batch["token_level_rewards"].dtype != torch.float32:
                fail(
                    f"token_level_rewards dtype must be float32, "
                    f"got {self.batch['token_level_rewards'].dtype}"
                )

            nonzero_rows = (self.batch["token_level_rewards"].abs().sum(dim=1) > 0).sum().item()
            if nonzero_rows != batch_size:
                fail(f"token_level_rewards nonzero rows {nonzero_rows} != batch_size {batch_size}")

            if self.batch["response_attention_mask"].sum(dim=1).min().item() <= 0:
                fail("response_attention_mask must have at least one valid token per row")

        except AssertionError as exc:
            fail(str(exc))

        return checks


def check_actor_inputs(data_proto: DataProtoMock) -> Dict[str, Any]:
    """Check actor forward input fields without calling actor.forward."""
    missing_keys = [key for key in ACTOR_REQUIRED_KEYS if key not in data_proto.batch]
    passed = len(missing_keys) == 0

    result: Dict[str, Any] = {
        "actor_input_check_passed": passed,
        "required_keys_present": [k for k in ACTOR_REQUIRED_KEYS if k in data_proto.batch],
        "missing_keys": missing_keys,
    }

    if passed:
        batch = data_proto.batch
        result["input_ids_shape"] = list(batch["input_ids"].shape)
        result["responses_shape"] = list(batch["responses"].shape)
        result["attention_mask_shape"] = list(batch["attention_mask"].shape)
        result["position_ids_shape"] = list(batch["position_ids"].shape)
        result["response_attention_mask_shape"] = list(batch["response_attention_mask"].shape)

    return result


def build_dataproto_shapes(data_proto: DataProtoMock) -> Dict[str, Any]:
    """Serialize tensor shapes/dtypes for logging."""
    shapes: Dict[str, Any] = {"batch": {}, "non_tensor_batch": {}, "meta_info": data_proto.meta_info}
    for key, tensor in data_proto.batch.items():
        shapes["batch"][key] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
    for key, value in data_proto.non_tensor_batch.items():
        if isinstance(value, list):
            shapes["non_tensor_batch"][key] = {"length": len(value), "type": "list"}
        else:
            shapes["non_tensor_batch"][key] = {"type": type(value).__name__}
    return shapes
