"""
Phase 1.12 Actor LogProb Interface Mock.

This module does NOT call actor.forward or compute real logprobs.
It inspects DataProto-like payloads for fields required by verl actor.compute_log_prob,
builds an actor-logprob-ready request dict, and emits mock logprob tensors for shape check.

Reference (Rec-R1/verl dp_actor.compute_log_prob select_keys):
    responses, input_ids, attention_mask, position_ids
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import torch

MOCK_LOGPROB_WARNING = "Mock old_log_probs/entropys only. Do not use for training."

VERL_COMPUTE_LOG_PROB_KEYS = [
    "responses",
    "input_ids",
    "attention_mask",
    "position_ids",
]


def _get_batch(data_proto: Any) -> Dict[str, torch.Tensor]:
    """Extract batch dict from DataProtoMock or verl.protocol.DataProto."""
    batch = data_proto.batch
    if hasattr(batch, "keys"):
        return {key: batch[key] for key in batch.keys()}
    return batch


class ActorLogProbInterfaceMock:
    """
    Phase 1.12 actor logprob interface mock.

    Does NOT call actor.forward.
    Only checks required fields and creates mock logprob tensors.
    """

    REQUIRED_ACTOR_KEYS = [
        "input_ids",
        "attention_mask",
        "position_ids",
        "responses",
        "response_attention_mask",
    ]

    def __init__(self):
        self.verl_compute_log_prob_keys = list(VERL_COMPUTE_LOG_PROB_KEYS)

    def inspect_required_fields(self, data_proto: Any) -> Dict[str, Any]:
        """Check actor logprob input fields on DataProtoMock or real DataProto."""
        batch = _get_batch(data_proto)
        missing_keys = [key for key in self.REQUIRED_ACTOR_KEYS if key not in batch]
        passed = len(missing_keys) == 0

        result: Dict[str, Any] = {
            "actor_input_check_passed": passed,
            "required_keys": list(self.REQUIRED_ACTOR_KEYS),
            "verl_compute_log_prob_keys": self.verl_compute_log_prob_keys,
            "missing_keys": missing_keys,
        }

        if passed:
            result["input_ids_shape"] = list(batch["input_ids"].shape)
            result["responses_shape"] = list(batch["responses"].shape)
            result["response_attention_mask_shape"] = list(batch["response_attention_mask"].shape)
            result["attention_mask_shape"] = list(batch["attention_mask"].shape)
            result["position_ids_shape"] = list(batch["position_ids"].shape)

        return result

    def build_actor_logprob_request(self, data_proto: Any) -> Dict[str, Any]:
        """Build actor-logprob-ready request dict (tensor references only)."""
        batch = _get_batch(data_proto)
        inspect_result = self.inspect_required_fields(data_proto)
        if not inspect_result["actor_input_check_passed"]:
            raise KeyError(f"missing actor keys: {inspect_result['missing_keys']}")

        request: Dict[str, Any] = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "position_ids": batch["position_ids"],
            "responses": batch["responses"],
            "response_attention_mask": batch["response_attention_mask"],
            "response_mask": batch.get("response_mask"),
        }
        return request

    def mock_compute_log_prob(self, actor_request: Dict[str, Any]) -> Dict[str, Any]:
        """Return zero mock logprobs with response-token shape. Not for training."""
        responses = actor_request["responses"]
        batch_size, response_len = responses.shape

        mock_old_log_probs = torch.zeros((batch_size, response_len), dtype=torch.float32)
        mock_entropys = torch.zeros((batch_size, response_len), dtype=torch.float32)

        return {
            "old_log_probs": mock_old_log_probs,
            "entropys": mock_entropys,
            "is_mock": True,
            "warning": MOCK_LOGPROB_WARNING,
        }

    def check_logprob_output(
        self,
        data_proto: Any,
        logprob_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate mock logprob output shapes/dtypes against responses."""
        batch = _get_batch(data_proto)
        responses = batch["responses"]
        response_attention_mask = batch["response_attention_mask"]

        old_log_probs = logprob_output["old_log_probs"]
        entropys = logprob_output["entropys"]

        assert old_log_probs.shape == responses.shape
        assert entropys.shape == responses.shape
        assert old_log_probs.dtype == torch.float32
        assert entropys.dtype == torch.float32
        assert response_attention_mask.sum(dim=1).min().item() > 0

        return {
            "logprob_shape_check_passed": True,
            "old_log_probs_shape": list(old_log_probs.shape),
            "entropys_shape": list(entropys.shape),
            "responses_shape": list(responses.shape),
            "is_mock": bool(logprob_output.get("is_mock", True)),
        }


def tensor_shape_report(tensors: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize tensor shapes/dtypes for JSON output."""
    report: Dict[str, Any] = {}
    for key, value in tensors.items():
        if value is None:
            report[key] = None
        elif isinstance(value, torch.Tensor):
            report[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
        else:
            report[key] = {"type": type(value).__name__}
    return report
