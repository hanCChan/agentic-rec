"""
Phase 1.13 Real Actor LogProb Dry-Run.

Loads a HuggingFace causal LM and computes response-token log probabilities under
torch.no_grad(). Does NOT train, does NOT connect to VERL trainer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from src.agents.actor_logprob_mock import _get_batch

DRYRUN_WARNING = (
    "Real model forward was run under torch.no_grad(); "
    "no training or optimizer.step was performed."
)

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _extract_lengths(data_proto: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """Read prompt/response lengths from meta_info."""
    meta = getattr(data_proto, "meta_info", None) or {}
    if "prompt_lengths" not in meta or "response_lengths" not in meta:
        raise KeyError(
            "prompt_lengths/response_lengths missing from data_proto.meta_info; "
            "rebuild DataProto from Phase 1.9 fields"
        )
    prompt_lengths = torch.tensor(meta["prompt_lengths"], dtype=torch.long)
    response_lengths = torch.tensor(meta["response_lengths"], dtype=torch.long)
    return prompt_lengths, response_lengths


class ActorLogProbDryRun:
    """
    Phase 1.13 real actor logprob dry-run.

    Loads HuggingFace AutoModelForCausalLM and computes response logprobs under
    torch.no_grad(). No training, no VERL trainer, no vLLM.
    """

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        max_batch_size: int = 2,
    ):
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.max_batch_size = max_batch_size
        self.model: Optional[AutoModelForCausalLM] = None

    def load_model(self) -> None:
        if self.dtype not in _DTYPE_MAP:
            raise ValueError(f"unsupported dtype: {self.dtype}")
        torch_dtype = _DTYPE_MAP[self.dtype]

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        self.model.eval()

    @property
    def model_device(self) -> torch.device:
        if self.model is None:
            raise RuntimeError("model not loaded; call load_model() first")
        return next(self.model.parameters()).device

    def compute_response_log_probs(self, data_proto: Any) -> Dict[str, Any]:
        """Forward pass and slice response-token logprobs from causal LM logits."""
        if self.model is None:
            raise RuntimeError("model not loaded; call load_model() first")

        batch = _get_batch(data_proto)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        responses = batch["responses"]
        response_attention_mask = batch["response_attention_mask"]
        prompt_lengths, response_lengths = _extract_lengths(data_proto)

        batch_size = input_ids.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"batch_size {batch_size} exceeds max_batch_size {self.max_batch_size}"
            )

        for i in range(batch_size):
            plen = int(prompt_lengths[i].item())
            rlen = int(response_lengths[i].item())
            if plen <= 0:
                raise ValueError(f"sample {i}: prompt_len must be > 0, got {plen}")
            if rlen <= 0:
                raise ValueError(f"sample {i}: response_len must be > 0, got {rlen}")

        device = self.model_device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            log_probs_all = F.log_softmax(shift_logits.float(), dim=-1)
            token_log_probs = torch.gather(
                log_probs_all,
                dim=-1,
                index=shift_labels.unsqueeze(-1),
            ).squeeze(-1)

        response_len = responses.shape[1]
        real_old_log_probs = torch.zeros(
            (batch_size, response_len), dtype=torch.float32, device="cpu"
        )
        real_entropys = torch.zeros(
            (batch_size, response_len), dtype=torch.float32, device="cpu"
        )
        debug_slices: List[Dict[str, Any]] = []

        for i in range(batch_size):
            plen = int(prompt_lengths[i].item())
            rlen = int(response_lengths[i].item())
            slice_start = plen - 1
            slice_end = plen - 1 + rlen
            real_old_log_probs[i, :rlen] = token_log_probs[i, slice_start:slice_end].cpu()
            debug_slices.append(
                {
                    "sample_index": i,
                    "prompt_len": plen,
                    "response_len": rlen,
                    "slice_start": slice_start,
                    "slice_end": slice_end,
                    "first_5_response_token_ids": responses[i, : min(5, rlen)].tolist(),
                }
            )

        return {
            "real_old_log_probs": real_old_log_probs,
            "real_entropys": real_entropys,
            "is_real": True,
            "warning": "Dry-run only. Do not use for training yet.",
            "debug_slices": debug_slices,
        }

    def check_real_logprob_output(
        self,
        data_proto: Any,
        output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate real logprob shapes, finiteness, and padding zeros."""
        batch = _get_batch(data_proto)
        responses = batch["responses"]
        response_attention_mask = batch["response_attention_mask"]

        real_old_log_probs = output["real_old_log_probs"]
        real_entropys = output["real_entropys"]

        assert real_old_log_probs.shape == responses.shape
        assert real_entropys.shape == responses.shape
        assert real_old_log_probs.dtype == torch.float32

        valid_mask = response_attention_mask.bool()
        finite_valid = torch.isfinite(real_old_log_probs[valid_mask]).all().item()
        padding_zero = (real_old_log_probs[~valid_mask] == 0).all().item()

        valid_logprobs = real_old_log_probs[valid_mask]
        stats = {
            "mean_valid_logprob": float(valid_logprobs.mean().item()) if valid_logprobs.numel() else 0.0,
            "min_valid_logprob": float(valid_logprobs.min().item()) if valid_logprobs.numel() else 0.0,
            "max_valid_logprob": float(valid_logprobs.max().item()) if valid_logprobs.numel() else 0.0,
        }

        return {
            "real_logprob_check_passed": bool(finite_valid and padding_zero),
            "real_old_log_probs_shape": list(real_old_log_probs.shape),
            "responses_shape": list(responses.shape),
            "finite_valid_logprobs": bool(finite_valid),
            "padding_logprobs_zero": bool(padding_zero),
            **stats,
        }
