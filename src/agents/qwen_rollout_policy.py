"""Qwen2.5 rollout policy via vLLM for CommerceAgentEnv (Phase 1.5)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.action_parser import ActionParseError, parse_action
from src.agents.prompts import build_rollout_prompt


def _looks_like_markdown_wrapped(text: str) -> bool:
    return "```" in text


def _count_json_objects(text: str) -> int:
    return len(re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL))


class QwenRolloutPolicy:
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 256,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.45,
        cuda_device: int | None = None,
    ):
        try:
            import torch
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("vLLM is required for QwenRolloutPolicy") from exc

        if cuda_device is not None and torch.cuda.is_available():
            torch.cuda.set_device(cuda_device)

        self.model_path = str(model_path)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        self._sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=1,
        )

        self.llm = LLM(
            model=self.model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )

        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        except Exception:
            self.tokenizer = None

    def _format_prompt(self, filled_prompt: str) -> str:
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": filled_prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return filled_prompt

    def count_output_tokens(self, raw_output: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(raw_output, add_special_tokens=False))
        return len(raw_output.split())

    def act(
        self,
        user_query: str,
        search_history: List[Dict[str, Any]],
        observation: str,
        max_steps: int,
        current_step: int = 1,
        remaining_steps: int | None = None,
        best_query_by_ndcg: str | None = None,
        best_ndcg_at_10: float | None = None,
        seed: int | None = None,
        strategy_name: str | None = None,
        strategy_instruction: str | None = None,
    ) -> Dict[str, Any]:
        filled = build_rollout_prompt(
            user_query=user_query,
            search_history=search_history,
            observation=observation,
            max_steps=max_steps,
            current_step=current_step,
            remaining_steps=remaining_steps,
            best_query_by_ndcg=best_query_by_ndcg,
            best_ndcg_at_10=best_ndcg_at_10,
            strategy_name=strategy_name,
            strategy_instruction=strategy_instruction,
        )
        prompt = self._format_prompt(filled)

        if seed is not None:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                n=1,
                seed=seed,
            )
        else:
            sampling_params = self._sampling_params

        outputs = self.llm.generate([prompt], sampling_params)
        raw_output = outputs[0].outputs[0].text.strip()

        error_parts: List[str] = []
        action: Optional[Dict[str, Any]] = None
        parse_ok = False

        if _looks_like_markdown_wrapped(raw_output):
            error_parts.append("markdown_code_block")

        if _count_json_objects(raw_output) > 1:
            error_parts.append("multiple_json_objects")

        try:
            action = parse_action(raw_output)
            parse_ok = True
        except ActionParseError as exc:
            error_parts.append(str(exc))

        return {
            "prompt": filled,
            "raw_output": raw_output,
            "action": action,
            "parse_ok": parse_ok and not error_parts,
            "error": "; ".join(error_parts) if error_parts else None,
        }
