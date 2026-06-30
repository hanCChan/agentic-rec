"""
Phase 2.3: Controlled 10-step GRPO smoke trainer.

Extends TinyGrpoSmokeTrainer with selective checkpoint saving and
Phase 2.3 output naming conventions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.agents.grpo_stability_monitor import GRPOStabilityMonitor
from src.agents.tiny_grpo_smoke_trainer import (
    CHECKPOINT_LABEL,
    TINY_TRAIN_WARNING,
    TinyGrpoSmokeTrainer,
)

CONTROLLED_SMOKE_WARNING = (
    "Phase 2.3 controlled 10-step GRPO smoke only. "
    "Checkpoints are SMOKE_ONLY_DO_NOT_PROMOTE and must not be promoted."
)


class ControlledGrpoSmokeTrainer(TinyGrpoSmokeTrainer):
    """Phase 2.3 controlled multi-step GRPO smoke trainer."""

    def run_controlled(
        self,
        rollout_path: str | Path,
        shaped_reward_path: str | Path,
        output_dir: str | Path,
        *,
        save_steps: Optional[List[int]] = None,
        max_prompt_length: int = 1024,
        max_response_length: int = 2048,
        max_total_length: int = 3072,
        stability_monitor: Optional[GRPOStabilityMonitor] = None,
        metrics_filename: str = "ten_step_train_metrics.jsonl",
        summary_filename: str = "ten_step_train_summary.json",
        config_filename: str = "ten_step_train_config.yaml",
        phase: str = "2.3",
        mode: str = "10step_grpo_controlled_smoke",
    ) -> Dict[str, Any]:
        monitor = stability_monitor or GRPOStabilityMonitor(
            max_signed_logprob_gap_abs=5.0,
            max_approx_kl=0.2,
            max_grad_norm=10.0,
        )

        result = self.run(
            rollout_path=rollout_path,
            shaped_reward_path=shaped_reward_path,
            output_dir=output_dir,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            max_total_length=max_total_length,
            metrics_filename=metrics_filename,
            summary_filename=summary_filename,
            config_filename=config_filename,
            phase=phase,
            mode=mode,
            stability_monitor=monitor,
            save_steps=save_steps,
        )

        summary = result["summary"]
        summary["dryrun_warning"] = CONTROLLED_SMOKE_WARNING
        summary["optimizer_steps_called"] = summary.get("actual_update_steps", 0)
        summary["checkpoint_label"] = CHECKPOINT_LABEL

        out = Path(output_dir)
        (out / summary_filename).write_text(
            __import__("json").dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result["summary"] = summary
        return result
