#!/usr/bin/env python3
"""
Phase 2.5e: KL / loss wiring audit.

Runs fixed-batch 10-step sweeps at kl_coef = 0.00 / 0.01 / 0.10 to verify
whether kl_coef affects backward loss and gradients.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  CUDA_VISIBLE_DEVICES=4,5 python scripts/audit_kl_loss_wiring.py \\
    --output-dir experiments/phase25e_kl_loss_audit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.kl_loss_audit import (  # noqa: E402
    _step_record,
    analyze_kl_audit_runs,
    write_audit_outputs,
)
from src.agents.tiny_grpo_smoke_trainer import TinyGrpoSmokeTrainer  # noqa: E402


DEFAULT_PREFLIGHT = (
    ROOT / "experiments/phase25_expanded_clean_set/preflight_v2_rollout_50_g4"
)


def _coef_label(coef: float) -> str:
    s = f"{coef:.2f}".rstrip("0").rstrip(".")
    return f"kl_{s}"


def run_sweep(
    *,
    kl_coefs: List[float],
    args: argparse.Namespace,
) -> Dict[str, List[Dict[str, Any]]]:
    rollout_path = args.preflight_rollout_dir / "v2_rollout_records.jsonl"
    shaped_path = args.preflight_rollout_dir / "large_k_shaped_record_rewards.jsonl"
    if not rollout_path.exists() or not shaped_path.exists():
        raise FileNotFoundError(f"missing preflight under {args.preflight_rollout_dir}")

    runs: Dict[str, List[Dict[str, Any]]] = {}
    for coef in kl_coefs:
        label = _coef_label(coef)
        out = args.output_dir / label
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Audit sweep {label} (kl_coef={coef}) ===")

        trainer = TinyGrpoSmokeTrainer(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            candidate_name=args.reward_candidate,
            train_batch_size=args.train_batch_size,
            rollout_n=args.rollout_n,
            ppo_mini_batch_size=args.ppo_mini_batch_size,
            micro_batch_size=args.micro_batch_size,
            max_update_steps=args.max_steps,
            learning_rate=args.learning_rate,
            kl_coef=coef,
            cliprange=args.cliprange,
            seed=args.seed,
            cuda_device_index=args.train_gpu,
        )

        result = trainer.run(
            rollout_path=rollout_path,
            shaped_reward_path=shaped_path,
            output_dir=out,
            metrics_filename=f"{label}_metrics.jsonl",
            summary_filename=f"{label}_summary.json",
            config_filename=f"{label}_config.yaml",
            phase="2.5e",
            mode="kl_loss_wiring_audit",
            save_steps=[],
            eval_steps=[],
        )

        records = [
            _step_record(m, effective_kl_coef=coef)
            for m in result["step_metrics"]
            if m.get("optimizer_step_called")
        ]
        runs[label] = records

        effective = {
            "effective_learning_rate": args.learning_rate,
            "effective_kl_coef": coef,
            "loss_formula": "policy_loss + kl_coef * kl_loss (expected)",
            "loss_formula_observed": "policy_loss only in backward",
            "kl_coef_source": "cli",
            "steps_completed": len(records),
        }
        (out / "effective_config.json").write_text(
            json.dumps(effective, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(effective, ensure_ascii=False, indent=2))

    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2.5e KL/loss wiring audit")
    parser.add_argument(
        "--preflight-rollout-dir",
        type=Path,
        default=DEFAULT_PREFLIGHT,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data1/hcc/.hf_home/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/data1/hcc/.hf_home/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/phase25e_kl_loss_audit",
    )
    parser.add_argument("--reward-candidate", type=str, default="reward_largek_mix_1000")
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--kl-coefs", type=float, nargs="+", default=[0.0, 0.01, 0.10])
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=50)
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=50)
    parser.add_argument("--micro-batch-size", type=int, default=5)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-gpu", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = run_sweep(kl_coefs=args.kl_coefs, args=args)
    coef_map = {_coef_label(c): c for c in args.kl_coefs}
    summary = analyze_kl_audit_runs(runs, kl_coef_by_label=coef_map)
    write_audit_outputs(args.output_dir, runs=runs, summary=summary)

    print("\n=== Phase 2.5e KL Audit Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary.get("audit_passed"):
        raise SystemExit("KL/loss wiring audit FAILED — fix loss wiring before config B")


if __name__ == "__main__":
    main()
