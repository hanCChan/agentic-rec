#!/usr/bin/env python3
"""
Phase 2.4: 50-step controlled GRPO pilot.

Phase 2.4a — plan only (see docs/PHASE2_4_50STEP_PILOT_PLAN.md)
Phase 2.4b — dry-config check:
  python scripts/run_50step_grpo_pilot.py --dry-config-check
Phase 2.4  — pilot training:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_50step_grpo_pilot.py \\
    --output-dir experiments/phase24_50step_grpo_pilot/lr_5e-7
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
REC_R1 = ROOT / "Rec-R1"
PHASE21_DIR = ROOT / "experiments/phase21_tiny_grpo_smoke"
PHASE23_DIR = ROOT / "experiments/phase23_10step_grpo_controlled_smoke"
DEFAULT_OUTPUT = ROOT / "experiments/phase24_50step_grpo_pilot"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REC_R1))

from src.agents.controlled_grpo_smoke_trainer import ControlledGrpoSmokeTrainer
from src.agents.grpo_curve_analyzer import GRPOCurveAnalyzer
from src.agents.grpo_pilot_monitor import GRPOPilotMonitor, PILOT_CHECKPOINT_LABEL
from src.agents.periodic_fresh_eval import make_periodic_eval_hook, run_fresh_eval
from src.agents.phase2_smoke_dataset import load_clean_set_rows
from src.agents.tiny_grpo_smoke_trainer import CHECKPOINT_LABEL

PILOT_CHECKPOINT_PREFIX = "pilot_step"

DEFAULT_CONFIG: Dict[str, Any] = {
    "phase": "2.4",
    "mode": "50step_grpo_pilot",
    "max_update_steps": 50,
    "learning_rate": 5e-7,
    "kl_coef": 0.01,
    "cliprange": 0.2,
    "train_batch_size": 20,
    "rollout_n": 4,
    "ppo_mini_batch_size": 20,
    "micro_batch_size": 4,
    "max_prompt_length": 1024,
    "max_response_length": 2048,
    "max_total_length": 3072,
    "loss_agg_mode": "token-mean",
    "reward_candidate": "reward_largek_mix_1000",
    "penalties_in_advantage": False,
    "diagnostic_oracle_reward_used": False,
    "save_steps": [10, 25, 50],
    "eval_steps": [0, 10, 25, 50],
    "checkpoint_label": PILOT_CHECKPOINT_LABEL,
    "checkpoint_promoted": False,
    "cuda_visible_devices": "0,1,2,3",
    "train_gpu": 0,
    "eval_gpu": 1,
    "min_disk_gb": 40,
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_clean_set(clean_set_path: Path, summary_path: Path) -> Dict[str, Any]:
    rows = load_clean_set_rows(clean_set_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"missing clean set summary: {summary_path}")
    summary = _load_json(summary_path)
    if not summary.get("phase2_clean_set_ready"):
        raise ValueError("clean set not ready")
    if len(rows) != int(summary.get("num_groups", 20)):
        raise ValueError(f"clean set row count {len(rows)} != summary num_groups")
    if "esci_val_3" in [r["group_id"] for r in rows]:
        raise ValueError("esci_val_3 must not be in clean set")
    return summary


def _make_eval_fn(args: argparse.Namespace, output_dir: Path):
    def run_eval_at_step(eval_step: int, checkpoint_path: Path) -> Dict[str, Any]:
        return run_fresh_eval(
            clean_set_path=args.clean_set_path,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            eval_step=eval_step,
            data_path=args.data_path,
            preflight_rollout_dir=args.preflight_rollout_dir,
            candidate_name=args.reward_candidate,
            model_path=args.model_path,
            temperature=args.temperature,
            top_p=args.top_p,
            topk=args.topk,
            seed=args.seed,
            root=ROOT,
            checkpoint_prefix=PILOT_CHECKPOINT_PREFIX,
            checkpoint_label=PILOT_CHECKPOINT_LABEL,
        )

    return run_eval_at_step


def dry_config_check(args: argparse.Namespace) -> Dict[str, Any]:
    """Phase 2.4b: validate inputs, disk, GPU, config without training."""
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    checks: Dict[str, Any] = {"phase": "2.4b", "mode": "dry_config_check", "checks": {}}

    def _ok(name: str, passed: bool, detail: str = "") -> None:
        checks["checks"][name] = {"passed": passed, "detail": detail}

    # clean set
    try:
        summary = validate_clean_set(args.clean_set_path, args.clean_set_summary_path)
        _ok("clean_set", True, f"{summary.get('num_groups')} groups ready")
    except Exception as exc:
        _ok("clean_set", False, str(exc))

    # preflight rollout
    rollout_path = args.preflight_rollout_dir / "v2_rollout_records.jsonl"
    shaped_path = args.preflight_rollout_dir / "large_k_shaped_record_rewards.jsonl"
    if rollout_path.exists() and shaped_path.exists():
        _ok("preflight_rollout", True, str(args.preflight_rollout_dir))
    else:
        _ok("preflight_rollout", False, f"missing artifacts under {args.preflight_rollout_dir}")

    # model path
    model_ok = Path(args.model_path).exists()
    _ok("model_path", model_ok, args.model_path)

    # disk
    usage = shutil.disk_usage(out if out.exists() else out.parent)
    free_gb = usage.free / (1024**3)
    disk_ok = free_gb >= args.min_disk_gb
    _ok("disk_space", disk_ok, f"free={free_gb:.1f}GB required>={args.min_disk_gb}GB")

    # GPU
    gpu_info = "not checked (no torch)"
    gpu_ok = True
    try:
        import torch

        gpu_ok = torch.cuda.is_available()
        if gpu_ok:
            n = torch.cuda.device_count()
            names = [torch.cuda.get_device_name(i) for i in range(n)]
            gpu_info = f"{n} devices: {names}"
        else:
            gpu_info = "CUDA not available"
            gpu_ok = False
    except ImportError:
        gpu_ok = False
        gpu_info = "torch not installed"
    _ok("gpu", gpu_ok, gpu_info)

    # phase 2.3 baseline reference
    p23 = PHASE23_DIR / "lr_5e-7/ten_step_train_summary.json"
    if p23.exists():
        p23s = _load_json(p23)
        _ok(
            "phase23_baseline",
            p23s.get("ten_step_smoke_passed") is True,
            f"max_kl={p23s.get('max_approx_kl_nonnegative')}",
        )
    else:
        _ok("phase23_baseline", False, "missing phase23 lr_5e-7 summary")

    # config snapshot
    config = {**DEFAULT_CONFIG}
    config.update(
        {
            "learning_rate": args.learning_rate,
            "max_update_steps": args.max_update_steps,
            "save_steps": args.save_steps,
            "eval_steps": args.eval_steps,
            "clean_set_path": str(args.clean_set_path),
            "preflight_rollout_dir": str(args.preflight_rollout_dir),
            "model_path": args.model_path,
            "output_dir": str(out / "lr_5e-7"),
        }
    )
    config_path = out / "pilot_config.yaml"
    try:
        import yaml

        config_path.write_text(yaml.dump(config, default_flow_style=False), encoding="utf-8")
        _ok("config_write", True, str(config_path))
    except ImportError:
        config_path = out / "pilot_config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        _ok("config_write", True, str(config_path))

    all_passed = all(c["passed"] for c in checks["checks"].values())
    checks["dry_config_check_passed"] = all_passed
    checks["ready_for_pilot"] = all_passed
    checks["next_step"] = (
        "Phase 2.4: run 50-step pilot with CUDA_VISIBLE_DEVICES=0,1,2,3"
        if all_passed
        else "Fix failing checks before pilot"
    )

    check_path = out / "dry_config_check.json"
    check_path.write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return checks


def run_pilot(args: argparse.Namespace) -> None:
    """Phase 2.4: 50-step controlled pilot training."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validate_clean_set(args.clean_set_path, args.clean_set_summary_path)

    rollout_path = args.preflight_rollout_dir / "v2_rollout_records.jsonl"
    shaped_path = args.preflight_rollout_dir / "large_k_shaped_record_rewards.jsonl"
    if not rollout_path.exists() or not shaped_path.exists():
        raise SystemExit(f"missing preflight artifacts under {args.preflight_rollout_dir}")

    monitor = GRPOPilotMonitor(
        max_approx_kl=0.2,
        max_grad_norm=10.0,
        max_reward_drop_ratio=0.30,
    )

    eval_summaries: List[Dict[str, Any]] = []
    early_stop = False
    early_stop_reason: Optional[str] = None

    # Step-0 baseline eval (before training)
    if 0 in args.eval_steps and not args.skip_eval:
        print("\n=== Step-0 baseline fresh eval ===")
        try:
            baseline = run_fresh_eval(
                clean_set_path=args.clean_set_path,
                checkpoint_path=None,
                output_dir=args.output_dir,
                eval_step=0,
                data_path=args.data_path,
                preflight_rollout_dir=args.preflight_rollout_dir,
                candidate_name=args.reward_candidate,
                model_path=args.model_path,
                temperature=args.temperature,
                top_p=args.top_p,
                topk=args.topk,
                seed=args.seed,
                root=ROOT,
                checkpoint_prefix=PILOT_CHECKPOINT_PREFIX,
                checkpoint_label=PILOT_CHECKPOINT_LABEL,
            )
            eval_summaries.append(baseline)
            monitor.set_baseline_reward(baseline["mean_reward_largek_mix_1000"])
            check = monitor.check_eval_summary(baseline, step=0)
            if check["should_stop"]:
                early_stop = True
                early_stop_reason = check["stop_reason"]
        except Exception as exc:
            early_stop = True
            early_stop_reason = f"step-0 eval failed: {exc}"

    if early_stop:
        _write_failure(args.output_dir, early_stop_reason or "unknown")
        raise SystemExit(f"Pilot stopped before training: {early_stop_reason}")

    # Periodic eval hook: eval immediately after each checkpoint save step
    early_stop_state: Dict[str, Any] = {"stopped": False, "reason": None}
    eval_fn = _make_eval_fn(args, args.output_dir)
    step_hook = None
    if not args.skip_eval:
        step_hook = make_periodic_eval_hook(
            eval_steps=args.eval_steps,
            eval_summaries=eval_summaries,
            run_eval_fn=eval_fn,
            monitor=monitor,
            early_stop_state=early_stop_state,
        )

    trainer = ControlledGrpoSmokeTrainer(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        candidate_name=args.reward_candidate,
        train_batch_size=args.train_batch_size,
        rollout_n=args.rollout_n,
        ppo_mini_batch_size=args.ppo_mini_batch_size,
        micro_batch_size=args.micro_batch_size,
        max_update_steps=args.max_update_steps,
        learning_rate=args.learning_rate,
        kl_coef=args.kl_coef,
        cliprange=args.cliprange,
        seed=args.seed,
    )

    try:
        result = trainer.run_controlled(
            rollout_path=rollout_path,
            shaped_reward_path=shaped_path,
            output_dir=args.output_dir,
            save_steps=args.save_steps,
            max_prompt_length=args.max_prompt_length,
            max_response_length=args.max_response_length,
            max_total_length=args.max_total_length,
            stability_monitor=monitor,
            metrics_filename="pilot_train_metrics.jsonl",
            summary_filename="pilot_train_summary.json",
            config_filename="pilot_train_config.yaml",
            phase="2.4",
            mode="50step_grpo_pilot",
            checkpoint_prefix=PILOT_CHECKPOINT_PREFIX,
            step_hook=step_hook,
        )
    except Exception as exc:
        (args.output_dir / "failure_report.md").write_text(
            f"# Phase 2.4 Pilot Failure\n\n{exc}\n\n{traceback.format_exc()}\n",
            encoding="utf-8",
        )
        raise SystemExit(1) from exc

    step_metrics = result["step_metrics"]

    if early_stop_state.get("stopped"):
        early_stop = True
        early_stop_reason = early_stop_state.get("reason")

    monitor_report = monitor.summarize_pilot(step_metrics, eval_summaries)
    analyzer = GRPOCurveAnalyzer()
    curve = analyzer.analyze_trends(step_metrics)
    rec = analyzer.recommend_next_step(curve)

    (args.output_dir / "curve_analysis.json").write_text(
        json.dumps({**curve, "recommendation": rec}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    passed = (
        result["summary"].get("actual_update_steps") == args.max_update_steps
        and monitor_report["optimizer_steps_called"] == args.max_update_steps
        and monitor_report["all_steps_passed"]
        and not monitor_report["nan_detected"]
        and not monitor_report["oom_detected"]
        and not monitor_report["kl_exploded"]
        and curve["max_approx_kl_nonnegative"] <= 0.2
        and curve["max_grad_norm"] <= 10.0
        and monitor_report.get("eval_passed_all", True)
        and not early_stop
    )

    summary = {
        "phase": "2.4",
        "mode": "50step_grpo_pilot",
        **result["summary"],
        "learning_rate": args.learning_rate,
        "max_update_steps": args.max_update_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "pilot_passed": passed,
        "early_stop": early_stop,
        "early_stop_reason": early_stop_reason,
        "optimizer_steps_called": monitor_report["optimizer_steps_called"],
        "loss_finite_all_steps": monitor_report["loss_finite_all_steps"],
        "grad_norm_finite_all_steps": monitor_report["grad_norm_finite_all_steps"],
        "max_grad_norm": curve["max_grad_norm"],
        "max_approx_kl_nonnegative": curve["max_approx_kl_nonnegative"],
        "max_abs_signed_logprob_gap": curve["max_abs_signed_logprob_gap"],
        "max_clipfrac": curve["max_clipfrac"],
        "stability_class": curve["stability_class"],
        "baseline_reward": monitor.baseline_reward,
        "eval_summaries": [
            {
                "step": e.get("eval_step"),
                "mean_reward": e.get("mean_reward_largek_mix_1000"),
                "parse_success_rate": e.get("parse_success_rate"),
            }
            for e in eval_summaries
        ],
        "checkpoint_label": PILOT_CHECKPOINT_LABEL,
        "checkpoint_promoted": False,
        "safe_for_larger_training": False,
        "next_recommendation": rec["action"],
    }

    (args.output_dir / "pilot_train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = _build_manifest(args.output_dir, args.save_steps)
    (args.output_dir / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# Phase 2.4 50-Step Pilot Report",
        "",
        f"- learning_rate: **{args.learning_rate}**",
        f"- actual_update_steps: **{summary.get('actual_update_steps')}**",
        f"- pilot_passed: **{passed}**",
        f"- stability_class: **{curve.get('stability_class')}**",
        f"- max_approx_kl_nonnegative: **{curve.get('max_approx_kl_nonnegative')}**",
        "",
        "## Eval Timeline",
        "",
    ]
    for ev in eval_summaries:
        report_lines.append(
            f"- step {ev.get('eval_step')}: reward={ev.get('mean_reward_largek_mix_1000'):.4f} "
            f"parse={ev.get('parse_success_rate')}"
        )
    (args.output_dir / "pilot_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("\n=== Phase 2.4 Pilot Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not passed:
        _write_failure(args.output_dir, early_stop_reason or "acceptance criteria not met")
        raise SystemExit("50-step pilot did not pass acceptance criteria")


def _build_manifest(output_dir: Path, save_steps: List[int]) -> Dict[str, Any]:
    checkpoints = []
    for step in save_steps:
        path = output_dir / "checkpoints" / f"pilot_step_{step}"
        if path.exists():
            checkpoints.append(
                {
                    "step": step,
                    "path": str(path),
                    "label": PILOT_CHECKPOINT_LABEL,
                    "optimizer_step_called": True,
                }
            )
    return {
        "checkpoint_promoted": False,
        "checkpoint_label": PILOT_CHECKPOINT_LABEL,
        "save_steps": save_steps,
        "checkpoints": checkpoints,
    }


def _write_failure(output_dir: Path, reason: str) -> None:
    (output_dir / "failure_report.md").write_text(
        f"# Phase 2.4 Pilot Failure\n\n{reason}\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2.4 50-step GRPO pilot")
    parser.add_argument("--dry-config-check", action="store_true", help="Phase 2.4b only")
    parser.add_argument(
        "--clean-set-path",
        type=Path,
        default=PHASE21_DIR / "phase2_clean_20_groups.jsonl",
    )
    parser.add_argument(
        "--clean-set-summary-path",
        type=Path,
        default=PHASE21_DIR / "phase2_clean_set_summary.json",
    )
    parser.add_argument(
        "--preflight-rollout-dir",
        type=Path,
        default=PHASE21_DIR / "preflight_v2_rollout_20_g4",
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "lr_5e-7")
    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Root for dry-config-check output",
    )
    parser.add_argument("--reward-candidate", type=str, default="reward_largek_mix_1000")
    parser.add_argument("--max-update-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--train-batch-size", type=int, default=20)
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=20)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--save-steps", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--eval-steps", type=int, nargs="+", default=[0, 10, 25, 50])
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--max-total-length", type=int, default=3072)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "Rec-R1/data/esci/inst/sparse/subset/val.parquet",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-gpu", type=int, default=0)
    parser.add_argument("--eval-gpu", type=int, default=1)
    parser.add_argument("--min-disk-gb", type=float, default=40.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_config_check:
        args.output_dir = args.pilot_root
        result = dry_config_check(args)
        if not result.get("dry_config_check_passed"):
            raise SystemExit("dry-config check failed")
        return
    run_pilot(args)


if __name__ == "__main__":
    main()
