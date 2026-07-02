#!/usr/bin/env python3
"""
Phase 2.5c: 200-step controlled GRPO pilot with train + heldout fresh eval.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  CUDA_VISIBLE_DEVICES=6,7 python scripts/run_200step_grpo_pilot.py \\
    --train-clean-path experiments/phase25_expanded_clean_set/train_clean_50_groups.jsonl \\
    --heldout-clean-path experiments/phase25_expanded_clean_set/heldout_clean_20_groups.jsonl \\
    --output-dir experiments/phase25_200step_grpo_pilot/lr_5e-7
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
REC_R1 = ROOT / "Rec-R1"
PHASE25_DIR = ROOT / "experiments/phase25_expanded_clean_set"
DEFAULT_OUTPUT = ROOT / "experiments/phase25_200step_grpo_pilot"
PILOT_CHECKPOINT_PREFIX = "pilot_step"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REC_R1))

from src.agents.controlled_grpo_smoke_trainer import ControlledGrpoSmokeTrainer
from src.agents.grpo_curve_analyzer import GRPOCurveAnalyzer
from src.agents.grpo_pilot_heldout_monitor import GRPOPilotHeldoutMonitor
from src.agents.grpo_pilot_monitor import PILOT_CHECKPOINT_LABEL
from src.agents.periodic_fresh_eval import (
    build_samples_from_clean_rows,
    make_periodic_dual_eval_hook,
    run_dual_fresh_eval,
    run_final_stop_dual_eval,
)
from src.agents.phase2_smoke_dataset import load_clean_set_rows


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_v2_module() -> Any:
    script_path = ROOT / "scripts/smoke_strategy_prompt_v2.py"
    spec = importlib.util.spec_from_file_location("smoke_strategy_prompt_v2", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_expanded_clean_set(
    train_path: Path,
    heldout_path: Path,
    summary_path: Path,
) -> Dict[str, Any]:
    train_rows = load_clean_set_rows(train_path)
    heldout_rows = load_clean_set_rows(heldout_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary: {summary_path}")
    summary = _load_json(summary_path)
    if not summary.get("expanded_clean_set_ready"):
        raise ValueError("expanded clean set not ready")
    train_ids = {r["group_id"] for r in train_rows}
    heldout_ids = {r["group_id"] for r in heldout_rows}
    overlap = train_ids & heldout_ids
    if overlap:
        raise ValueError(f"train/heldout overlap: {sorted(overlap)}")
    if "esci_val_3" in train_ids | heldout_ids:
        raise ValueError("esci_val_3 must not be in clean sets")
    return summary


def ensure_preflight_rollout(
    *,
    clean_path: Path,
    output_dir: Path,
    model_path: str,
    data_path: Path,
    temperature: float,
    top_p: float,
    topk: int,
    seed: int,
    candidate_name: str,
) -> Path:
    """Build preflight v2 rollout artifacts if missing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = output_dir / "v2_rollout_records.jsonl"
    shaped_path = output_dir / "large_k_shaped_record_rewards.jsonl"
    if rollout_path.exists() and shaped_path.exists():
        print(f"[phase25c] preflight ready: {output_dir}")
        return output_dir

    print(f"[phase25c] building preflight rollout: {output_dir}")
    v2_module = _load_v2_module()
    candidate_rows = load_clean_set_rows(clean_path)
    samples = build_samples_from_clean_rows(candidate_rows, data_path, v2_module)

    groups, rollout_records, failures = v2_module.run_v2_rollout(
        samples,
        model_path=Path(model_path),
        temperature=temperature,
        top_p=top_p,
        max_tokens=256,
        max_steps=3,
        topk=topk,
        seed=seed,
        strategies=["exact_match", "attribute_expansion", "broad_recall", "constraint_preserving"],
    )
    if failures:
        raise RuntimeError(f"preflight rollout failures: {failures[:3]}")

    v2_module._write_jsonl(rollout_path, rollout_records)
    post = v2_module.run_post_analysis(
        rollout_path,
        output_dir,
        k_list=[10, 50, 100, 1000],
        candidate_name=candidate_name,
    )
    summary = {
        "num_groups": len(groups),
        "num_rollout_records": len(rollout_records),
        "candidate_name": candidate_name,
        "v2_gate_passed": post.get("gate", {}).get("gate_passed", False),
        "v2_retrieval_quality_spread_group_rate": post.get("large_k_summary", {}).get(
            "retrieval_quality_spread_group_rate"
        ),
        "v2_zero_std_group_rate": post.get("large_k_summary", {}).get("zero_std_group_rate"),
        "preflight_built_for": str(clean_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir


def build_train_vs_heldout_curve(periodic_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for rec in periodic_records:
        rows.append(
            {
                "step": rec.get("step"),
                "train_reward": rec.get("train_mean_reward_largek_mix_1000"),
                "heldout_reward": rec.get("heldout_mean_reward_largek_mix_1000"),
                "train_parse": rec.get("train_parse_success_rate"),
                "heldout_parse": rec.get("heldout_parse_success_rate"),
                "train_invalid": rec.get("train_invalid_action_rate"),
                "heldout_invalid": rec.get("heldout_invalid_action_rate"),
                "overfit_risk": rec.get("overfit_risk", False),
            }
        )
    return {"curve": rows}


def build_curve_markdown(
    periodic_records: List[Dict[str, Any]],
    step_metrics: List[Dict[str, Any]],
) -> str:
    metric_by_step = {m.get("step", i + 1): m for i, m in enumerate(step_metrics)}
    lines = [
        "# Train vs Heldout Curve",
        "",
        "| step | train_reward | heldout_reward | train_parse | heldout_parse | approx_kl | grad_norm |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rec in periodic_records:
        step = rec.get("step", 0)
        m = metric_by_step.get(step, {})
        lines.append(
            f"| {step} | "
            f"{rec.get('train_mean_reward_largek_mix_1000', 0):.4f} | "
            f"{rec.get('heldout_mean_reward_largek_mix_1000', 0):.4f} | "
            f"{rec.get('train_parse_success_rate', 0):.4f} | "
            f"{rec.get('heldout_parse_success_rate', 0):.4f} | "
            f"{m.get('approx_kl_nonnegative', '')} | "
            f"{m.get('grad_norm', '')} |"
        )
    return "\n".join(lines) + "\n"


def build_overfit_report(periodic_records: List[Dict[str, Any]]) -> str:
    risky = [r for r in periodic_records if r.get("overfit_risk")]
    lines = [
        "# Overfit Risk Report",
        "",
        f"- eval_points: **{len(periodic_records)}**",
        f"- overfit_risk_points: **{len(risky)}**",
        "",
    ]
    if risky:
        lines.append("## Overfit Signals")
        lines.append("")
        for r in risky:
            lines.append(
                f"- step {r.get('step')}: train={r.get('train_mean_reward_largek_mix_1000'):.4f}, "
                f"heldout={r.get('heldout_mean_reward_largek_mix_1000'):.4f} — {r.get('overfit_reason')}"
            )
    else:
        lines.append("No overfit_risk hard-stop signals recorded.")
    return "\n".join(lines) + "\n"


def dry_config_check(args: argparse.Namespace) -> Dict[str, Any]:
    out = args.output_dir if args.output_dir.name != "lr_5e-7" else args.pilot_root
    out.mkdir(parents=True, exist_ok=True)
    checks: Dict[str, Any] = {"phase": "2.5c", "mode": "dry_config_check", "checks": {}}

    def _ok(name: str, passed: bool, detail: str = "") -> None:
        checks["checks"][name] = {"passed": passed, "detail": detail}

    try:
        summary = validate_expanded_clean_set(
            args.train_clean_path, args.heldout_clean_path, args.clean_set_summary_path
        )
        _ok(
            "expanded_clean_set",
            True,
            f"train={summary.get('train_clean_groups')} heldout={summary.get('heldout_clean_groups')}",
        )
    except Exception as exc:
        _ok("expanded_clean_set", False, str(exc))

    model_ok = Path(args.model_path).exists()
    _ok("model_path", model_ok, args.model_path)

    usage = shutil.disk_usage(out if out.exists() else out.parent)
    free_gb = usage.free / (1024**3)
    _ok("disk_space", free_gb >= args.min_disk_gb, f"free={free_gb:.1f}GB")

    gpu_ok = True
    gpu_info = "not checked"
    try:
        import torch

        gpu_ok = torch.cuda.is_available()
        if gpu_ok:
            gpu_info = f"{torch.cuda.device_count()} devices"
        else:
            gpu_info = "CUDA not available"
    except ImportError:
        gpu_ok = False
        gpu_info = "torch not installed"
    _ok("gpu", gpu_ok, gpu_info)

    config = {
        "phase": "2.5c",
        "mode": "200step_grpo_pilot",
        "train_clean_path": str(args.train_clean_path),
        "heldout_clean_path": str(args.heldout_clean_path),
        "preflight_rollout_dir": str(args.preflight_rollout_dir),
        "max_update_steps": args.max_update_steps,
        "learning_rate": args.learning_rate,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "train_batch_size": args.train_batch_size,
        "output_dir": str(args.output_dir),
    }
    config_path = out / "pilot_200step_config.yaml"
    config_path.write_text(
        "\n".join(f"{k}: {v}" for k, v in config.items()) + "\n",
        encoding="utf-8",
    )
    _ok("config_write", True, str(config_path))

    checks["dry_config_check_passed"] = all(c["passed"] for c in checks["checks"].values())
    (out / "dry_config_check.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return checks


def run_pilot(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validate_expanded_clean_set(
        args.train_clean_path, args.heldout_clean_path, args.clean_set_summary_path
    )

    if not args.skip_preflight_build:
        ensure_preflight_rollout(
            clean_path=args.train_clean_path,
            output_dir=args.preflight_rollout_dir,
            model_path=args.model_path,
            data_path=args.data_path,
            temperature=args.temperature,
            top_p=args.top_p,
            topk=args.topk,
            seed=args.seed,
            candidate_name=args.reward_candidate,
        )

    rollout_path = args.preflight_rollout_dir / "v2_rollout_records.jsonl"
    shaped_path = args.preflight_rollout_dir / "large_k_shaped_record_rewards.jsonl"
    if not rollout_path.exists() or not shaped_path.exists():
        raise SystemExit(f"missing preflight artifacts under {args.preflight_rollout_dir}")

    monitor = GRPOPilotHeldoutMonitor(
        max_approx_kl=0.2,
        max_grad_norm=10.0,
        max_train_reward_drop_ratio=0.30,
        max_heldout_reward_drop_ratio=0.20,
    )

    periodic_records: List[Dict[str, Any]] = []
    early_stop = False
    early_stop_reason: Optional[str] = None

    def _dual_eval_fn(eval_step: int, checkpoint_path: Optional[Path]) -> Dict[str, Any]:
        return run_dual_fresh_eval(
            train_clean_path=args.train_clean_path,
            heldout_clean_path=args.heldout_clean_path,
            checkpoint_path=checkpoint_path,
            output_dir=args.output_dir,
            eval_step=eval_step,
            data_path=args.data_path,
            train_preflight_dir=args.preflight_rollout_dir,
            heldout_preflight_dir=None,
            candidate_name=args.reward_candidate,
            model_path=args.model_path,
            temperature=args.temperature,
            top_p=args.top_p,
            topk=args.topk,
            seed=args.seed,
            root=ROOT,
            checkpoint_prefix=PILOT_CHECKPOINT_PREFIX,
            checkpoint_label=PILOT_CHECKPOINT_LABEL,
            eval_cuda_device=args.eval_gpu,
        )

    if 0 in args.eval_steps and not args.skip_eval:
        print("\n=== Step-0 baseline train+heldout fresh eval ===")
        try:
            baseline = _dual_eval_fn(0, None)
            rec = monitor.check_eval_pair(baseline["train"], baseline["heldout"], step=0)
            periodic_records.append(rec)
            monitor.set_baselines(
                baseline["train"]["mean_reward_largek_mix_1000"],
                baseline["heldout"]["mean_reward_largek_mix_1000"],
            )
            if rec["should_stop"]:
                early_stop = True
                early_stop_reason = rec["stop_reason"]
        except Exception as exc:
            early_stop = True
            early_stop_reason = f"step-0 eval failed: {exc}"

    if early_stop:
        _write_failure(args.output_dir, early_stop_reason or "unknown")
        raise SystemExit(f"Pilot stopped before training: {early_stop_reason}")

    early_stop_state: Dict[str, Any] = {"stopped": False, "reason": None}
    step_hook = None
    if not args.skip_eval:
        step_hook = make_periodic_dual_eval_hook(
            eval_steps=args.eval_steps,
            periodic_records=periodic_records,
            run_dual_eval_fn=lambda step, ckpt: _dual_eval_fn(step, ckpt),
            monitor=monitor,
            early_stop_state=early_stop_state,
        )

    phase_label = getattr(args, "phase", "2.5c")
    mode_label = getattr(args, "mode", "200step_grpo_pilot")

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
        cuda_device_index=args.train_gpu,
    )

    try:
        result = trainer.run_controlled(
            rollout_path=rollout_path,
            shaped_reward_path=shaped_path,
            output_dir=args.output_dir,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            max_prompt_length=args.max_prompt_length,
            max_response_length=args.max_response_length,
            max_total_length=args.max_total_length,
            stability_monitor=monitor,
            metrics_filename="pilot_200step_train_metrics.jsonl",
            summary_filename="pilot_200step_summary.json",
            config_filename="pilot_200step_config.yaml",
            phase=phase_label,
            mode=mode_label,
            checkpoint_prefix=PILOT_CHECKPOINT_PREFIX,
            step_hook=step_hook,
        )
    except Exception as exc:
        (args.output_dir / "failure_report.md").write_text(
            f"# Phase 2.5c Pilot Failure\n\n{exc}\n\n{traceback.format_exc()}\n",
            encoding="utf-8",
        )
        raise SystemExit(1) from exc

    step_metrics = result["step_metrics"]
    if early_stop_state.get("stopped"):
        early_stop = True
        early_stop_reason = early_stop_state.get("reason")

    actual_step = result["summary"].get("actual_update_steps", 0)
    eval_done_steps = {r.get("step") for r in periodic_records}
    kl_hard_stop = result["summary"].get("hard_stop") or result["summary"].get("kl_exploded")
    if (
        not args.skip_eval
        and actual_step > 0
        and actual_step not in eval_done_steps
        and (kl_hard_stop or actual_step < args.max_update_steps)
    ):
        stop_ckpt = result["summary"].get("stop_snapshot_path")
        print(f"\n=== Final stop eval at step {actual_step} ===")
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            dual = run_final_stop_dual_eval(
                train_clean_path=args.train_clean_path,
                heldout_clean_path=args.heldout_clean_path,
                checkpoint_path=Path(stop_ckpt) if stop_ckpt else None,
                output_dir=args.output_dir,
                stop_step=actual_step,
                data_path=args.data_path,
                train_preflight_dir=args.preflight_rollout_dir,
                heldout_preflight_dir=None,
                candidate_name=args.reward_candidate,
                model_path=args.model_path,
                temperature=args.temperature,
                top_p=args.top_p,
                topk=args.topk,
                seed=args.seed,
                root=ROOT,
                checkpoint_prefix=PILOT_CHECKPOINT_PREFIX,
                checkpoint_label=PILOT_CHECKPOINT_LABEL,
                eval_cuda_device=args.eval_gpu,
            )
            rec = monitor.check_eval_pair(dual["train"], dual["heldout"], step=actual_step)
            rec["final_stop_eval"] = True
            periodic_records.append(rec)
            if rec["should_stop"] and not early_stop:
                early_stop = True
                early_stop_reason = rec["stop_reason"]
        except Exception as exc:
            print(f"[final stop eval] failed: {exc}")

    monitor_report = monitor.summarize_pilot(step_metrics, [])
    analyzer = GRPOCurveAnalyzer()
    curve = analyzer.analyze_trends(step_metrics)
    rec = analyzer.recommend_next_step(curve)

    curve_data = build_train_vs_heldout_curve(periodic_records)
    (args.output_dir / "train_vs_heldout_curve.json").write_text(
        json.dumps(curve_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "train_vs_heldout_curve.md").write_text(
        build_curve_markdown(periodic_records, step_metrics),
        encoding="utf-8",
    )
    (args.output_dir / "overfit_risk_report.md").write_text(
        build_overfit_report(periodic_records),
        encoding="utf-8",
    )
    (args.output_dir / "periodic_eval_summary.json").write_text(
        json.dumps({"eval_history": periodic_records}, ensure_ascii=False, indent=2),
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
        and not early_stop
    )

    summary = {
        "phase": phase_label,
        "mode": mode_label,
        **result["summary"],
        "learning_rate": args.learning_rate,
        "kl_coef": args.kl_coef,
        "max_update_steps": args.max_update_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "train_clean_path": str(args.train_clean_path),
        "heldout_clean_path": str(args.heldout_clean_path),
        "pilot_passed": passed,
        "early_stop": early_stop,
        "early_stop_reason": early_stop_reason,
        "baseline_train_reward": monitor.baseline_train_reward,
        "baseline_heldout_reward": monitor.baseline_heldout_reward,
        "max_approx_kl_nonnegative": curve["max_approx_kl_nonnegative"],
        "max_grad_norm": curve["max_grad_norm"],
        "stability_class": curve["stability_class"],
        "periodic_eval_points": len(periodic_records),
        "overfit_risk_detected": any(r.get("overfit_risk") for r in periodic_records),
        "checkpoint_label": PILOT_CHECKPOINT_LABEL,
        "checkpoint_promoted": False,
        "safe_for_larger_training": passed,
        "next_recommendation": rec["action"],
    }
    if actual_step < args.max_update_steps:
        if kl_hard_stop:
            summary["failure_type"] = "kl_guard_stop"
            summary["stopped_by"] = result["summary"].get(
                "hard_stop_reason", "KL hard stop"
            )
        elif early_stop:
            summary["failure_type"] = "eval_early_stop"
            summary["stopped_by"] = early_stop_reason or "eval early stop"
        summary["stopped_at_step"] = actual_step
    (args.output_dir / "pilot_200step_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = _build_manifest(args.output_dir, args.save_steps)
    (args.output_dir / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = args.output_dir / "README.md"
    readme.write_text(
        "# Phase 2.5c 200-Step GRPO Pilot\n\n"
        f"- pilot_passed: **{passed}**\n"
        f"- early_stop: **{early_stop}**\n"
        f"- train groups: **50**, heldout groups: **20**\n"
        f"- checkpoint_label: **{PILOT_CHECKPOINT_LABEL}**\n\n"
        "See `train_vs_heldout_curve.md` and `periodic_eval_summary.json`.\n",
        encoding="utf-8",
    )

    print("\n=== Phase 2.5c Pilot Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not passed:
        _write_failure(args.output_dir, early_stop_reason or "acceptance criteria not met")
        raise SystemExit("200-step pilot did not pass acceptance criteria")


def _build_manifest(output_dir: Path, save_steps: List[int]) -> Dict[str, Any]:
    checkpoints = []
    for step in save_steps:
        path = output_dir / "checkpoints" / f"{PILOT_CHECKPOINT_PREFIX}_{step}"
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
        f"# Phase 2.5c Pilot Failure\n\n{reason}\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2.5c 200-step GRPO pilot")
    parser.add_argument("--dry-config-check", action="store_true")
    parser.add_argument(
        "--train-clean-path",
        type=Path,
        default=PHASE25_DIR / "train_clean_50_groups.jsonl",
    )
    parser.add_argument(
        "--heldout-clean-path",
        type=Path,
        default=PHASE25_DIR / "heldout_clean_20_groups.jsonl",
    )
    parser.add_argument(
        "--clean-set-summary-path",
        type=Path,
        default=PHASE25_DIR / "expanded_clean_set_summary.json",
    )
    parser.add_argument(
        "--preflight-rollout-dir",
        type=Path,
        default=PHASE25_DIR / "preflight_v2_rollout_50_g4",
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
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reward-candidate", type=str, default="reward_largek_mix_1000")
    parser.add_argument("--max-update-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--train-batch-size", type=int, default=50)
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=50)
    parser.add_argument("--micro-batch-size", type=int, default=5)
    parser.add_argument("--save-steps", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument(
        "--eval-steps", type=int, nargs="+", default=[0, 25, 50, 100, 150, 200]
    )
    parser.add_argument("--phase", type=str, default="2.5c")
    parser.add_argument("--mode", type=str, default="200step_grpo_pilot")
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--max-total-length", type=int, default=3072)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "Rec-R1/data/esci/inst/sparse/subset/val.parquet",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-preflight-build", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-gpu", type=int, default=0)
    parser.add_argument("--eval-gpu", type=int, default=1)
    parser.add_argument("--min-disk-gb", type=float, default=80.0)
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
