#!/usr/bin/env python3
"""
Phase 2.2: 3-step GRPO stability smoke training.

Reuses Phase 2.1 clean set + preflight rollout. Validates consecutive
optimizer.step stability with non-negative KL diagnostics.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  CUDA_VISIBLE_DEVICES=2 python scripts/run_3step_grpo_stability_smoke.py \
    --output-dir experiments/phase22_3step_grpo_stability_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
REC_R1 = ROOT / "Rec-R1"
PHASE21_DIR = ROOT / "experiments/phase21_tiny_grpo_smoke"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REC_R1))

from src.agents.grpo_stability_monitor import GRPOStabilityMonitor
from src.agents.phase2_smoke_dataset import load_clean_set_rows
from src.agents.tiny_grpo_smoke_trainer import (
    CHECKPOINT_LABEL,
    TinyGrpoSmokeTrainer,
)


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
    group_ids = [r["group_id"] for r in rows]
    if "esci_val_3" in group_ids:
        raise ValueError("esci_val_3 must not be in clean set")
    return summary


def run_post_train_eval(
    *,
    clean_set_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    data_path: Path,
    preflight_rollout_dir: Path,
    candidate_name: str,
    temperature: float,
    top_p: float,
    topk: int,
    seed: int,
    output_name: str,
) -> Dict[str, Any]:
    import importlib.util
    from statistics import mean, pstdev

    script_path = ROOT / "scripts/smoke_strategy_prompt_v2.py"
    spec = importlib.util.spec_from_file_location("smoke_strategy_prompt_v2", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script_path}")
    v2_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2_module)

    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    candidate_rows = _load_jsonl(clean_set_path)
    preflight_records = _load_jsonl(preflight_rollout_dir / "v2_rollout_records.jsonl")
    old_by_group = v2_module._index_old_rollout(preflight_records)
    samples = v2_module.build_samples(candidate_rows, old_by_group, data_path)

    eval_dir = output_dir / f"post_eval_{output_name}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    groups, rollout_records, failures = v2_module.run_v2_rollout(
        samples,
        model_path=checkpoint_path,
        temperature=temperature,
        top_p=top_p,
        max_tokens=256,
        max_steps=3,
        topk=topk,
        seed=seed + 1,
        strategies=["exact_match", "attribute_expansion", "broad_recall", "constraint_preserving"],
    )

    rollout_path = eval_dir / "post_train_rollout_records.jsonl"
    v2_module._write_jsonl(rollout_path, rollout_records)
    v2_module.run_post_analysis(
        rollout_path,
        eval_dir,
        k_list=[10, 50, 100, 1000],
        candidate_name=candidate_name,
    )

    parse_rates = [float(g["group_metrics"].get("json_parse_success_rate", 0.0)) for g in groups]
    finish_rates = [float(g["group_metrics"].get("finish_rate", 0.0)) for g in groups]
    invalid_rates = [float(g["group_metrics"].get("invalid_action_rate", 0.0)) for g in groups]

    shaped_path = eval_dir / "large_k_shaped_record_rewards.jsonl"
    shaped_rows = _load_jsonl(shaped_path)
    rewards = [float(r[candidate_name]) for r in shaped_rows if candidate_name in r]
    ndcg_vals = [float(r.get("ndcg@1000", 0.0)) for r in shaped_rows]
    recall_vals = [float(r.get("recall@1000", 0.0)) for r in shaped_rows]
    mrr_vals = [float(r.get("mrr@1000", 0.0)) for r in shaped_rows]

    grouped_rewards: Dict[str, List[float]] = {}
    shaped_by_id = {r["sample_id"]: float(r[candidate_name]) for r in shaped_rows}
    for record in rollout_records:
        sid = record.get("sample_id")
        gid = record.get("group_id")
        if sid in shaped_by_id:
            grouped_rewards.setdefault(gid, []).append(shaped_by_id[sid])

    zero_std_groups = sum(
        1 for vals in grouped_rewards.values() if len(vals) > 1 and pstdev(vals) <= 1e-6
    )
    spread_groups = sum(
        1 for vals in grouped_rewards.values() if len(vals) > 1 and (max(vals) - min(vals)) > 1e-6
    )
    num_groups = len(grouped_rewards) or 1
    parse_success_rate = float(mean(parse_rates)) if parse_rates else 0.0

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_label": CHECKPOINT_LABEL,
        "num_groups": len(groups),
        "num_rollout_records": len(rollout_records),
        "num_failures": len(failures),
        "parse_success_rate": parse_success_rate,
        "finish_rate": float(mean(finish_rates)) if finish_rates else 0.0,
        "invalid_action_rate": float(mean(invalid_rates)) if invalid_rates else 0.0,
        "zero_std_group_rate": zero_std_groups / num_groups,
        "retrieval_quality_spread_group_rate": spread_groups / num_groups,
        "mean_reward_largek_mix_1000": float(mean(rewards)) if rewards else 0.0,
        "mean_ndcg1000": float(mean(ndcg_vals)) if ndcg_vals else 0.0,
        "mean_recall1000": float(mean(recall_vals)) if recall_vals else 0.0,
        "mean_mrr1000": float(mean(mrr_vals)) if mrr_vals else 0.0,
        "json_format_ok": parse_success_rate >= 0.95,
        "eval_passed": len(failures) == 0 and parse_success_rate >= 0.95,
    }
    return summary


def build_stability_report(
    summary: Dict[str, Any],
    monitor_report: Dict[str, Any],
    step_metrics: List[Dict[str, Any]],
    post_1step: Optional[Dict[str, Any]],
    post_3step: Optional[Dict[str, Any]],
) -> str:
    lines = [
        "# Phase 2.2 Three-Step GRPO Stability Smoke Report",
        "",
        "## Mode",
        "",
        f"- phase: `{summary.get('phase')}`",
        f"- mode: `{summary.get('mode')}`",
        f"- reward_candidate: `{summary.get('reward_candidate')}`",
        f"- checkpoint_label: **{summary.get('checkpoint_label')}**",
        "",
        "## Training Summary",
        "",
        f"- max_update_steps: **{summary.get('max_update_steps')}**",
        f"- actual_update_steps: **{summary.get('actual_update_steps')}**",
        f"- optimizer_steps_called: **{summary.get('optimizer_steps_called')}**",
        f"- three_step_smoke_passed: **{summary.get('three_step_smoke_passed')}**",
        f"- nan_detected: **{summary.get('nan_detected')}**",
        f"- oom_detected: **{summary.get('oom_detected')}**",
        f"- kl_exploded: **{summary.get('kl_exploded')}**",
        f"- max_approx_kl_nonnegative: **{summary.get('max_approx_kl_nonnegative')}**",
        f"- max_grad_norm: **{summary.get('max_grad_norm')}**",
        f"- max_abs_signed_logprob_gap: **{summary.get('max_abs_signed_logprob_gap')}**",
        "",
        "## Per-Step Metrics",
        "",
        "| step | policy_loss | approx_kl | signed_gap | clipfrac | grad_norm |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for m in step_metrics:
        lines.append(
            f"| {m.get('step')} | {m.get('policy_loss')} | "
            f"{m.get('approx_kl_nonnegative')} | {m.get('signed_logprob_gap_mean')} | "
            f"{m.get('clipfrac')} | {m.get('grad_norm')} |"
        )

    if post_1step and post_3step:
        lines.extend(
            [
                "",
                "## Post-Train Eval Comparison",
                "",
                f"- 1-step parse_success_rate: **{post_1step.get('parse_success_rate')}**",
                f"- 3-step parse_success_rate: **{post_3step.get('parse_success_rate')}**",
                f"- 1-step mean_reward: **{post_1step.get('mean_reward_largek_mix_1000')}**",
                f"- 3-step mean_reward: **{post_3step.get('mean_reward_largek_mix_1000')}**",
                f"- 3-step json_format_ok: **{post_3step.get('json_format_ok')}**",
            ]
        )

    if monitor_report.get("warnings"):
        lines.extend(["", "## Monitor Warnings", ""])
        for w in monitor_report["warnings"]:
            lines.append(f"- {w}")

    lines.extend(
        [
            "",
            summary.get("dryrun_warning", ""),
            "",
            "## Next Steps",
            "",
            summary.get("next_recommendation", ""),
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2.2 3-step GRPO stability smoke")
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
        "--phase21-post-eval-path",
        type=Path,
        default=PHASE21_DIR / "post_train_eval_summary.json",
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
        default=ROOT / "experiments/phase22_3step_grpo_stability_smoke",
    )
    parser.add_argument("--reward-candidate", type=str, default="reward_largek_mix_1000")
    parser.add_argument("--max-update-steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--train-batch-size", type=int, default=20)
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=20)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--max-total-length", type=int, default=3072)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "Rec-R1/data/esci/inst/sparse/subset/val.parquet",
    )
    parser.add_argument("--skip-post-eval", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean_summary = validate_clean_set(args.clean_set_path, args.clean_set_summary_path)

    rollout_path = args.preflight_rollout_dir / "v2_rollout_records.jsonl"
    shaped_path = args.preflight_rollout_dir / "large_k_shaped_record_rewards.jsonl"
    if not rollout_path.exists() or not shaped_path.exists():
        raise SystemExit(f"missing preflight artifacts under {args.preflight_rollout_dir}")

    if args.phase21_post_eval_path.exists():
        post_1step = _load_json(args.phase21_post_eval_path)
        (args.output_dir / "post_1step_eval_summary.json").write_text(
            json.dumps(post_1step, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    else:
        post_1step = None

    monitor = GRPOStabilityMonitor(
        max_signed_logprob_gap_abs=5.0,
        max_approx_kl=0.2,
        max_grad_norm=10.0,
    )

    trainer = TinyGrpoSmokeTrainer(
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
        result = trainer.run(
            rollout_path=rollout_path,
            shaped_reward_path=shaped_path,
            output_dir=args.output_dir,
            max_prompt_length=args.max_prompt_length,
            max_response_length=args.max_response_length,
            max_total_length=args.max_total_length,
            metrics_filename="three_step_train_metrics.jsonl",
            summary_filename="three_step_train_summary.json",
            config_filename="three_step_train_config.yaml",
            phase="2.2",
            mode="3step_grpo_stability_smoke",
            stability_monitor=monitor,
        )
    except Exception as exc:
        failure = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        (args.output_dir / "failure_report.md").write_text(
            "# Phase 2.2 Failure Report\n\n```json\n"
            + json.dumps(failure, ensure_ascii=False, indent=2)
            + "\n```\n",
            encoding="utf-8",
        )
        raise SystemExit(1) from exc

    step_metrics = result["step_metrics"]
    monitor_report = monitor.summarize(step_metrics)
    (args.output_dir / "stability_monitor_report.json").write_text(
        json.dumps(monitor_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    base_summary = result["summary"]
    three_step_summary = {
        **base_summary,
        "optimizer_steps_called": monitor_report["optimizer_steps_called"],
        "loss_finite_all_steps": monitor_report["loss_finite_all_steps"],
        "grad_norm_finite_all_steps": monitor_report["grad_norm_finite_all_steps"],
        "max_grad_norm": monitor_report["max_grad_norm"],
        "max_approx_kl_nonnegative": monitor_report["max_approx_kl_nonnegative"],
        "max_abs_signed_logprob_gap": monitor_report["max_abs_signed_logprob_gap"],
        "max_clipfrac": monitor_report["max_clipfrac"],
        "three_step_smoke_passed": False,
        "next_recommendation": "Run Phase 2.2b stability fix (lr=5e-7, higher kl_coef).",
    }

    post_3step: Optional[Dict[str, Any]] = None
    if not args.skip_post_eval and base_summary.get("checkpoint_path"):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            post_3step = run_post_train_eval(
                clean_set_path=args.clean_set_path,
                checkpoint_path=Path(base_summary["checkpoint_path"]),
                output_dir=args.output_dir,
                data_path=args.data_path,
                preflight_rollout_dir=args.preflight_rollout_dir,
                candidate_name=args.reward_candidate,
                temperature=args.temperature,
                top_p=args.top_p,
                topk=args.topk,
                seed=args.seed,
                output_name="3step",
            )
            (args.output_dir / "post_3step_eval_summary.json").write_text(
                json.dumps(post_3step, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            post_3step = {"eval_passed": False, "json_format_ok": False, "error": str(exc)}
            (args.output_dir / "post_3step_eval_summary.json").write_text(
                json.dumps(post_3step, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    eval_ok = post_3step is None or (
        post_3step.get("json_format_ok") is True
        and float(post_3step.get("parse_success_rate", 0.0)) >= 0.95
        and float(post_3step.get("invalid_action_rate", 1.0)) <= 0.05
    )

    passed = (
        monitor_report["optimizer_steps_called"] == args.max_update_steps
        and monitor_report["all_steps_passed"]
        and not monitor_report["nan_detected"]
        and not monitor_report["oom_detected"]
        and not monitor_report["kl_exploded"]
        and monitor_report["loss_finite_all_steps"]
        and monitor_report["grad_norm_finite_all_steps"]
        and monitor_report["max_approx_kl_nonnegative"] <= 0.2
        and monitor_report["max_grad_norm"] <= 10.0
        and eval_ok
    )

    if passed:
        three_step_summary["three_step_smoke_passed"] = True
        three_step_summary["next_recommendation"] = (
            "Run 10-step controlled smoke only after reviewing stability and post-train JSON metrics."
        )
    elif post_3step and not post_3step.get("json_format_ok", True):
        three_step_summary["next_recommendation"] = (
            "Phase 2.2c: Format preservation fix (lower lr, stronger KL, optional JSON SFT mix)."
        )

    three_step_summary["checkpoint_label"] = CHECKPOINT_LABEL
    (args.output_dir / "three_step_train_summary.json").write_text(
        json.dumps(three_step_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = build_stability_report(
        three_step_summary,
        monitor_report,
        step_metrics,
        post_1step,
        post_3step,
    )
    (args.output_dir / "three_step_grpo_stability_report.md").write_text(report, encoding="utf-8")

    readme = args.output_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Phase 2.2 Three-Step GRPO Stability Smoke\n\n"
            "Validates consecutive optimizer.step stability on clean 20_g4. "
            "Checkpoints are SMOKE_ONLY_DO_NOT_PROMOTE.\n",
            encoding="utf-8",
        )

    print("\n=== Phase 2.2 Three-Step GRPO Stability Summary ===")
    print(json.dumps(three_step_summary, ensure_ascii=False, indent=2))
    if post_3step:
        print("\n=== Post-Train Eval (3-step) ===")
        print(json.dumps(post_3step, ensure_ascii=False, indent=2))

    if not passed:
        (args.output_dir / "failure_report.md").write_text(
            "# Phase 2.2 Failure Report\n\n"
            f"three_step_smoke_passed=false\n\n"
            f"monitor_report:\n```json\n{json.dumps(monitor_report, ensure_ascii=False, indent=2)}\n```\n",
            encoding="utf-8",
        )
        raise SystemExit("3-step GRPO stability smoke did not pass acceptance criteria")


if __name__ == "__main__":
    main()
