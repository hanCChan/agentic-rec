#!/usr/bin/env python3
"""
Phase 1.19b: Scale gate check for large-K quality reward.

Orchestrates strategy rollout, qrels/metric blindness, large-K reward dry-run,
and optional GRPO loss dry-run at 10_g4 / 20_g4 scales.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  CUDA_VISIBLE_DEVICES=2 python scripts/run_scale_gate_check.py \
    --scales 10 20 \
    --group-size 4 \
    --model-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
    --output-root experiments/phase119b_scale_gate_check
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.scale_gate_check import (
    ScaleGateCheck,
    build_scale_gate_comparison_md,
    build_scale_gate_recommendations_md,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.19b scale gate check orchestration")
    parser.add_argument("--scales", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "Rec-R1/data/esci/inst/sparse/subset_smoke/val.parquet",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data1/hcc/.hf_home/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/data1/hcc/.hf_home/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "experiments/phase119b_scale_gate_check",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--metric-k-list", type=int, nargs="+", default=[10, 50, 100, 1000])
    parser.add_argument("--candidate-name", type=str, default="reward_largek_mix_1000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-visible-devices", type=str, default="2")
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--skip-loss-dryrun", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: List[str], env: Dict[str, str], cwd: Path) -> None:
    print(f"[phase119b] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed (exit {result.returncode}): {' '.join(cmd)}")


def run_scale_pipeline(
    scale: int,
    args: argparse.Namespace,
    checker: ScaleGateCheck,
    env: Dict[str, str],
) -> Dict[str, Any]:
    group_size = args.group_size
    scale_tag = f"{scale}_g{group_size}"
    output_root = args.output_root

    rollout_dir = output_root / f"strategy_rollout_{scale_tag}"
    qrels_dir = output_root / f"qrels_metric_{scale_tag}"
    large_k_dir = output_root / f"large_k_reward_{scale_tag}"
    grpo_dir = output_root / f"grpo_loss_{scale_tag}_largek1000"

    rollout_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_rollout:
        rollout_cmd = [
            sys.executable,
            str(ROOT / "scripts/smoke_strategy_multisample_rollout.py"),
            "--data",
            str(args.data),
            "--num-base-records",
            str(scale),
            "--group-size",
            str(group_size),
            "--max-steps",
            str(args.max_steps),
            "--model-path",
            str(args.model_path),
            "--tokenizer-path",
            args.tokenizer_path,
            "--output-dir",
            str(rollout_dir),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--topk",
            str(args.topk),
            "--seed",
            str(args.seed),
        ]
        run_cmd(rollout_cmd, env=env, cwd=ROOT)

    rollout_summary_path = rollout_dir / "summary.json"
    if not rollout_summary_path.exists():
        return {
            "scale": scale_tag,
            "completed": False,
            "error": "strategy rollout summary missing",
        }

    rollout_summary = json.loads(rollout_summary_path.read_text(encoding="utf-8"))
    rollout_path = rollout_dir / "rollout_records.jsonl"
    group_summary_path = rollout_dir / "group_summaries.jsonl"

    qrels_cmd = [
        sys.executable,
        str(ROOT / "scripts/analyze_qrels_metric_blindness.py"),
        "--rollout-path",
        str(rollout_path),
        "--group-summary-path",
        str(group_summary_path),
        "--output-dir",
        str(qrels_dir),
        "--k-list",
        *[str(k) for k in args.metric_k_list],
    ]
    run_cmd(qrels_cmd, env=env, cwd=ROOT)

    large_k_cmd = [
        sys.executable,
        str(ROOT / "scripts/dryrun_large_k_reward.py"),
        "--rollout-path",
        str(rollout_path),
        "--metric-by-k-path",
        str(qrels_dir / "metric_by_k_diagnostics.jsonl"),
        "--group-metric-spread-path",
        str(qrels_dir / "group_metric_spread_by_k.jsonl"),
        "--output-dir",
        str(large_k_dir),
    ]
    run_cmd(large_k_cmd, env=env, cwd=ROOT)

    large_k_summary = checker.load_large_k_summary(large_k_dir / "summary.json")
    comparison = checker.load_large_k_comparison(large_k_dir / "large_k_candidate_comparison.json")
    gate_eval = checker.evaluate_scale_gate(large_k_summary, comparison, args.candidate_name)

    grpo_loss_summary: Optional[Dict[str, Any]] = None
    loss_dryrun_skipped = True
    skip_reason: Optional[str] = None

    if gate_eval["gate_passed"] and not args.skip_loss_dryrun:
        loss_cmd = [
            sys.executable,
            str(ROOT / "scripts/smoke_real_grpo_loss_dryrun.py"),
            "--rollout-path",
            str(rollout_path),
            "--shaped-reward-path",
            str(large_k_dir / "large_k_shaped_record_rewards.jsonl"),
            "--candidate-name",
            args.candidate_name,
            "--tokenizer-path",
            args.tokenizer_path,
            "--output-dir",
            str(grpo_dir),
            "--synthetic-logprob-delta",
            "0.02",
            "--cliprange",
            "0.2",
            "--kl-coef",
            "0.01",
            "--loss-agg-mode",
            "token-mean",
            "--max-prompt-length",
            "1024",
            "--max-response-length",
            "2048",
            "--max-total-length",
            "3072",
        ]
        run_cmd(loss_cmd, env=env, cwd=ROOT)
        grpo_loss_summary = json.loads((grpo_dir / "summary.json").read_text(encoding="utf-8"))
        loss_dryrun_skipped = False
    elif not gate_eval["gate_passed"]:
        skip_reason = "Large-K gate failed at this scale."

    report = checker.build_scale_report(
        scale=scale,
        group_size=group_size,
        rollout_summary=rollout_summary,
        large_k_summary=large_k_summary,
        comparison=comparison,
        qrels_group_spread_path=qrels_dir / "group_metric_spread_by_k.jsonl",
        grpo_loss_summary=grpo_loss_summary,
        loss_dryrun_skipped=loss_dryrun_skipped,
        skip_reason=skip_reason,
    )
    report["completed"] = True
    report["paths"] = {
        "rollout_dir": str(rollout_dir),
        "qrels_dir": str(qrels_dir),
        "large_k_dir": str(large_k_dir),
        "grpo_dir": str(grpo_dir) if not loss_dryrun_skipped else None,
    }
    return report


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    checker = ScaleGateCheck()
    scale_reports: List[Dict[str, Any]] = []

    existing_reports: Dict[str, Dict[str, Any]] = {}
    reports_path = args.output_root / "scale_gate_reports.jsonl"
    if reports_path.exists():
        for line in reports_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing_reports[row["scale"]] = row

    for scale in args.scales:
        print(f"\n[phase119b] ===== Scale {scale}_g{args.group_size} =====")
        try:
            report = run_scale_pipeline(scale, args, checker, env)
            scale_reports.append(report)
            print(
                f"[phase119b] {report['scale']} gate_passed={report.get('gate_passed')} "
                f"zero_std={report.get('zero_std_group_rate')}"
            )
        except Exception as exc:
            print(f"[phase119b] ERROR scale {scale}_g{args.group_size}: {exc}")
            scale_reports.append(
                {
                    "scale": f"{scale}_g{args.group_size}",
                    "completed": False,
                    "error": str(exc),
                    "gate_passed": False,
                    "loss_dryrun_skipped": True,
                    "skip_reason": str(exc),
                }
            )

    for scale_tag, report in existing_reports.items():
        if not any(r.get("scale") == scale_tag for r in scale_reports):
            report["completed"] = report.get("completed", True)
            scale_reports.append(report)

    scale_reports.sort(key=lambda r: int(r["scale"].split("_")[0]))

    baseline = checker.load_baseline_5_g4(args.candidate_name)
    comparison = checker.compare_scales(scale_reports, baseline, args.candidate_name)
    recommendation = checker.recommend_next_step(comparison)

    with (args.output_root / "scale_gate_reports.jsonl").open("w", encoding="utf-8") as fout:
        for report in scale_reports:
            slim = {k: v for k, v in report.items() if k != "paths"}
            fout.write(json.dumps(slim, ensure_ascii=False) + "\n")

    comparison_payload = {**comparison, **recommendation}
    (args.output_root / "scale_gate_comparison.json").write_text(
        json.dumps(comparison_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_root / "scale_gate_comparison.md").write_text(
        build_scale_gate_comparison_md(comparison, recommendation),
        encoding="utf-8",
    )
    (args.output_root / "scale_gate_recommendations.md").write_text(
        build_scale_gate_recommendations_md(scale_reports, comparison, recommendation),
        encoding="utf-8",
    )

    summary = {
        "phase": "1.19b",
        "candidate_name": args.candidate_name,
        "scales_requested": args.scales,
        "group_size": args.group_size,
        "baseline_5_g4": baseline if baseline.get("available") else None,
        "scale_reports": [{k: v for k, v in r.items() if k != "paths"} for r in scale_reports],
        "stable_gate_passed": comparison["stable_gate_passed"],
        "safe_for_phase_120": comparison["safe_for_phase_120"],
        "next_phase": recommendation["next_phase"],
        "main_conclusion": recommendation["main_conclusion"],
        "is_training": False,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = args.output_root / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Phase 1.19b Scale Gate Check\n\n"
            "Strategy rollout + qrels/metric + large-K reward gate validation at 10_g4 / 20_g4.\n",
            encoding="utf-8",
        )

    print("\n=== Phase 1.19b Scale Gate Check Summary ===")
    print(f"candidate_name: {args.candidate_name}")
    print(f"stable_gate_passed: {comparison['stable_gate_passed']}")
    print(f"safe_for_phase_120: {comparison['safe_for_phase_120']}")
    for report in scale_reports:
        if not report.get("completed", True):
            print(f"{report['scale']}: NOT COMPLETED — {report.get('error')}")
            continue
        print(
            f"{report['scale']}: gate={report.get('gate_passed')} "
            f"zero_std={report.get('zero_std_group_rate'):.2f} "
            f"retrieval_spread={report.get('retrieval_quality_spread_group_rate'):.2f} "
            f"loss_check={report.get('loss_check_passed')}"
        )
    print(f"next_phase: {recommendation['next_phase']}")
    print(f"output_root: {args.output_root}")


if __name__ == "__main__":
    main()
