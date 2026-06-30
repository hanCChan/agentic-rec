#!/usr/bin/env python3
"""
Phase 2.3: Compare LR sweep runs (1e-6 vs 5e-7).

Usage:
  python scripts/compare_grpo_lr_sweep.py \
    --run-a experiments/phase23_10step_grpo_controlled_smoke/lr_1e-6 \
    --run-b experiments/phase23_10step_grpo_controlled_smoke/lr_5e-7 \
    --output-dir experiments/phase23_10step_grpo_controlled_smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from src.agents.grpo_curve_analyzer import GRPOCurveAnalyzer


def _load_run(run_dir: Path) -> dict:
    summary_path = run_dir / "ten_step_train_summary.json"
    curve_path = run_dir / "curve_analysis.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    curve = {}
    if curve_path.exists():
        curve = json.loads(curve_path.read_text(encoding="utf-8"))
    summary["curve_analysis"] = curve
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GRPO LR sweep runs")
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/phase23_10step_grpo_controlled_smoke",
    )
    args = parser.parse_args()

    run_a = _load_run(args.run_a)
    run_b = _load_run(args.run_b)

    analyzer = GRPOCurveAnalyzer()
    comparison = analyzer.compare_lr_runs(run_a, run_b)

    lines = [
        "# Phase 2.3 LR Sweep Comparison",
        "",
        f"- run_a (lr={comparison['run_a_learning_rate']}): "
        f"passed={comparison['run_a_ten_step_smoke_passed']}, "
        f"class={comparison['run_a_stability_class']}, "
        f"max_kl={comparison['run_a_max_approx_kl']}",
        f"- run_b (lr={comparison['run_b_learning_rate']}): "
        f"passed={comparison['run_b_ten_step_smoke_passed']}, "
        f"class={comparison['run_b_stability_class']}, "
        f"max_kl={comparison['run_b_max_approx_kl']}",
        "",
        f"- recommended_learning_rate: **{comparison['recommended_learning_rate']}**",
        f"- both_stable: **{comparison['both_stable']}**",
        "",
    ]

    if comparison["both_stable"]:
        lines.append("Next: Phase 2.4 — write 50-step pilot plan (do not launch yet).")
    elif comparison["run_b_ten_step_smoke_passed"] and not comparison["run_a_ten_step_smoke_passed"]:
        lines.append("Next: Phase 2.3b — adopt conservative LR=5e-7.")
    elif not comparison["run_a_ten_step_smoke_passed"]:
        lines.append("Next: Phase 2.3c — training stability fix.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lr_sweep_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "lr_sweep_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "lr_1e-6_completed": args.run_a.name == "lr_1e-6" or "1e-6" in str(args.run_a),
        "lr_5e-7_completed": args.run_b.exists() and (args.run_b / "ten_step_train_summary.json").exists(),
        "lr_1e-6_passed": run_a.get("ten_step_smoke_passed"),
        "lr_5e-7_passed": run_b.get("ten_step_smoke_passed"),
        "comparison": comparison,
    }
    if not summary["lr_5e-7_completed"]:
        summary["reason"] = "5e-7 comparison pending or not run."

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
