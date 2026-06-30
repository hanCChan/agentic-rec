#!/usr/bin/env python3
"""
Phase 2.5c: 200-step GRPO pilot (skeleton — do not run until Phase 2.5a/b pass).

Usage (after expanded clean set ready):
  CUDA_VISIBLE_DEVICES=6,7 python scripts/run_200step_grpo_pilot.py \\
    --output-dir experiments/phase25_200step_grpo_pilot/lr_5e-7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments/phase25_200step_grpo_pilot"
PHASE25_DIR = ROOT / "experiments/phase25_expanded_clean_set"

sys.path.insert(0, str(ROOT))

from scripts.run_50step_grpo_pilot import DEFAULT_CONFIG, dry_config_check, run_pilot  # noqa: E402


DEFAULT_200_CONFIG = {
    **DEFAULT_CONFIG,
    "phase": "2.5c",
    "mode": "200step_grpo_pilot",
    "max_update_steps": 200,
    "save_steps": [50, 100, 200],
    "eval_steps": [0, 25, 50, 100, 200],
    "cuda_visible_devices": "6,7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2.5c 200-step GRPO pilot")
    parser.add_argument("--dry-config-check", action="store_true")
    parser.add_argument(
        "--clean-set-path",
        type=Path,
        default=PHASE25_DIR / "train_clean_50_groups.jsonl",
    )
    parser.add_argument(
        "--heldout-set-path",
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "lr_5e-7")
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-update-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=5e-7)
    parser.add_argument("--save-steps", type=int, nargs="+", default=[50, 100, 200])
    parser.add_argument("--eval-steps", type=int, nargs="+", default=[0, 25, 50, 100, 200])
    parser.add_argument("--train-batch-size", type=int, default=50)
    parser.add_argument("--train-gpu", type=int, default=0)
    parser.add_argument("--eval-gpu", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_config_check:
        args.output_dir = args.pilot_root
        dry_config_check(args)
        return

    summary_path = args.clean_set_summary_path
    if summary_path.exists():
        import json

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("expanded_clean_set_ready"):
            raise SystemExit(
                "Phase 2.5b expanded clean set not ready — run build_expanded_clean_set.py first"
            )

    run_pilot(args)


if __name__ == "__main__":
    main()
