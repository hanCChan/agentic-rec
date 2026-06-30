#!/usr/bin/env python3
"""
Phase 2.1 Step A: Build clean 20-group smoke candidate set.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  python scripts/build_phase2_clean_smoke_set.py \
    --candidate-v2-path experiments/phase118h_strategy_prompt_v2_20_g4/phase2_candidate_smoke_set_v2.jsonl \
    --replacement-candidates-path experiments/phase118g_bm25_failure_cleanup_20_g4/replacement_candidates.jsonl \
    --output-dir experiments/phase21_tiny_grpo_smoke \
    --target-groups 20 \
    --exclude-group-ids esci_val_3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.phase2_smoke_dataset import Phase2SmokeDatasetBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2.1 clean smoke set builder")
    parser.add_argument(
        "--candidate-v2-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--replacement-candidates-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/phase21_tiny_grpo_smoke",
    )
    parser.add_argument("--target-groups", type=int, default=20)
    parser.add_argument(
        "--exclude-group-ids",
        nargs="*",
        default=["esci_val_3"],
    )
    parser.add_argument(
        "--drop-group-ids",
        nargs="*",
        default=["esci_val_6"],
        help="Additional v2 groups to drop and backfill via replacement candidates",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    builder = Phase2SmokeDatasetBuilder(
        target_groups=args.target_groups,
        exclude_group_ids=args.exclude_group_ids,
        drop_group_ids=args.drop_group_ids,
    )

    from src.agents.phase2_smoke_dataset import _load_jsonl

    v2_rows = _load_jsonl(args.candidate_v2_path)
    replacement_rows = _load_jsonl(args.replacement_candidates_path)

    result = builder.build_clean_set(v2_rows, replacement_rows)
    paths = builder.write_outputs(
        result["rows"],
        result["validation"],
        args.output_dir,
    )

    print("\n=== Phase 2.1 Clean Smoke Set Summary ===")
    print(json.dumps(result["validation"], ensure_ascii=False, indent=2))
    print(f"output_jsonl: {paths['jsonl_path']}")
    print(f"output_summary: {paths['summary_path']}")

    if not result["validation"]["phase2_clean_set_ready"]:
        raise SystemExit("clean set validation failed")


if __name__ == "__main__":
    main()
