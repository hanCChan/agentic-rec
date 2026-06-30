#!/usr/bin/env python3
"""
Phase 2.5b: Build expanded train / held-out clean set from ESCI val rescan.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  CUDA_VISIBLE_DEVICES=6,7 python scripts/build_expanded_clean_set.py \
    --output-dir experiments/phase25_expanded_clean_set \
    --target-train-groups 50 \
    --target-heldout-groups 20 \
    --replacement-pool-size 300 \
    --group-size 4 \
    --model-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
    --tokenizer-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
    --temperature 0.7 \
    --top-p 0.95 \
    --topk 20 \
    --metric-k-list 10 50 100 1000 \
    --reward-candidate reward_largek_mix_1000 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REC_R1 = ROOT / "Rec-R1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REC_R1))

from src.agents.expanded_clean_set_builder import ExpandedCleanSetBuilder
from src.tools.bm25_tool import BM25SearchTool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2.5b expanded clean set builder")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/phase25_expanded_clean_set",
    )
    parser.add_argument("--target-train-groups", type=int, default=50)
    parser.add_argument("--target-heldout-groups", type=int, default=20)
    parser.add_argument("--replacement-pool-size", type=int, default=300)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "Rec-R1/data/esci/inst/sparse/subset/val.parquet",
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
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--metric-k-list", type=int, nargs="+", default=[10, 50, 100, 1000])
    parser.add_argument("--reward-candidate", type=str, default="reward_largek_mix_1000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-group-ids",
        nargs="*",
        default=["esci_val_3"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    builder = ExpandedCleanSetBuilder(
        target_train_groups=args.target_train_groups,
        target_heldout_groups=args.target_heldout_groups,
        replacement_pool_size=args.replacement_pool_size,
        group_size=args.group_size,
        exclude_group_ids=set(args.exclude_group_ids),
        k_list=args.metric_k_list,
        candidate_name=args.reward_candidate,
    )

    search_tool = BM25SearchTool(rec_r1_root=REC_R1)

    result = builder.build(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_path=args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        topk=args.topk,
        seed=args.seed,
        root=ROOT,
        search_tool=search_tool,
    )

    summary = result["summary"]
    print("\n=== Phase 2.5b Expanded Clean Set Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"output_dir: {args.output_dir}")

    if not summary.get("expanded_clean_set_ready"):
        print("\nWARNING: expanded clean set not ready — see why_not_enough_clean_groups.md if present")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
