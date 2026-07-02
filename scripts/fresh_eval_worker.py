#!/usr/bin/env python3
"""Isolated fresh-eval worker (separate process for dedicated eval GPUs)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.periodic_fresh_eval import run_fresh_eval  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one fresh eval split in an isolated GPU process")
    parser.add_argument("--clean-set-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-step", type=int, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--preflight-rollout-dir", type=Path, required=True)
    parser.add_argument("--candidate-name", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-split", type=str, default="train")
    parser.add_argument("--eval-root-name", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--summary-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_fresh_eval(
        clean_set_path=args.clean_set_path,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        eval_step=args.eval_step,
        data_path=args.data_path,
        preflight_rollout_dir=args.preflight_rollout_dir,
        candidate_name=args.candidate_name,
        model_path=args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        topk=args.topk,
        seed=args.seed,
        root=ROOT,
        eval_split=args.eval_split,
        eval_root_name=args.eval_root_name,
        eval_tensor_parallel_size=args.tensor_parallel_size,
        in_subprocess=True,
    )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
