#!/usr/bin/env python3
"""
Phase 2.1: Tiny GRPO smoke training orchestrator.

Verifies preflight gate, runs 1-3 step real GRPO update, post-train eval, and reports.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  CUDA_VISIBLE_DEVICES=2 python scripts/run_tiny_grpo_smoke_training.py \
    --clean-set-path experiments/phase21_tiny_grpo_smoke/phase2_clean_20_groups.jsonl \
    --preflight-rollout-dir experiments/phase21_tiny_grpo_smoke/preflight_v2_rollout_20_g4 \
    --model-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
    --tokenizer-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
    --output-dir experiments/phase21_tiny_grpo_smoke \
    --max-update-steps 1
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
REC_R1 = ROOT / "Rec-R1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REC_R1))

from src.agents.phase2_smoke_dataset import load_clean_set_rows
from src.agents.tiny_grpo_smoke_trainer import (
    CHECKPOINT_LABEL,
    TinyGrpoSmokeTrainer,
    build_tiny_grpo_smoke_report,
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def check_preflight_gate(preflight_dir: Path) -> Dict[str, Any]:
    summary_path = preflight_dir / "summary.json"
    lk_summary_path = preflight_dir / "v2_large_k_reward_summary.json"
    strategy_summary_path = preflight_dir / "strategy_v2_summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"missing preflight summary: {summary_path}")

    summary = _load_json(summary_path)
    lk_summary = _load_json(lk_summary_path) if lk_summary_path.exists() else {}
    strategy_summary = _load_json(strategy_summary_path) if strategy_summary_path.exists() else {}

    v2_effect = strategy_summary.get("v2_effect", {})
    v2_strategy_collapse_count = int(v2_effect.get("strategy_collapse_count", summary.get("v2_strategy_collapse_count", 0)))

    spread_rate = float(
        summary.get("v2_retrieval_quality_spread_group_rate")
        or lk_summary.get("retrieval_quality_spread_group_rate")
        or 0.0
    )
    zero_std_rate = float(
        summary.get("v2_zero_std_group_rate")
        or lk_summary.get("zero_std_group_rate")
        or 1.0
    )
    penalty_only_rate = float(
        summary.get("v2_penalty_only_spread_group_rate")
        or lk_summary.get("penalty_only_spread_group_rate")
        or 0.0
    )
    v2_gate_passed = bool(summary.get("v2_gate_passed") or lk_summary.get("gate_passed"))

    gate_checks = {
        "v2_gate_passed": v2_gate_passed,
        "v2_retrieval_quality_spread_group_rate": spread_rate,
        "v2_zero_std_group_rate": zero_std_rate,
        "v2_penalty_only_spread_group_rate": penalty_only_rate,
        "v2_strategy_collapse_count": v2_strategy_collapse_count,
        "spread_rate_ok": spread_rate >= 0.80,
        "zero_std_ok": zero_std_rate <= 0.20,
        "penalty_only_ok": penalty_only_rate == 0.0,
        "collapse_ok": v2_strategy_collapse_count == 0,
    }
    gate_checks["preflight_passed"] = all(
        [
            gate_checks["v2_gate_passed"],
            gate_checks["spread_rate_ok"],
            gate_checks["zero_std_ok"],
            gate_checks["penalty_only_ok"],
            gate_checks["collapse_ok"],
        ]
    )
    return gate_checks


def write_preflight_failed_report(output_dir: Path, gate: Dict[str, Any]) -> None:
    lines = [
        "# Phase 2.1 Preflight Failed Report",
        "",
        "Preflight gate did not pass. Training was NOT started.",
        "",
        "## Gate Checks",
        "",
        "```json",
        json.dumps(gate, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Next Steps",
        "",
        "- Do not run tiny GRPO training until preflight passes.",
        "- Revisit sample replacement or prompt V2 if collapse or spread fails.",
    ]
    (output_dir / "preflight_failed_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
) -> Dict[str, Any]:
    import importlib.util

    script_path = ROOT / "scripts/smoke_strategy_prompt_v2.py"
    spec = importlib.util.spec_from_file_location("smoke_strategy_prompt_v2", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script_path}")
    v2_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2_module)

    build_samples = v2_module.build_samples
    run_post_analysis = v2_module.run_post_analysis
    run_v2_rollout = v2_module.run_v2_rollout
    _index_old_rollout = v2_module._index_old_rollout
    _write_jsonl = v2_module._write_jsonl

    eval_dir = output_dir / "post_train_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = _load_jsonl(clean_set_path)
    preflight_records = _load_jsonl(preflight_rollout_dir / "v2_rollout_records.jsonl")
    old_by_group = _index_old_rollout(preflight_records)
    samples = build_samples(candidate_rows, old_by_group, data_path)

    groups, rollout_records, failures = run_v2_rollout(
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
    _write_jsonl(rollout_path, rollout_records)

    post = run_post_analysis(
        rollout_path,
        eval_dir,
        k_list=[10, 50, 100, 1000],
        candidate_name=candidate_name,
    )

    parse_rates = []
    finish_rates = []
    invalid_rates = []
    for group in groups:
        gm = group.get("group_metrics", {})
        parse_rates.append(float(gm.get("json_parse_success_rate", 0.0)))
        finish_rates.append(float(gm.get("finish_rate", 0.0)))
        invalid_rates.append(float(gm.get("invalid_action_rate", 0.0)))

    shaped = post["shaped_by_sample"]
    rewards = list(shaped.values())
    ndcg_vals = []
    recall_vals = []
    mrr_vals = []
    for row in _load_jsonl(eval_dir / "large_k_shaped_record_rewards.jsonl"):
        ndcg_vals.append(float(row.get("ndcg@1000", 0.0)))
        recall_vals.append(float(row.get("recall@1000", 0.0)))
        mrr_vals.append(float(row.get("mrr@1000", 0.0)))

    grouped_rewards: Dict[str, List[float]] = {}
    for record in rollout_records:
        sid = record.get("sample_id")
        gid = record.get("group_id")
        if sid in shaped:
            grouped_rewards.setdefault(gid, []).append(float(shaped[sid]))

    zero_std_groups = sum(
        1
        for vals in grouped_rewards.values()
        if len(vals) > 1 and pstdev(vals) <= 1e-6
    )
    spread_groups = sum(
        1
        for vals in grouped_rewards.values()
        if len(vals) > 1 and (max(vals) - min(vals)) > 1e-6
    )
    num_groups = len(grouped_rewards) or 1

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_label": CHECKPOINT_LABEL,
        "num_groups": len(groups),
        "num_rollout_records": len(rollout_records),
        "num_failures": len(failures),
        "parse_success_rate": float(mean(parse_rates)) if parse_rates else 0.0,
        "finish_rate": float(mean(finish_rates)) if finish_rates else 0.0,
        "invalid_action_rate": float(mean(invalid_rates)) if invalid_rates else 0.0,
        "zero_std_group_rate": zero_std_groups / num_groups,
        "retrieval_quality_spread_group_rate": spread_groups / num_groups,
        "mean_reward_largek_mix_1000": float(mean(rewards)) if rewards else 0.0,
        "mean_ndcg1000": float(mean(ndcg_vals)) if ndcg_vals else 0.0,
        "mean_recall1000": float(mean(recall_vals)) if recall_vals else 0.0,
        "mean_mrr1000": float(mean(mrr_vals)) if mrr_vals else 0.0,
        "json_format_ok": (float(mean(parse_rates)) if parse_rates else 0.0) >= 0.95,
        "eval_passed": len(failures) == 0,
    }
    (output_dir / "post_train_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2.1 tiny GRPO smoke training")
    parser.add_argument(
        "--clean-set-path",
        type=Path,
        default=ROOT / "experiments/phase21_tiny_grpo_smoke/phase2_clean_20_groups.jsonl",
    )
    parser.add_argument(
        "--preflight-rollout-dir",
        type=Path,
        default=ROOT / "experiments/phase21_tiny_grpo_smoke/preflight_v2_rollout_20_g4",
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
        default=ROOT / "experiments/phase21_tiny_grpo_smoke",
    )
    parser.add_argument("--reward-candidate", type=str, default="reward_largek_mix_1000")
    parser.add_argument("--max-update-steps", type=int, default=1)
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
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-post-eval", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    clean_rows = load_clean_set_rows(args.clean_set_path)
    if len(clean_rows) != args.train_batch_size:
        raise SystemExit(
            f"clean set has {len(clean_rows)} groups, expected {args.train_batch_size}"
        )

    clean_summary_path = args.output_dir / "phase2_clean_set_summary.json"
    clean_summary = (
        _load_json(clean_summary_path) if clean_summary_path.exists() else None
    )

    gate = check_preflight_gate(args.preflight_rollout_dir)
    print("\n=== Phase 2.1 Preflight Gate ===")
    print(json.dumps(gate, ensure_ascii=False, indent=2))

    if not gate["preflight_passed"]:
        write_preflight_failed_report(args.output_dir, gate)
        raise SystemExit("preflight gate failed; see preflight_failed_report.md")

    rollout_path = args.preflight_rollout_dir / "v2_rollout_records.jsonl"
    shaped_path = args.preflight_rollout_dir / "large_k_shaped_record_rewards.jsonl"
    if not rollout_path.exists() or not shaped_path.exists():
        raise SystemExit(f"missing preflight rollout artifacts under {args.preflight_rollout_dir}")

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
        if args.skip_training:
            summary_path = args.output_dir / "tiny_train_summary.json"
            if not summary_path.exists():
                raise SystemExit("--skip-training requires existing tiny_train_summary.json")
            summary = _load_json(summary_path)
            result = {"summary": summary}
        else:
            result = trainer.run(
                rollout_path=rollout_path,
                shaped_reward_path=shaped_path,
                output_dir=args.output_dir,
                max_prompt_length=args.max_prompt_length,
                max_response_length=args.max_response_length,
                max_total_length=args.max_total_length,
            )
    except Exception as exc:
        print(f"[phase21] training failed: {exc}")
        traceback.print_exc()
        raise SystemExit(1) from exc

    summary = result["summary"]
    post_eval_summary: Optional[Dict[str, Any]] = None

    if summary.get("checkpoint_path") and not args.skip_post_eval:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            post_eval_summary = run_post_train_eval(
                clean_set_path=args.clean_set_path,
                checkpoint_path=Path(summary["checkpoint_path"]),
                output_dir=args.output_dir,
                data_path=args.data_path,
                preflight_rollout_dir=args.preflight_rollout_dir,
                candidate_name=args.reward_candidate,
                temperature=args.temperature,
                top_p=args.top_p,
                topk=args.topk,
                seed=args.seed,
            )
        except Exception as exc:
            print(f"[phase21] post-train eval failed: {exc}")
            traceback.print_exc()
            post_eval_summary = {"eval_passed": False, "error": str(exc)}

    preflight_summary = _load_json(args.preflight_rollout_dir / "summary.json")
    report = build_tiny_grpo_smoke_report(
        summary,
        clean_summary=clean_summary,
        preflight_summary={
            "v2_gate_passed": gate["v2_gate_passed"],
            "v2_retrieval_quality_spread_group_rate": gate["v2_retrieval_quality_spread_group_rate"],
            "v2_zero_std_group_rate": gate["v2_zero_std_group_rate"],
            "v2_strategy_collapse_count": gate["v2_strategy_collapse_count"],
        },
        post_eval_summary=post_eval_summary,
    )
    (args.output_dir / "tiny_grpo_smoke_report.md").write_text(report, encoding="utf-8")

    readme = args.output_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Phase 2.1 Tiny GRPO Smoke Training\n\n"
            "Minimal real GRPO update on clean 20_g4 strategy groups. "
            "Checkpoints are SMOKE_ONLY_DO_NOT_PROMOTE.\n",
            encoding="utf-8",
        )

    print("\n=== Phase 2.1 Tiny GRPO Smoke Training Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if post_eval_summary:
        print("\n=== Post-Train Eval ===")
        print(json.dumps(post_eval_summary, ensure_ascii=False, indent=2))

    if not summary.get("training_smoke_passed"):
        raise SystemExit("tiny GRPO smoke training did not pass stability checks")


if __name__ == "__main__":
    main()
