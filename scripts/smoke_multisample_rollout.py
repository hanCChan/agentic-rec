#!/usr/bin/env python3
"""
Phase 1.17: Real Multi-Sample Rollout Smoke.

Same query -> Qwen samples G trajectories -> real BM25 rewards -> GRPO group advantage.
No training, no GRPO trainer, no optimizer.step.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  CUDA_VISIBLE_DEVICES=2 python scripts/smoke_multisample_rollout.py \
    --num-base-records 2 \
    --group-size 4 \
    --model-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
    --output-dir experiments/phase117_multisample_rollout_2_g4 \
    --temperature 0.7 \
    --top-p 0.95 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
REC_R1 = ROOT / "Rec-R1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REC_R1))

from src.agents.commerce_agent_env import CommerceAgentEnv
from src.agents.dataproto_mock import DataProtoMock
from src.agents.episode_runner import load_esci_samples
from src.agents.grpo_advantage_mock import GRPOAdvantageMock
from src.agents.multisample_episode_runner import MultiSampleEpisodeRunner
from src.agents.qwen_rollout_policy import QwenRolloutPolicy
from src.agents.real_dataproto_adapter import RealDataProtoAdapter
from src.agents.verl_batch_builder import VerlBatchBuilder, check_batch_shapes
from src.agents.verl_training_field_builder import VerlTrainingFieldBuilder, check_training_fields
from src.tools.bm25_tool import BM25SearchTool

DRYRUN_WARNING = (
    "Real multi-sample rollouts were collected for smoke testing only; "
    "no GRPO update was performed."
)


def attach_group_metadata(fields: Dict[str, Any], records: List[Dict[str, Any]], group_size: int) -> None:
    fields["group_ids"] = [r["group_id"] for r in records]
    fields["group_indices"] = [r["group_index"] for r in records]
    fields["group_size"] = [group_size] * len(records)


def aggregate_rollout_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {}

    def avg_metric(key: str) -> float:
        vals = [float(r["metrics"][key]) for r in records if r.get("metrics", {}).get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    total_policy_steps = sum(r["extra_info"].get("total_policy_steps", 0) for r in records)
    parse_ok_steps = sum(r["extra_info"].get("parse_ok_steps", 0) for r in records)
    total_invalid = sum(r["metrics"].get("num_invalid_actions", 0) for r in records)
    total_env_steps = sum(len(r["trajectory"].get("steps", [])) for r in records)

    return {
        "parse_success_rate": parse_ok_steps / total_policy_steps if total_policy_steps else 1.0,
        "invalid_action_rate": total_invalid / total_env_steps if total_env_steps else 0.0,
        "finish_rate": sum(1 for r in records if r["metrics"]["finished"]) / len(records),
        "llm_finish_rate": sum(1 for r in records if r["metrics"]["llm_finished"]) / len(records),
        "auto_finish_rate": sum(1 for r in records if r["metrics"]["auto_finished"]) / len(records),
        "avg_reward": sum(float(r["reward"]) for r in records) / len(records),
        "avg_ndcg_at_10": avg_metric("ndcg_at_10"),
        "avg_search_calls": avg_metric("num_search_calls"),
    }


def aggregate_group_stats(groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not groups:
        return {}

    metrics_list = [g["group_metrics"] for g in groups]
    num_groups = len(groups)
    zero_std_count = sum(1 for m in metrics_list if m.get("zero_std_reward"))

    def mean_key(key: str) -> float:
        return sum(float(m[key]) for m in metrics_list) / num_groups

    def rate_key(key: str) -> float:
        return sum(1 for m in metrics_list if m.get(key)) / num_groups

    return {
        "num_groups": num_groups,
        "zero_std_group_count": zero_std_count,
        "zero_std_group_rate": zero_std_count / num_groups,
        "mean_group_reward_std": mean_key("reward_std"),
        "mean_unique_final_query_count": mean_key("unique_final_query_count"),
        "mean_unique_trajectory_count": mean_key("unique_trajectory_count"),
        "all_same_final_query_rate": rate_key("all_same_final_query"),
        "all_same_trajectory_rate": rate_key("all_same_trajectory"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.17 real multi-sample rollout smoke")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "Rec-R1/data/esci/inst/sparse/subset_smoke/val.parquet",
    )
    parser.add_argument("--num-base-records", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--topk", type=int, default=20)
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--max-total-length", type=int, default=3072)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_esci_samples(args.data, args.num_base_records)
    print(f"[phase117] loaded {len(samples)} base samples from {args.data}")
    print(f"[phase117] model={args.model_path} group_size={args.group_size}")

    def env_factory() -> CommerceAgentEnv:
        search_tool = BM25SearchTool(rec_r1_root=REC_R1)
        return CommerceAgentEnv(
            search_tool=search_tool,
            max_steps=args.max_steps,
            default_topk=args.topk,
        )

    policy = QwenRolloutPolicy(
        model_path=str(args.model_path),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    runner = MultiSampleEpisodeRunner(
        env_factory=env_factory,
        policy=policy,
        group_size=args.group_size,
        max_steps=args.max_steps,
        topk=args.topk,
        base_seed=args.seed,
        sampling_temperature=args.temperature,
        sampling_top_p=args.top_p,
    )

    groups: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for sample in samples:
        try:
            group = runner.run_group(sample)
            groups.append(group)
            gm = group["group_metrics"]
            print(
                f"[phase117] {group['group_id']} "
                f"reward_std={gm['reward_std']:.4f} "
                f"unique_traj={gm['unique_trajectory_count']} "
                f"unique_final={gm['unique_final_query_count']}"
            )
        except Exception as exc:
            failures.append(
                {"group_id": sample.get("qid"), "error": str(exc), "trace": traceback.format_exc()}
            )
            print(f"[phase117] ERROR {sample.get('qid')}: {exc}")

    rollout_records: List[Dict[str, Any]] = []
    for group in groups:
        rollout_records.extend(group["records"])

    rollout_path = args.output_dir / "rollout_records.jsonl"
    with rollout_path.open("w", encoding="utf-8") as fout:
        for record in rollout_records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    group_summary_path = args.output_dir / "group_summaries.jsonl"
    with group_summary_path.open("w", encoding="utf-8") as fout:
        for group in groups:
            summary_row = {
                "group_id": group["group_id"],
                "original_query": group["original_query"],
                "group_size": group["group_size"],
                "group_metrics": group["group_metrics"],
            }
            fout.write(json.dumps(summary_row, ensure_ascii=False) + "\n")

    grpo_mock = GRPOAdvantageMock(
        group_size=args.group_size,
        normalize_by_std=True,
        synthetic_reward_jitter=0.0,
        seed=args.seed,
    )

    batch_builder = VerlBatchBuilder(
        tokenizer_path=args.tokenizer_path,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        max_total_length=args.max_total_length,
    )
    batch = batch_builder.build_batch(rollout_records)
    check_batch_shapes(batch)

    pad_token_id = batch_builder.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = batch_builder.tokenizer.eos_token_id

    field_builder = VerlTrainingFieldBuilder(pad_token_id=pad_token_id)
    fields = field_builder.build_training_fields(batch)
    check_training_fields(fields)
    attach_group_metadata(fields, rollout_records, args.group_size)

    mock_proto = DataProtoMock.from_fields(fields)
    mock_validate = mock_proto.validate()
    if not mock_validate["passed"]:
        raise SystemExit(f"DataProtoMock validate failed: {mock_validate['errors']}")

    adapter = RealDataProtoAdapter()
    convert_result = adapter.to_real_dataproto(mock_proto)
    data_proto = convert_result["real_proto"] if convert_result["used_real_dataproto"] else mock_proto
    used_real_dataproto = convert_result["used_real_dataproto"]

    adv_output = grpo_mock.compute_group_advantages(data_proto)
    check_result = grpo_mock.check_group_advantages(data_proto, adv_output)

    rollout_stats = aggregate_rollout_stats(rollout_records)
    group_stats = aggregate_group_stats(groups)

    summary: Dict[str, Any] = {
        "phase": "1.17",
        "num_base_records": len(samples),
        "group_size": args.group_size,
        "num_rollout_records": len(rollout_records),
        "used_real_multisample": True,
        "used_real_dataproto": used_real_dataproto,
        **rollout_stats,
        **group_stats,
        "mean_abs_sequence_advantage": check_result["mean_abs_sequence_advantage"],
        "advantage_check_passed": check_result["advantage_check_passed"],
        "padding_advantages_zero": check_result["padding_advantages_zero"],
        "is_training": False,
        "dryrun_warning": DRYRUN_WARNING,
        "failures": failures,
        "config": {
            "data": str(args.data),
            "model_path": str(args.model_path),
            "tokenizer_path": args.tokenizer_path,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "max_steps": args.max_steps,
            "topk": args.topk,
            "seed": args.seed,
        },
    }

    group_advantage_shapes = {
        "sequence_rewards": check_result["sequence_rewards_shape"],
        "sequence_advantages": check_result["sequence_advantages_shape"],
        "token_level_advantages": check_result["token_level_advantages_shape"],
    }
    group_advantage_stats = {
        "num_groups": check_result["num_groups"],
        "group_size": check_result["group_size"],
        "zero_std_group_count": check_result["zero_std_group_count"],
        "zero_std_group_rate": check_result["zero_std_group_rate"],
        "mean_group_reward_std": check_result["mean_group_reward_std"],
        "min_group_reward_std": check_result["min_group_reward_std"],
        "max_group_reward_std": check_result["max_group_reward_std"],
        "mean_abs_sequence_advantage": check_result["mean_abs_sequence_advantage"],
        "zero_advantage_token_rate": check_result["zero_advantage_token_rate"],
        "group_mean_advantage_close_to_zero": check_result["group_mean_advantage_close_to_zero"],
        "padding_advantages_zero": check_result["padding_advantages_zero"],
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "group_advantage_shapes.json").write_text(
        json.dumps(group_advantage_shapes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "group_advantage_stats.json").write_text(
        json.dumps(group_advantage_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = args.output_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Phase 1.17 Real Multi-Sample Rollout Smoke\n\n"
            "Real Qwen multi-sample rollouts with BM25 rewards and GRPO group advantages.\n"
            "No training was performed.\n",
            encoding="utf-8",
        )

    print("\n=== Phase 1.17 Real Multi-Sample Rollout Smoke Summary ===")
    print(f"num_base_records: {summary['num_base_records']}")
    print(f"group_size: {summary['group_size']}")
    print(f"num_rollout_records: {summary['num_rollout_records']}")
    print(f"parse_success_rate: {summary['parse_success_rate']:.4f}")
    print(f"invalid_action_rate: {summary['invalid_action_rate']:.4f}")
    print(f"finish_rate: {summary['finish_rate']:.4f}")
    print(f"avg_reward: {summary['avg_reward']:.4f}")
    print(f"avg_ndcg_at_10: {summary['avg_ndcg_at_10']:.4f}")
    print(f"zero_std_group_rate: {summary['zero_std_group_rate']:.4f}")
    print(f"mean_group_reward_std: {summary['mean_group_reward_std']:.6f}")
    print(f"mean_abs_sequence_advantage: {summary['mean_abs_sequence_advantage']:.4f}")
    print(f"mean_unique_final_query_count: {summary['mean_unique_final_query_count']:.4f}")
    print(f"mean_unique_trajectory_count: {summary['mean_unique_trajectory_count']:.4f}")
    print(f"advantage_check_passed: {summary['advantage_check_passed']}")
    print(f"output_dir: {args.output_dir}")

    if summary.get("zero_std_group_rate", 0.0) >= 1.0:
        print(
            "\n[phase117] NOTE: zero_std_group_rate=1.0 — real multi-sample rollouts may lack "
            "reward variance; consider tuning temperature / prompt / action diversity."
        )

    if not check_result["advantage_check_passed"]:
        raise SystemExit(f"advantage check failed: {check_result}")

    print(f"\n[phase117] {DRYRUN_WARNING}")
    print(f"[phase117] wrote {rollout_path}")
    print(f"[phase117] wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
