#!/usr/bin/env python3
"""
Phase 1.18h: Strategy Prompt V2 for Collapse Cases.

Rerolls strategy-controlled groups with anti-collapse V2 prompts,
recomputes qrels/large-K gate, and exports phase2 candidate set v2.
No training.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  CUDA_VISIBLE_DEVICES=2 python scripts/smoke_strategy_prompt_v2.py \
    --target-collapse-only true \
    --output-dir experiments/phase118h_strategy_prompt_v2_targeted
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REC_R1 = ROOT / "Rec-R1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REC_R1))

from src.agents.commerce_agent_env import CommerceAgentEnv
from src.agents.large_k_reward_dryrun import GATE_THRESHOLDS, LargeKRewardDryRun
from src.agents.qrels_metric_blindness import QrelsMetricBlindness
from src.agents.qwen_rollout_policy import QwenRolloutPolicy
from src.agents.scale_gate_check import DEFAULT_GATE_THRESHOLDS, ScaleGateCheck
from src.agents.search_strategy_prompts import DEFAULT_STRATEGY_ORDER
from src.agents.search_strategy_prompts_v2 import DEFAULT_STRATEGY_ORDER_V2, get_strategy_v2, validate_strategies_v2
from src.agents.strategy_collapse_diagnostics import StrategyCollapseDiagnostics
from src.agents.strategy_episode_runner import StrategyEpisodeRunner
from src.tools.bm25_tool import BM25SearchTool

DRYRUN_WARNING = (
    "Strategy prompt V2 rerollout for collapse diagnostics only; no GRPO training was performed."
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def _bool_arg(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def _index_old_rollout(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for record in records:
        gid = record.get("group_id")
        if gid and gid not in out:
            out[gid] = record
    return out


def _load_esci_sample_by_qid(data_path: Path, qid: str) -> Optional[Dict[str, Any]]:
    try:
        idx = int(qid.rsplit("_", 1)[-1])
    except ValueError:
        return None
    df = pd.read_parquet(data_path)
    if idx < 0 or idx >= len(df):
        return None
    row = df.iloc[idx]
    return {
        "qid": qid,
        "user_query": str(row["query"]),
        "target_items": [str(x) for x in row["item_id"]],
    }


def build_samples(
    candidate_rows: List[Dict[str, Any]],
    old_rollout_by_group: Dict[str, Dict[str, Any]],
    data_path: Path,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for row in candidate_rows:
        gid = row["group_id"]
        query = row.get("original_query", "")
        if gid in old_rollout_by_group:
            traj = old_rollout_by_group[gid]["trajectory"]
            samples.append(
                {
                    "qid": gid,
                    "user_query": query or traj.get("user_query", ""),
                    "target_items": list(traj.get("target_items", [])),
                }
            )
            continue
        sample = _load_esci_sample_by_qid(data_path, gid)
        if sample is None:
            raise KeyError(f"cannot resolve sample for group_id={gid}")
        if query:
            sample["user_query"] = query
        samples.append(sample)
    return samples


def select_candidate_rows(
    cleanup_rows: List[Dict[str, Any]],
    phase2_rows: List[Dict[str, Any]],
    *,
    target_collapse_only: bool,
) -> List[Dict[str, Any]]:
    if target_collapse_only:
        collapse_ids = {
            r["group_id"] for r in cleanup_rows if r.get("learnability_type") == "strategy_collapse"
        }
        old_by_group = {r["group_id"]: r for r in cleanup_rows}
        return [
            {
                "group_id": gid,
                "original_query": old_by_group[gid]["original_query"],
                "source": "strategy_collapse_targeted",
                "recommended_for_phase2": False,
            }
            for gid in sorted(collapse_ids)
        ]
    return phase2_rows


def run_v2_rollout(
    samples: List[Dict[str, Any]],
    *,
    model_path: Path,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_steps: int,
    topk: int,
    seed: int,
    strategies: List[str],
    cuda_device: int | None = None,
    tensor_parallel_size: int = 1,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    def env_factory() -> CommerceAgentEnv:
        search_tool = BM25SearchTool(rec_r1_root=REC_R1)
        return CommerceAgentEnv(
            search_tool=search_tool,
            max_steps=max_steps,
            default_topk=topk,
        )

    policy = QwenRolloutPolicy(
        model_path=str(model_path),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        cuda_device=cuda_device,
        tensor_parallel_size=tensor_parallel_size,
    )
    runner = StrategyEpisodeRunner(
        env_factory=env_factory,
        policy=policy,
        strategies=strategies,
        max_steps=max_steps,
        topk=topk,
        base_seed=seed,
        sampling_temperature=temperature,
        sampling_top_p=top_p,
        strategy_getter=get_strategy_v2,
        strategy_version="v2",
    )

    groups: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for sample in samples:
        try:
            group = runner.run_group(sample)
            groups.append(group)
            gm = group["group_metrics"]
            print(
                f"[phase118h] {group['group_id']} "
                f"unique_final={gm['unique_strategy_final_query_count']} "
                f"jaccard={gm['avg_pairwise_final_query_jaccard']:.3f} "
                f"reward_std={gm['reward_std']:.4f}"
            )
        except Exception as exc:
            failures.append(
                {
                    "group_id": sample.get("qid"),
                    "error": str(exc),
                    "trace": traceback.format_exc(),
                }
            )
            print(f"[phase118h] ERROR {sample.get('qid')}: {exc}")

    rollout_records: List[Dict[str, Any]] = []
    for group in groups:
        rollout_records.extend(group["records"])
    return groups, rollout_records, failures


def run_post_analysis(
    rollout_path: Path,
    output_dir: Path,
    *,
    k_list: List[int],
    candidate_name: str,
) -> Dict[str, Any]:
    analyzer = QrelsMetricBlindness(k_list=k_list)
    inputs = analyzer.load_inputs(rollout_path)
    search_tool = BM25SearchTool(rec_r1_root=REC_R1)
    analysis = analyzer.analyze_all(inputs, search_tool=search_tool)

    qrels_dir = output_dir
    _write_jsonl(qrels_dir / "query_relevance_coverage.jsonl", analysis["query_coverage_rows"])
    _write_jsonl(qrels_dir / "metric_by_k_diagnostics.jsonl", analysis["metric_by_k_rows"])
    _write_jsonl(qrels_dir / "group_metric_spread_by_k.jsonl", analysis["group_spread_rows"])
    (qrels_dir / "v2_qrels_metric_summary.json").write_text(
        json.dumps(analysis["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dryrun = LargeKRewardDryRun()
    lk_result = dryrun.run(
        rollout_path=rollout_path,
        metric_by_k_path=qrels_dir / "metric_by_k_diagnostics.jsonl",
        group_metric_spread_path=qrels_dir / "group_metric_spread_by_k.jsonl",
        decomposition_path=None,
    )

    with (qrels_dir / "large_k_shaped_record_rewards.jsonl").open("w", encoding="utf-8") as fout:
        for row in lk_result["shaped_records"]:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (qrels_dir / "large_k_candidate_group_reports.jsonl").open("w", encoding="utf-8") as fout:
        for row in lk_result["candidate_group_reports"]:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    candidate_summary = next(
        s
        for s in lk_result["comparison"]["candidates"]
        if s["candidate_name"] == candidate_name
    )
    gate = dryrun.evaluate_gate(lk_result["comparison"])
    scale_gate = ScaleGateCheck().evaluate_scale_gate(
        {"gate_passed": gate["gate_passed"], "recommended_candidate": gate["recommended_candidate"]},
        lk_result["comparison"],
        candidate_name=candidate_name,
    )

    lk_summary = {
        "num_groups": lk_result["num_groups"],
        "num_rollout_records": lk_result["num_rollout_records"],
        "candidate_name": candidate_name,
        "candidate_summary": candidate_summary,
        "gate_passed": gate["gate_passed"],
        "scale_gate_passed": scale_gate.get("gate_passed", False),
        "recommended_candidate": gate.get("recommended_candidate"),
        "gate_reason": gate.get("reason"),
        "zero_std_group_rate": candidate_summary["zero_std_group_rate"],
        "retrieval_quality_spread_group_rate": candidate_summary[
            "retrieval_quality_spread_group_rate"
        ],
        "penalty_only_spread_group_rate": candidate_summary["penalty_only_spread_group_rate"],
    }
    (qrels_dir / "v2_large_k_reward_summary.json").write_text(
        json.dumps(lk_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    shaped_by_sample = {
        row["sample_id"]: float(row[candidate_name]) for row in lk_result["shaped_records"]
    }
    return {
        "qrels_summary": analysis["summary"],
        "large_k_summary": lk_summary,
        "gate": gate,
        "scale_gate": scale_gate,
        "shaped_by_sample": shaped_by_sample,
        "candidate_group_reports": lk_result["candidate_group_reports"],
    }


def build_phase2_candidate_v2(
    candidate_rows: List[Dict[str, Any]],
    comparison_rows: List[Dict[str, Any]],
    *,
    target_collapse_only: bool,
    collapse_fixed: bool,
) -> List[Dict[str, Any]]:
    comp_by_group = {r["group_id"]: r for r in comparison_rows}
    out: List[Dict[str, Any]] = []
    for row in candidate_rows:
        gid = row["group_id"]
        comp = comp_by_group.get(gid)
        entry = dict(row)
        entry["strategy_prompt_version"] = "v2"
        if comp:
            entry["v2_unique_final_query_count"] = comp["v2_unique_final_query_count"]
            entry["v2_strategy_collapse"] = comp["v2_strategy_collapse"]
            entry["recommended_for_phase2"] = comp["recommended_for_phase2"]
        out.append(entry)

    if target_collapse_only and collapse_fixed:
        for comp in comparison_rows:
            if comp.get("collapse_fixed"):
                out.append(
                    {
                        "group_id": comp["group_id"],
                        "original_query": comp["original_query"],
                        "source": "collapse_fixed_v2",
                        "strategy_prompt_version": "v2",
                        "v2_unique_final_query_count": comp["v2_unique_final_query_count"],
                        "v2_strategy_collapse": comp["v2_strategy_collapse"],
                        "recommended_for_phase2": comp["recommended_for_phase2"],
                        "collapse_fixed": True,
                    }
                )
    return out


def build_report(summary: Dict[str, Any], comparisons: List[Dict[str, Any]]) -> str:
    lines = [
        "# Phase 1.18h Strategy Prompt V2 Report",
        "",
        f"- mode: `{summary['mode']}`",
        f"- target_collapse_only: **{summary['target_collapse_only']}**",
        f"- num_groups: **{summary['num_groups']}**",
        f"- v1_strategy_collapse_count: **{summary['v1_strategy_collapse_count']}**",
        f"- v2_strategy_collapse_count: **{summary['v2_strategy_collapse_count']}**",
        f"- collapse_fixed_count: **{summary['collapse_fixed_count']}**",
        f"- collapse_fix_rate: **{summary['collapse_fix_rate']:.2f}**",
        f"- v2_gate_passed: **{summary['v2_gate_passed']}**",
        f"- phase2_candidate_ready: **{summary['phase2_candidate_ready']}**",
        "",
        "## Per-Group Comparison",
        "",
    ]
    for row in comparisons:
        lines.append(
            f"- `{row['group_id']}`: v1_unique={row['v1_unique_final_query_count']} "
            f"v2_unique={row['v2_unique_final_query_count']} "
            f"v1_jaccard={row['v1_avg_pairwise_final_query_jaccard']:.3f} "
            f"v2_jaccard={row['v2_avg_pairwise_final_query_jaccard']:.3f} "
            f"collapse_fixed={row['collapse_fixed']}"
        )
        if row.get("v2_final_queries"):
            lines.append(f"  - v2 queries: {row['v2_final_queries']}")
    lines.extend(["", summary.get("next_recommendation", ""), "", DRYRUN_WARNING])
    if summary.get("blocking_reason"):
        lines.extend(["", f"Blocking reason: {summary['blocking_reason']}"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1.18h strategy prompt V2 smoke")
    parser.add_argument(
        "--cleanup-report-path",
        type=Path,
        default=ROOT / "experiments/phase118g_bm25_failure_cleanup_20_g4/group_cleanup_report.jsonl",
    )
    parser.add_argument(
        "--phase2-candidate-path",
        type=Path,
        default=ROOT / "experiments/phase118g_bm25_failure_cleanup_20_g4/phase2_candidate_smoke_set.jsonl",
    )
    parser.add_argument(
        "--old-rollout-path",
        type=Path,
        default=ROOT / "experiments/phase119b_scale_gate_check/strategy_rollout_20_g4/rollout_records.jsonl",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "Rec-R1/data/esci/inst/sparse/subset/val.parquet",
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
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/phase118h_strategy_prompt_v2_20_g4",
    )
    parser.add_argument("--target-collapse-only", type=str, default="false")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--metric-k-list", type=int, nargs="+", default=[10, 50, 100, 1000])
    parser.add_argument("--candidate-name", type=str, default="reward_largek_mix_1000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_STRATEGY_ORDER_V2,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_strategies_v2(args.strategies)
    target_collapse_only = _bool_arg(args.target_collapse_only)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cleanup_rows = _load_jsonl(args.cleanup_report_path)
    phase2_rows = _load_jsonl(args.phase2_candidate_path)
    old_records = _load_jsonl(args.old_rollout_path)
    old_by_group = _index_old_rollout(old_records)

    candidate_rows = select_candidate_rows(
        cleanup_rows,
        phase2_rows,
        target_collapse_only=target_collapse_only,
    )
    samples = build_samples(candidate_rows, old_by_group, args.data_path)
    group_ids = [s["qid"] for s in samples]

    old_subset = [r for r in old_records if r.get("group_id") in group_ids]
    diagnostics = StrategyCollapseDiagnostics()
    v1_effect = diagnostics.summarize_strategy_effect(old_subset, candidate_name=args.candidate_name)

    groups, rollout_records, failures = run_v2_rollout(
        samples,
        model_path=args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_steps=args.max_steps,
        topk=args.topk,
        seed=args.seed,
        strategies=args.strategies,
    )

    rollout_path = args.output_dir / "v2_rollout_records.jsonl"
    _write_jsonl(rollout_path, rollout_records)

    group_summaries = []
    for group in groups:
        group_summaries.append(
            {
                "group_id": group["group_id"],
                "original_query": group["original_query"],
                "group_size": group["group_size"],
                "strategies": group["strategies"],
                "strategy_version": "v2",
                "group_metrics": group["group_metrics"],
            }
        )
    _write_jsonl(args.output_dir / "v2_group_summaries.jsonl", group_summaries)

    post = run_post_analysis(
        rollout_path,
        args.output_dir,
        k_list=args.metric_k_list,
        candidate_name=args.candidate_name,
    )

    v1_shaped = None
    v2_shaped = post["shaped_by_sample"]
    compare = diagnostics.compare_v1_v2(
        old_subset,
        rollout_records,
        candidate_name=args.candidate_name,
        v2_shaped=v2_shaped,
    )
    v2_effect = diagnostics.summarize_strategy_effect(
        rollout_records,
        candidate_name=args.candidate_name,
        shaped_by_sample=v2_shaped,
    )

    _write_jsonl(args.output_dir / "strategy_collapse_comparison.jsonl", compare["comparisons"])

    collapse_fixed = compare["collapse_fixed_count"] > 0
    phase2_v2 = build_phase2_candidate_v2(
        candidate_rows,
        compare["comparisons"],
        target_collapse_only=target_collapse_only,
        collapse_fixed=collapse_fixed,
    )
    _write_jsonl(args.output_dir / "phase2_candidate_smoke_set_v2.jsonl", phase2_v2)

    lk = post["large_k_summary"]
    v2_gate_passed = bool(lk.get("gate_passed") or lk.get("scale_gate_passed"))
    strategy_collapse_remaining = v2_effect["strategy_collapse_count"] > 0

    if target_collapse_only:
        phase2_ready = collapse_fixed and not strategy_collapse_remaining
        blocking_reason = None if phase2_ready else (
            "strategy collapse remains; replace collapse group or revise prompt again."
            if strategy_collapse_remaining
            else "collapse not fixed in targeted run."
        )
    else:
        phase2_ready = v2_gate_passed and not strategy_collapse_remaining
        blocking_reason = None
        if strategy_collapse_remaining:
            blocking_reason = "strategy_collapse remains; run Phase 1.18h targeted or replace group."
        elif not v2_gate_passed:
            blocking_reason = "V2 full gate did not pass; keep V1 default and use V2 targeted only."

    summary = {
        "phase": "1.18h",
        "mode": "strategy_prompt_v2",
        "target_collapse_only": target_collapse_only,
        "num_groups": len(groups),
        "group_size": args.group_size,
        "num_rollout_records": len(rollout_records),
        "v1_strategy_collapse_count": v1_effect["strategy_collapse_count"],
        "v2_strategy_collapse_count": v2_effect["strategy_collapse_count"],
        "collapse_fixed_count": compare["collapse_fixed_count"],
        "collapse_fix_rate": compare["collapse_fix_rate"],
        "targeted_collapse_group_count": compare["targeted_collapse_group_count"],
        "targeted_fix_rate": compare["targeted_fix_rate"],
        "v1_zero_std_group_rate": v1_effect["zero_std_group_rate"],
        "v2_zero_std_group_rate": lk.get("zero_std_group_rate", v2_effect["zero_std_group_rate"]),
        "v1_retrieval_quality_spread_group_rate": None,
        "v2_retrieval_quality_spread_group_rate": lk.get("retrieval_quality_spread_group_rate"),
        "v2_penalty_only_spread_group_rate": lk.get("penalty_only_spread_group_rate"),
        "v2_gate_passed": v2_gate_passed,
        "phase2_candidate_ready": phase2_ready,
        "safe_for_phase2_tiny_training": phase2_ready,
        "blocking_reason": blocking_reason,
        "is_training": False,
        "failures": failures,
        "next_recommendation": (
            "Proceed to Phase 2.1 Tiny GRPO Smoke Training with strict step limits and no checkpoint promotion."
            if phase2_ready
            else "Fix remaining collapse or rerun full candidate set after targeted fix."
        ),
        "dryrun_warning": DRYRUN_WARNING,
        "gate_thresholds": {**GATE_THRESHOLDS, **DEFAULT_GATE_THRESHOLDS},
    }

    strategy_v2_summary = {
        "strategy_version": "v2",
        "strategies": args.strategies,
        "v1_effect": v1_effect,
        "v2_effect": v2_effect,
        "compare": {
            "targeted_collapse_group_count": compare["targeted_collapse_group_count"],
            "collapse_fixed_count": compare["collapse_fixed_count"],
            "targeted_fix_rate": compare["targeted_fix_rate"],
        },
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "strategy_v2_summary.json").write_text(
        json.dumps(strategy_v2_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "strategy_prompt_v2_report.md").write_text(
        build_report(summary, compare["comparisons"]),
        encoding="utf-8",
    )
    readme = args.output_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# Phase 1.18h Strategy Prompt V2\n\n"
            "Anti-collapse strategy prompts with targeted/full rerollout. No training.\n",
            encoding="utf-8",
        )

    print("\n=== Phase 1.18h Strategy Prompt V2 Summary ===")
    print(f"target_collapse_only: {summary['target_collapse_only']}")
    print(f"num_groups: {summary['num_groups']}")
    print(f"num_rollout_records: {summary['num_rollout_records']}")
    print(f"v1_strategy_collapse_count: {summary['v1_strategy_collapse_count']}")
    print(f"v2_strategy_collapse_count: {summary['v2_strategy_collapse_count']}")
    print(f"collapse_fixed_count: {summary['collapse_fixed_count']}")
    print(f"collapse_fix_rate: {summary['collapse_fix_rate']:.4f}")
    print(f"v2_zero_std_group_rate: {summary['v2_zero_std_group_rate']}")
    print(f"v2_retrieval_quality_spread_group_rate: {summary['v2_retrieval_quality_spread_group_rate']}")
    print(f"v2_penalty_only_spread_group_rate: {summary['v2_penalty_only_spread_group_rate']}")
    print(f"v2_gate_passed: {summary['v2_gate_passed']}")
    print(f"phase2_candidate_ready: {summary['phase2_candidate_ready']}")
    print(f"safe_for_phase2_tiny_training: {summary['safe_for_phase2_tiny_training']}")
    print(f"blocking_reason: {summary.get('blocking_reason')}")
    print(f"output_dir: {args.output_dir}")
    print(f"\n[phase118h] {DRYRUN_WARNING}")


if __name__ == "__main__":
    main()
