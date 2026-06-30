#!/usr/bin/env python3
"""
Phase 2.3: 10-step controlled GRPO smoke training.

Usage:
  source /data1/hcc/agentic-rec/env.sh
  cd /data1/hcc/agentic-rec
  CUDA_VISIBLE_DEVICES=2 python scripts/run_10step_grpo_controlled_smoke.py \
    --output-dir experiments/phase23_10step_grpo_controlled_smoke/lr_1e-6 \
    --learning-rate 1e-6
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

from src.agents.controlled_grpo_smoke_trainer import ControlledGrpoSmokeTrainer
from src.agents.grpo_curve_analyzer import GRPOCurveAnalyzer
from src.agents.grpo_stability_monitor import GRPOStabilityMonitor
from src.agents.phase2_smoke_dataset import load_clean_set_rows
from src.agents.tiny_grpo_smoke_trainer import CHECKPOINT_LABEL


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
    if "esci_val_3" in [r["group_id"] for r in rows]:
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
    samples = v2_module.build_samples(
        candidate_rows,
        v2_module._index_old_rollout(preflight_records),
        data_path,
    )

    eval_dir = output_dir / "post_eval_10step"
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
        rollout_path, eval_dir, k_list=[10, 50, 100, 1000], candidate_name=candidate_name
    )

    parse_rates = [float(g["group_metrics"].get("json_parse_success_rate", 0.0)) for g in groups]
    finish_rates = [float(g["group_metrics"].get("finish_rate", 0.0)) for g in groups]
    invalid_rates = [float(g["group_metrics"].get("invalid_action_rate", 0.0)) for g in groups]

    shaped_rows = _load_jsonl(eval_dir / "large_k_shaped_record_rewards.jsonl")
    rewards = [float(r[candidate_name]) for r in shaped_rows if candidate_name in r]
    ndcg_vals = [float(r.get("ndcg@1000", 0.0)) for r in shaped_rows]
    recall_vals = [float(r.get("recall@1000", 0.0)) for r in shaped_rows]
    mrr_vals = [float(r.get("mrr@1000", 0.0)) for r in shaped_rows]

    grouped: Dict[str, List[float]] = {}
    shaped_by_id = {r["sample_id"]: float(r[candidate_name]) for r in shaped_rows}
    for record in rollout_records:
        sid = record.get("sample_id")
        gid = record.get("group_id")
        if sid in shaped_by_id:
            grouped.setdefault(gid, []).append(shaped_by_id[sid])

    zero_std = sum(1 for v in grouped.values() if len(v) > 1 and pstdev(v) <= 1e-6)
    spread = sum(1 for v in grouped.values() if len(v) > 1 and (max(v) - min(v)) > 1e-6)
    ng = len(grouped) or 1
    parse_success_rate = float(mean(parse_rates)) if parse_rates else 0.0

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_label": CHECKPOINT_LABEL,
        "num_groups": len(groups),
        "num_rollout_records": len(rollout_records),
        "num_failures": len(failures),
        "parse_success_rate": parse_success_rate,
        "finish_rate": float(mean(finish_rates)) if finish_rates else 0.0,
        "invalid_action_rate": float(mean(invalid_rates)) if invalid_rates else 0.0,
        "zero_std_group_rate": zero_std / ng,
        "retrieval_quality_spread_group_rate": spread / ng,
        "mean_reward_largek_mix_1000": float(mean(rewards)) if rewards else 0.0,
        "mean_ndcg1000": float(mean(ndcg_vals)) if ndcg_vals else 0.0,
        "mean_recall1000": float(mean(recall_vals)) if recall_vals else 0.0,
        "mean_mrr1000": float(mean(mrr_vals)) if mrr_vals else 0.0,
        "json_format_ok": parse_success_rate >= 0.95,
        "eval_passed": len(failures) == 0 and parse_success_rate >= 0.95,
    }


def build_report(
    summary: Dict[str, Any],
    curve: Dict[str, Any],
    post_eval: Optional[Dict[str, Any]],
    prior_evals: Dict[str, Optional[Dict[str, Any]]],
) -> str:
    lines = [
        "# Phase 2.3 Ten-Step Controlled GRPO Smoke Report",
        "",
        f"- learning_rate: **{summary.get('learning_rate')}**",
        f"- actual_update_steps: **{summary.get('actual_update_steps')}**",
        f"- ten_step_smoke_passed: **{summary.get('ten_step_smoke_passed')}**",
        f"- stability_class: **{curve.get('stability_class')}**",
        f"- max_approx_kl_nonnegative: **{curve.get('max_approx_kl_nonnegative')}**",
        f"- max_grad_norm: **{curve.get('max_grad_norm')}**",
        "",
        "## Trends",
        "",
        f"- policy_loss_trend: `{curve.get('policy_loss_trend')}`",
        f"- approx_kl_trend: `{curve.get('approx_kl_trend')}`",
        f"- grad_norm_trend: `{curve.get('grad_norm_trend')}`",
        f"- clipfrac_trend: `{curve.get('clipfrac_trend')}`",
        f"- reward_trend: `{curve.get('reward_trend')}`",
        "",
    ]
    if post_eval:
        lines.extend(
            [
                "## Post-Train Eval (10-step)",
                "",
                f"- parse_success_rate: **{post_eval.get('parse_success_rate')}**",
                f"- invalid_action_rate: **{post_eval.get('invalid_action_rate')}**",
                f"- json_format_ok: **{post_eval.get('json_format_ok')}**",
                f"- mean_reward_largek_mix_1000: **{post_eval.get('mean_reward_largek_mix_1000')}**",
                "",
            ]
        )
    for label, ev in prior_evals.items():
        if ev:
            lines.append(
                f"- {label} mean_reward: **{ev.get('mean_reward_largek_mix_1000')}** "
                f"parse: **{ev.get('parse_success_rate')}**"
            )
    lines.extend(["", summary.get("next_recommendation", ""), ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2.3 10-step controlled GRPO smoke")
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
        "--model-path",
        type=str,
        default="/data1/hcc/.hf_home/Qwen2.5-3B-Instruct",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/data1/hcc/.hf_home/Qwen2.5-3B-Instruct",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reward-candidate", type=str, default="reward_largek_mix_1000")
    parser.add_argument("--max-update-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--cliprange", type=float, default=0.2)
    parser.add_argument("--train-batch-size", type=int, default=20)
    parser.add_argument("--rollout-n", type=int, default=4)
    parser.add_argument("--ppo-mini-batch-size", type=int, default=20)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--save-steps", type=int, nargs="+", default=[1, 5, 10])
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

    validate_clean_set(args.clean_set_path, args.clean_set_summary_path)

    rollout_path = args.preflight_rollout_dir / "v2_rollout_records.jsonl"
    shaped_path = args.preflight_rollout_dir / "large_k_shaped_record_rewards.jsonl"
    if not rollout_path.exists() or not shaped_path.exists():
        raise SystemExit(f"missing preflight artifacts under {args.preflight_rollout_dir}")

    monitor = GRPOStabilityMonitor(max_approx_kl=0.2, max_grad_norm=10.0)
    trainer = ControlledGrpoSmokeTrainer(
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
        result = trainer.run_controlled(
            rollout_path=rollout_path,
            shaped_reward_path=shaped_path,
            output_dir=args.output_dir,
            save_steps=args.save_steps,
            max_prompt_length=args.max_prompt_length,
            max_response_length=args.max_response_length,
            max_total_length=args.max_total_length,
            stability_monitor=monitor,
        )
    except Exception as exc:
        (args.output_dir / "failure_report.md").write_text(
            f"# Phase 2.3 Failure\n\n{exc}\n\n{traceback.format_exc()}\n",
            encoding="utf-8",
        )
        raise SystemExit(1) from exc

    step_metrics = result["step_metrics"]
    monitor_report = monitor.summarize(step_metrics)

    analyzer = GRPOCurveAnalyzer()
    curve = analyzer.analyze_trends(step_metrics)
    rec = analyzer.recommend_next_step(curve)
    (args.output_dir / "curve_analysis.json").write_text(
        json.dumps({**curve, "recommendation": rec}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    post_eval: Optional[Dict[str, Any]] = None
    ckpt_path = result["summary"].get("checkpoint_path")
    if not args.skip_post_eval and ckpt_path:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            post_eval = run_post_train_eval(
                clean_set_path=args.clean_set_path,
                checkpoint_path=Path(ckpt_path),
                output_dir=args.output_dir,
                data_path=args.data_path,
                preflight_rollout_dir=args.preflight_rollout_dir,
                candidate_name=args.reward_candidate,
                temperature=args.temperature,
                top_p=args.top_p,
                topk=args.topk,
                seed=args.seed,
            )
            (args.output_dir / "post_10step_eval_summary.json").write_text(
                json.dumps(post_eval, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            post_eval = {"eval_passed": False, "json_format_ok": False, "error": str(exc)}

    eval_ok = post_eval is None or (
        post_eval.get("json_format_ok") is True
        and float(post_eval.get("parse_success_rate", 0.0)) >= 0.95
        and float(post_eval.get("invalid_action_rate", 1.0)) <= 0.05
    )

    passed = (
        result["summary"].get("actual_update_steps") == args.max_update_steps
        and monitor_report["optimizer_steps_called"] == args.max_update_steps
        and monitor_report["all_steps_passed"]
        and not monitor_report["nan_detected"]
        and not monitor_report["oom_detected"]
        and not monitor_report["kl_exploded"]
        and curve["max_approx_kl_nonnegative"] <= 0.2
        and curve["max_grad_norm"] <= 10.0
        and eval_ok
    )

    summary = {
        **result["summary"],
        "ten_step_smoke_passed": passed,
        "optimizer_steps_called": monitor_report["optimizer_steps_called"],
        "loss_finite_all_steps": monitor_report["loss_finite_all_steps"],
        "grad_norm_finite_all_steps": monitor_report["grad_norm_finite_all_steps"],
        "max_grad_norm": curve["max_grad_norm"],
        "max_approx_kl_nonnegative": curve["max_approx_kl_nonnegative"],
        "max_abs_signed_logprob_gap": curve["max_abs_signed_logprob_gap"],
        "max_clipfrac": curve["max_clipfrac"],
        "stability_class": curve["stability_class"],
        "next_recommendation": rec["action"] if not passed else rec["action"],
    }
    (args.output_dir / "ten_step_train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    prior: Dict[str, Optional[Dict[str, Any]]] = {}
    for label, path in [
        ("1-step", PHASE21_DIR / "post_train_eval_summary.json"),
        ("3-step", ROOT / "experiments/phase22_3step_grpo_stability_smoke/post_3step_eval_summary.json"),
    ]:
        prior[label] = _load_json(path) if path.exists() else None

    (args.output_dir / "ten_step_grpo_controlled_report.md").write_text(
        build_report(summary, curve, post_eval, prior),
        encoding="utf-8",
    )

    readme = args.output_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# Phase 2.3 LR={args.learning_rate} Ten-Step Controlled Smoke\n\n"
            "SMOKE_ONLY_DO_NOT_PROMOTE checkpoints.\n",
            encoding="utf-8",
        )

    print("\n=== Phase 2.3 Ten-Step Controlled Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if not passed:
        (args.output_dir / "failure_report.md").write_text(
            f"ten_step_smoke_passed=false\n\n{json.dumps(monitor_report, indent=2)}\n",
            encoding="utf-8",
        )
        raise SystemExit("10-step controlled smoke did not pass acceptance criteria")


if __name__ == "__main__":
    main()
