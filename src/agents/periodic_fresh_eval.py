"""
Phase 2.5a: Periodic fresh rollout eval for GRPO pilot training.

Runs fresh v2 strategy rollout eval at checkpoint steps during training.
Fixes strategy_distribution field mapping (strategy_name vs strategy).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Dict, List, Optional

from src.agents.grpo_pilot_monitor import PILOT_CHECKPOINT_LABEL


def extract_strategy_from_record(record: Dict[str, Any]) -> str:
    """Unified strategy field resolution for rollout records."""
    return (
        record.get("strategy_name")
        or record.get("search_strategy")
        or record.get("strategy")
        or "unknown"
    )


def compute_strategy_distribution(rollout_records: List[Dict[str, Any]]) -> Dict[str, float]:
    strategy_counts: Dict[str, int] = {}
    for record in rollout_records:
        strat = extract_strategy_from_record(record)
        strategy_counts[strat] = strategy_counts.get(strat, 0) + 1
    total_records = len(rollout_records) or 1
    return {k: v / total_records for k, v in strategy_counts.items()}


def resolve_checkpoint_path(
    output_dir: Path,
    eval_step: int,
    *,
    checkpoint_prefix: str = "pilot_step",
) -> Path:
    return output_dir / "checkpoints" / f"{checkpoint_prefix}_{eval_step}"


def run_fresh_eval(
    *,
    clean_set_path: Path,
    checkpoint_path: Optional[Path],
    output_dir: Path,
    eval_step: int,
    data_path: Path,
    preflight_rollout_dir: Path,
    candidate_name: str,
    model_path: str,
    temperature: float,
    top_p: float,
    topk: int,
    seed: int,
    root: Path,
    checkpoint_prefix: str = "pilot_step",
    checkpoint_label: str = PILOT_CHECKPOINT_LABEL,
) -> Dict[str, Any]:
    """Fresh rollout eval on clean set (step 0 uses base model_path)."""
    script_path = root / "scripts/smoke_strategy_prompt_v2.py"
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

    eval_dir = output_dir / f"eval_step_{eval_step}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    ckpt = str(checkpoint_path) if checkpoint_path else model_path

    groups, rollout_records, failures = v2_module.run_v2_rollout(
        samples,
        model_path=Path(ckpt),
        temperature=temperature,
        top_p=top_p,
        max_tokens=256,
        max_steps=3,
        topk=topk,
        seed=seed + eval_step,
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

    strategy_distribution = compute_strategy_distribution(rollout_records)

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

    resolved_ckpt_path = str(checkpoint_path) if checkpoint_path else model_path

    summary = {
        "eval_step": eval_step,
        "checkpoint_path": resolved_ckpt_path,
        "checkpoint_label": checkpoint_label,
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
        "strategy_distribution": strategy_distribution,
        "json_format_ok": parse_success_rate >= 0.95,
        "eval_passed": len(failures) == 0 and parse_success_rate >= 0.95,
    }
    (eval_dir / "post_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def make_periodic_eval_hook(
    *,
    eval_steps: List[int],
    eval_summaries: List[Dict[str, Any]],
    run_eval_fn: Callable[[int, Path], Dict[str, Any]],
    monitor: Any,
    early_stop_state: Dict[str, Any],
) -> Callable[[int, Dict[str, Any]], None]:
    """
    Build a step_hook for TinyGrpoSmokeTrainer.

    Calls fresh eval immediately after each checkpoint save at eval_steps.
    """

    def hook(step_idx: int, step_result: Dict[str, Any]) -> None:
        if early_stop_state.get("stopped"):
            return
        if step_idx not in eval_steps or step_idx <= 0:
            return
        if not step_result.get("checkpoint_saved"):
            return

        ckpt_path = Path(step_result["checkpoint_path"])
        print(f"\n=== Step-{step_idx} periodic fresh eval ===")
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ev = run_eval_fn(step_idx, ckpt_path)
            eval_summaries.append(ev)
            check = monitor.check_eval_summary(ev, step=step_idx)
            if check["should_stop"]:
                early_stop_state["stopped"] = True
                early_stop_state["reason"] = check["stop_reason"]
                step_result["hard_stop"] = True
                step_result["hard_stop_reason"] = check["stop_reason"]
        except Exception as exc:
            early_stop_state["stopped"] = True
            early_stop_state["reason"] = f"step-{step_idx} eval failed: {exc}"
            step_result["hard_stop"] = True
            step_result["hard_stop_reason"] = early_stop_state["reason"]

    return hook
