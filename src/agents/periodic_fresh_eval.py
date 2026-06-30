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


def build_samples_from_clean_rows(
    candidate_rows: List[Dict[str, Any]],
    data_path: Path,
    v2_module: Any,
    *,
    preflight_rollout_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Build rollout samples from clean-set rows, with optional preflight fallback."""
    if preflight_rollout_dir is not None:
        preflight_path = preflight_rollout_dir / "v2_rollout_records.jsonl"
        if preflight_path.exists():
            preflight_records = _load_jsonl_local(preflight_path)
            return v2_module.build_samples(
                candidate_rows,
                v2_module._index_old_rollout(preflight_records),
                data_path,
            )

    samples: List[Dict[str, Any]] = []
    for row in candidate_rows:
        target_items = list(row.get("target_items", []))
        sample = {
            "qid": row["group_id"],
            "user_query": row.get("original_query", ""),
            "target_items": target_items,
        }
        if not target_items:
            loaded = v2_module._load_esci_sample_by_qid(data_path, row["group_id"])
            if loaded is None:
                raise KeyError(f"cannot resolve sample for group_id={row['group_id']}")
            sample = loaded
            if row.get("original_query"):
                sample["user_query"] = row["original_query"]
        samples.append(sample)
    return samples


def _load_jsonl_local(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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
    eval_split: str = "train",
    eval_root_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Fresh rollout eval on clean set (step 0 uses base model_path)."""
    script_path = root / "scripts/smoke_strategy_prompt_v2.py"
    spec = importlib.util.spec_from_file_location("smoke_strategy_prompt_v2", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script_path}")
    v2_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2_module)

    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        return _load_jsonl_local(path)

    candidate_rows = _load_jsonl(clean_set_path)
    samples = build_samples_from_clean_rows(
        candidate_rows,
        data_path,
        v2_module,
        preflight_rollout_dir=preflight_rollout_dir,
    )

    eval_root = eval_root_name if eval_root_name is not None else f"eval_step_{eval_step}"
    eval_dir = output_dir / eval_root / eval_split
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
        "eval_split": eval_split,
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


def resolve_eval_checkpoint_path(step_result: Dict[str, Any]) -> Optional[Path]:
    """Resolve checkpoint path for fresh eval from step result fields."""
    for key in ("checkpoint_path", "eval_snapshot_path", "stop_snapshot_path"):
        value = step_result.get(key)
        if value:
            return Path(value)
    return None


def run_dual_fresh_eval(
    *,
    train_clean_path: Path,
    heldout_clean_path: Path,
    checkpoint_path: Optional[Path],
    output_dir: Path,
    eval_step: int,
    data_path: Path,
    train_preflight_dir: Path,
    heldout_preflight_dir: Optional[Path],
    candidate_name: str,
    model_path: str,
    temperature: float,
    top_p: float,
    topk: int,
    seed: int,
    root: Path,
    checkpoint_prefix: str = "pilot_step",
    checkpoint_label: str = PILOT_CHECKPOINT_LABEL,
    eval_root_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run fresh eval on both train and heldout clean sets."""
    train_eval = run_fresh_eval(
        clean_set_path=train_clean_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        eval_step=eval_step,
        data_path=data_path,
        preflight_rollout_dir=train_preflight_dir,
        candidate_name=candidate_name,
        model_path=model_path,
        temperature=temperature,
        top_p=top_p,
        topk=topk,
        seed=seed,
        root=root,
        checkpoint_prefix=checkpoint_prefix,
        checkpoint_label=checkpoint_label,
        eval_split="train",
        eval_root_name=eval_root_name,
    )
    heldout_eval = run_fresh_eval(
        clean_set_path=heldout_clean_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        eval_step=eval_step,
        data_path=data_path,
        preflight_rollout_dir=heldout_preflight_dir or train_preflight_dir,
        candidate_name=candidate_name,
        model_path=model_path,
        temperature=temperature,
        top_p=top_p,
        topk=topk,
        seed=seed + 1000,
        root=root,
        checkpoint_prefix=checkpoint_prefix,
        checkpoint_label=checkpoint_label,
        eval_split="heldout",
        eval_root_name=eval_root_name,
    )
    return {"train": train_eval, "heldout": heldout_eval}


def run_final_stop_dual_eval(
    *,
    train_clean_path: Path,
    heldout_clean_path: Path,
    checkpoint_path: Optional[Path],
    output_dir: Path,
    stop_step: int,
    data_path: Path,
    train_preflight_dir: Path,
    heldout_preflight_dir: Optional[Path],
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
    """Run fresh eval at early stop step into final_stop_eval_step_<n>/."""
    return run_dual_fresh_eval(
        train_clean_path=train_clean_path,
        heldout_clean_path=heldout_clean_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        eval_step=stop_step,
        data_path=data_path,
        train_preflight_dir=train_preflight_dir,
        heldout_preflight_dir=heldout_preflight_dir,
        candidate_name=candidate_name,
        model_path=model_path,
        temperature=temperature,
        top_p=top_p,
        topk=topk,
        seed=seed,
        root=root,
        checkpoint_prefix=checkpoint_prefix,
        checkpoint_label=checkpoint_label,
        eval_root_name=f"final_stop_eval_step_{stop_step}",
    )


def _run_dual_eval_at_step(
    *,
    step_idx: int,
    step_result: Dict[str, Any],
    run_dual_eval_fn: Callable[[int, Optional[Path]], Dict[str, Any]],
    periodic_records: List[Dict[str, Any]],
    monitor: Any,
    early_stop_state: Dict[str, Any],
    eval_label: str,
) -> None:
    ckpt_path = resolve_eval_checkpoint_path(step_result)
    print(f"\n=== Step-{step_idx} {eval_label} train+heldout fresh eval ===")
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dual = run_dual_eval_fn(step_idx, ckpt_path)
        check = monitor.check_eval_pair(dual["train"], dual["heldout"], step=step_idx)
        periodic_records.append(check)
        if check["should_stop"]:
            early_stop_state["stopped"] = True
            early_stop_state["reason"] = check["stop_reason"]
            step_result["hard_stop"] = True
            step_result["hard_stop_reason"] = check["stop_reason"]
    except Exception as exc:
        early_stop_state["stopped"] = True
        early_stop_state["reason"] = f"step-{step_idx} dual eval failed: {exc}"
        step_result["hard_stop"] = True
        step_result["hard_stop_reason"] = early_stop_state["reason"]


def make_periodic_dual_eval_hook(
    *,
    eval_steps: List[int],
    periodic_records: List[Dict[str, Any]],
    run_dual_eval_fn: Callable[[int, Optional[Path]], Dict[str, Any]],
    monitor: Any,
    early_stop_state: Dict[str, Any],
) -> Callable[[int, Dict[str, Any]], None]:
    """Periodic hook that runs train + heldout eval independent of checkpoint saves."""

    def hook(step_idx: int, step_result: Dict[str, Any]) -> None:
        if early_stop_state.get("stopped"):
            return
        if step_idx not in eval_steps or step_idx <= 0:
            return
        if resolve_eval_checkpoint_path(step_result) is None:
            print(
                f"[eval hook] step {step_idx} in eval_steps but no checkpoint snapshot; skipping"
            )
            return
        _run_dual_eval_at_step(
            step_idx=step_idx,
            step_result=step_result,
            run_dual_eval_fn=run_dual_eval_fn,
            periodic_records=periodic_records,
            monitor=monitor,
            early_stop_state=early_stop_state,
            eval_label="periodic",
        )

    return hook


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

    Runs fresh eval at eval_steps using checkpoint or eval snapshot paths.
    Independent of save_steps / checkpoint_saved.
    """

    def hook(step_idx: int, step_result: Dict[str, Any]) -> None:
        if early_stop_state.get("stopped"):
            return
        if step_idx not in eval_steps or step_idx <= 0:
            return
        ckpt_path = resolve_eval_checkpoint_path(step_result)
        if ckpt_path is None:
            print(
                f"[eval hook] step {step_idx} in eval_steps but no checkpoint snapshot; skipping"
            )
            return

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
