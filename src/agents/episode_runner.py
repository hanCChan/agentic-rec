"""Finish-aware multi-step episode runner (Phase 1.6+)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.agents.commerce_agent_env import CommerceAgentEnv
from src.agents.qwen_rollout_policy import QwenRolloutPolicy


def load_esci_samples(parquet_path: Path, num_samples: int) -> List[Dict[str, Any]]:
    df = pd.read_parquet(parquet_path).head(num_samples)
    samples = []
    for idx, row in df.iterrows():
        targets = [str(x) for x in row["item_id"]]
        samples.append(
            {
                "qid": f"{row.get('data_source', 'esci')}_{idx}",
                "user_query": str(row["query"]),
                "target_items": targets,
            }
        )
    return samples


def _step_penalties(env_step, parse_ok: bool) -> Dict[str, float]:
    penalties = {"search_cost": 0.0, "repeat": 0.0, "invalid": 0.0}
    step = env_step
    if step.invalid:
        if step.invalid_reason == "repeated_query":
            penalties["repeat"] = step.penalty
        else:
            penalties["invalid"] = step.penalty
    elif step.penalty and step.action.get("tool") == "bm25_search":
        penalties["search_cost"] = step.penalty
    if not parse_ok:
        penalties["invalid"] = max(penalties["invalid"], env_step.penalty or 0.0)
    return penalties


def _make_auto_final_action(best_query: str) -> Dict[str, Any]:
    return {
        "tool": "final_answer",
        "final_query": best_query,
        "reason": "auto-finalize on last step using best query by ndcg_at_10",
    }


def _run_env_step(
    env: CommerceAgentEnv,
    action: Any,
    policy_out: Optional[Dict[str, Any]],
    *,
    auto_finish: bool = False,
    llm_called: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    try:
        result = env.step(action)
    except Exception as exc:
        record = {
            "step_id": env.state.step_count if env.state else 0,
            "prompt": (policy_out or {}).get("prompt", ""),
            "raw_output": (policy_out or {}).get("raw_output", json.dumps(action)),
            "parse_ok": False if policy_out else True,
            "action": action if isinstance(action, dict) else None,
            "invalid": True,
            "auto_finish": auto_finish,
            "llm_called": llm_called,
            "env_error": str(exc),
        }
        return None, record

    step = result["step"]
    parse_ok = True if auto_finish else bool((policy_out or {}).get("parse_ok"))
    penalties = _step_penalties(step, parse_ok)

    record = {
        "step_id": step.step_idx,
        "prompt": (policy_out or {}).get("prompt", "(auto-finalize; LLM not called on last step)"),
        "raw_output": (policy_out or {}).get("raw_output", json.dumps(action, ensure_ascii=False)),
        "parse_ok": parse_ok,
        "action": step.action if not step.invalid else (policy_out or {}).get("action", action),
        "invalid": step.invalid or (not parse_ok and not auto_finish),
        "invalid_reason": step.invalid_reason or (policy_out or {}).get("error"),
        "observation": step.observation,
        "ndcg_at_10": step.ndcg,
        "recall_at_10": step.recall,
        "delta_ndcg": step.delta_ndcg,
        "step_reward": step.step_reward,
        "penalties": penalties,
        "auto_finish": auto_finish,
        "llm_called": llm_called,
    }
    return result, record


def run_finish_aware_episode(
    env: CommerceAgentEnv,
    policy: QwenRolloutPolicy,
    sample: Dict[str, Any],
    sample_seed: int | None = None,
    strategy_name: str | None = None,
    strategy_instruction: str | None = None,
) -> Dict[str, Any]:
    """Run one finish-aware CommerceAgentEnv episode and return trajectory dict."""
    user_query = sample["user_query"]
    env.reset(
        qid=sample["qid"],
        original_query=user_query,
        target_items=sample["target_items"],
        mode="qwen_rollout_finish_aware",
    )

    init_metrics = env.evaluate_query(user_query)
    best_query_by_ndcg = user_query
    best_ndcg_at_10 = float(init_metrics["ndcg"])

    search_history: List[Dict[str, Any]] = []
    observation = ""
    rollout_steps: List[Dict[str, Any]] = []
    total_output_tokens = 0
    parse_ok_steps = 0
    total_policy_steps = 0
    llm_finished = False
    auto_finished = False

    while not env.state.done and env.state.step_count < env.max_steps:
        remaining = env.max_steps - env.state.step_count
        current_step = env.state.step_count + 1

        if remaining <= 1:
            auto_action = _make_auto_final_action(best_query_by_ndcg)
            result, step_record = _run_env_step(
                env,
                auto_action,
                policy_out=None,
                auto_finish=True,
                llm_called=False,
            )
            rollout_steps.append(step_record)
            auto_finished = True
            if result is not None:
                break
            continue

        policy_out = policy.act(
            user_query=user_query,
            search_history=search_history,
            observation=observation,
            max_steps=env.max_steps,
            current_step=current_step,
            remaining_steps=remaining,
            best_query_by_ndcg=best_query_by_ndcg,
            best_ndcg_at_10=best_ndcg_at_10,
            seed=(sample_seed + current_step) if sample_seed is not None else None,
            strategy_name=strategy_name,
            strategy_instruction=strategy_instruction,
        )
        total_output_tokens += policy.count_output_tokens(policy_out["raw_output"])
        total_policy_steps += 1

        if policy_out["parse_ok"]:
            parse_ok_steps += 1
            action_for_env = policy_out["action"]
        else:
            action_for_env = policy_out["raw_output"]

        if (
            policy_out["parse_ok"]
            and policy_out["action"]
            and policy_out["action"].get("tool") == "final_answer"
        ):
            llm_finished = True

        result, step_record = _run_env_step(
            env,
            action_for_env,
            policy_out,
            auto_finish=False,
            llm_called=True,
        )
        if result is None:
            rollout_steps.append(step_record)
            break

        rollout_steps.append(step_record)
        step = result["step"]

        if env.state.best_ndcg >= best_ndcg_at_10:
            best_ndcg_at_10 = float(env.state.best_ndcg)
            best_query_by_ndcg = env.state.best_query

        if step.action.get("tool") == "bm25_search" and not step.invalid:
            search_history.append(
                {
                    "query": step.action["query"],
                    "ndcg_at_10": step.ndcg,
                }
            )

        observation = result.get("observation") or step.observation or ""
        if result["done"]:
            break

    traj = env.build_trajectory()
    finished = bool(env.state.has_final_answer)
    final_query = traj.final_query or best_query_by_ndcg

    return {
        "qid": sample["qid"],
        "user_query": user_query,
        "target_items": sample["target_items"],
        "best_query_by_ndcg": best_query_by_ndcg,
        "best_ndcg_at_10": best_ndcg_at_10,
        "final_query": final_query,
        "steps": rollout_steps,
        "final_ndcg_at_10": traj.final_ndcg,
        "final_recall_at_10": traj.final_recall,
        "total_reward": float(traj.total_reward),
        "num_search_calls": traj.num_search_calls,
        "num_invalid_actions": env.state.invalid_count,
        "num_repeated_queries": env.state.repeat_count,
        "finished": finished,
        "llm_finished": llm_finished,
        "auto_finished": auto_finished,
        "terminated_reason": traj.terminated_reason,
        "output_tokens": total_output_tokens,
        "parse_ok_steps": parse_ok_steps,
        "total_policy_steps": total_policy_steps,
    }
