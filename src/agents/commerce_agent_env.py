"""Standalone multi-step commerce search agent environment (Phase 1 MVP)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from src.agents.action_parser import ActionParseError, normalize_query, parse_action
from src.agents.trajectory import EpisodeTrajectory, StepRecord
from src.reward.outcome_reward import compute_ndcg, compute_recall
from src.reward.process_reward import RewardConfig, compute_episode_reward
from src.tools.bm25_tool import BM25SearchTool


@dataclass
class EpisodeState:
    qid: str
    original_query: str
    target_items: List[str]
    rel_scores: List[float]
    step_count: int = 0
    search_count: int = 0
    invalid_count: int = 0
    repeat_count: int = 0
    done: bool = False
    has_final_answer: bool = False
    last_ndcg: float = 0.0
    best_ndcg: float = 0.0
    best_query: str = ""
    final_query: Optional[str] = None
    seen_queries: Set[str] = field(default_factory=set)
    delta_ndcg_list: List[float] = field(default_factory=list)
    steps: List[StepRecord] = field(default_factory=list)
    mode: str = "agent"


class CommerceAgentEnv:
    """
    Multi-step BM25 search environment for ESCI-style query refinement.

    Allowed tools:
      - bm25_search
      - final_answer
    """

    def __init__(
        self,
        search_tool: Optional[BM25SearchTool] = None,
        max_steps: int = 3,
        default_topk: int = 20,
        ndcg_k: int = 10,
        metric_topk: int = 100,
        reward_config: Optional[RewardConfig] = None,
    ):
        self.search_tool = search_tool or BM25SearchTool()
        self.max_steps = max_steps
        self.default_topk = default_topk
        self.ndcg_k = ndcg_k
        self.metric_topk = metric_topk
        self.reward_config = reward_config or RewardConfig(ndcg_k=ndcg_k)
        self.state: Optional[EpisodeState] = None

    def reset(
        self,
        *,
        qid: str,
        original_query: str,
        target_items: List[str],
        rel_scores: Optional[List[float]] = None,
        mode: str = "agent",
    ) -> EpisodeState:
        rel_scores = rel_scores or [1.0] * len(target_items)
        self.state = EpisodeState(
            qid=qid,
            original_query=original_query,
            target_items=list(target_items),
            rel_scores=list(rel_scores),
            best_query=original_query,
            mode=mode,
        )
        return self.state

    def _ensure_state(self) -> EpisodeState:
        if self.state is None:
            raise RuntimeError("call reset() before step()")
        return self.state

    def evaluate_query(self, query: str, topk: Optional[int] = None) -> Dict[str, Any]:
        state = self._ensure_state()
        obs_topk = topk or self.default_topk
        retrieve_k = max(obs_topk, self.metric_topk)
        hits = self.search_tool.search(query, topk=retrieve_k)
        retrieved = [doc_id for doc_id, _, _ in hits]
        ndcg = compute_ndcg(
            retrieved,
            state.target_items,
            self.ndcg_k,
            rel_scores=state.rel_scores,
        )
        recall = compute_recall(retrieved, state.target_items, self.ndcg_k)
        return {
            "query": query,
            "topk": obs_topk,
            "retrieved": retrieved,
            "hits": hits,
            "ndcg": ndcg,
            "recall": recall,
            "observation": self.search_tool.format_observation(query, hits, max_items=min(5, obs_topk)),
        }

    def step(self, action: Any) -> Dict[str, Any]:
        state = self._ensure_state()
        if state.done:
            raise RuntimeError("episode already finished")

        state.step_count += 1
        step_idx = state.step_count

        try:
            parsed = parse_action(action)
        except ActionParseError as exc:
            state.invalid_count += 1
            penalty = self.reward_config.invalid_penalty
            record = StepRecord(
                step_idx=step_idx,
                action={"raw": str(action)},
                invalid=True,
                invalid_reason=str(exc),
                penalty=penalty,
                step_reward=-penalty,
            )
            state.steps.append(record)
            return self._maybe_finish_after_step(record, terminated_reason="invalid_action")

        tool = parsed["tool"]

        if tool == "bm25_search":
            return self._step_search(state, step_idx, parsed)

        return self._step_final_answer(state, step_idx, parsed)

    def _step_search(self, state: EpisodeState, step_idx: int, action: Dict[str, Any]) -> Dict[str, Any]:
        query = action["query"]
        topk = action.get("topk", self.default_topk)
        norm = normalize_query(query)

        if norm in state.seen_queries:
            state.repeat_count += 1
            penalty = self.reward_config.repeat_penalty
            record = StepRecord(
                step_idx=step_idx,
                action=action,
                invalid=True,
                invalid_reason="repeated_query",
                penalty=penalty,
                step_reward=-penalty,
            )
            state.steps.append(record)
            return self._maybe_finish_after_step(record, terminated_reason="repeated_query")

        metrics = self.evaluate_query(query, topk=topk)
        state.seen_queries.add(norm)
        state.search_count += 1

        delta_ndcg = metrics["ndcg"] - state.last_ndcg
        state.delta_ndcg_list.append(delta_ndcg)
        step_reward = delta_ndcg - self.reward_config.search_cost

        state.last_ndcg = metrics["ndcg"]
        if metrics["ndcg"] >= state.best_ndcg:
            state.best_ndcg = metrics["ndcg"]
            state.best_query = query

        record = StepRecord(
            step_idx=step_idx,
            action=action,
            observation=metrics["observation"],
            ndcg=metrics["ndcg"],
            recall=metrics["recall"],
            delta_ndcg=delta_ndcg,
            step_reward=step_reward,
            penalty=self.reward_config.search_cost,
        )
        state.steps.append(record)
        return self._maybe_finish_after_step(record, terminated_reason="max_steps" if state.step_count >= self.max_steps else "continue")

    def _step_final_answer(self, state: EpisodeState, step_idx: int, action: Dict[str, Any]) -> Dict[str, Any]:
        final_query = action["final_query"]
        metrics = self.evaluate_query(final_query, topk=self.default_topk)
        state.has_final_answer = True
        state.final_query = final_query
        state.done = True

        delta_ndcg = metrics["ndcg"] - state.last_ndcg
        state.delta_ndcg_list.append(delta_ndcg)
        state.last_ndcg = metrics["ndcg"]
        if metrics["ndcg"] >= state.best_ndcg:
            state.best_ndcg = metrics["ndcg"]
            state.best_query = final_query

        record = StepRecord(
            step_idx=step_idx,
            action=action,
            observation=metrics["observation"],
            ndcg=metrics["ndcg"],
            recall=metrics["recall"],
            delta_ndcg=delta_ndcg,
            step_reward=delta_ndcg,
        )
        state.steps.append(record)
        return {
            "done": True,
            "observation": metrics["observation"],
            "info": {"terminated_reason": "final_answer"},
            "step": record,
        }

    def _maybe_finish_after_step(self, record: StepRecord, terminated_reason: str) -> Dict[str, Any]:
        state = self._ensure_state()
        if state.step_count >= self.max_steps and not state.done:
            state.done = True
            if not state.has_final_answer:
                state.final_query = state.best_query or state.original_query
            terminated_reason = "max_steps"
        return {
            "done": state.done,
            "observation": record.observation,
            "info": {"terminated_reason": terminated_reason},
            "step": record,
        }

    def build_trajectory(self, baseline_ndcg: Optional[float] = None) -> EpisodeTrajectory:
        state = self._ensure_state()
        final_query = state.final_query or state.best_query or state.original_query
        final_metrics = self.evaluate_query(final_query, topk=self.default_topk)

        breakdown = compute_episode_reward(
            final_ndcg=final_metrics["ndcg"],
            delta_ndcg_list=state.delta_ndcg_list,
            num_search_calls=state.search_count,
            num_invalid=state.invalid_count,
            num_repeated=state.repeat_count,
            has_final_answer=state.has_final_answer,
            config=self.reward_config,
        )

        terminated_reason = "final_answer" if state.has_final_answer else "max_steps_without_final"
        if state.invalid_count and not state.search_count and not state.has_final_answer:
            terminated_reason = "invalid_only"

        return EpisodeTrajectory(
            qid=state.qid,
            original_query=state.original_query,
            target_items=state.target_items,
            mode=state.mode,
            steps=state.steps,
            final_query=final_query,
            final_ndcg=final_metrics["ndcg"],
            final_recall=final_metrics["recall"],
            baseline_ndcg=baseline_ndcg,
            num_search_calls=state.search_count,
            penalties=breakdown.penalties,
            total_penalty=breakdown.total_penalty,
            process_reward_sum=breakdown.process_reward_sum,
            final_reward=breakdown.final_reward,
            total_reward=breakdown.total_reward,
            terminated_reason=terminated_reason,
        )
