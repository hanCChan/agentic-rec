"""Simple rule-based policy for Phase 1 env smoke tests."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _refine_query(original_query: str, step_idx: int, last_ndcg: float) -> str:
    """Heuristic query refinement for smoke testing."""
    q = original_query.strip()
    lower = q.lower()

    if step_idx == 1:
        return q

    tokens = re.findall(r"[A-Za-z0-9]+", q)
    if "without" in lower or " not " in f" {lower} ":
        parts = re.split(r"\bwithout\b|\bnot\b", q, flags=re.IGNORECASE)
        positive = parts[0].strip(" ,.")
        negative = parts[1].strip(" ,.") if len(parts) > 1 else ""
        refined = positive
        if negative:
            refined = f"{positive} NOT {negative}"
        return refined

    if len(tokens) >= 2:
        return " AND ".join(tokens[: min(6, len(tokens))])

    return f"{q} original"


class RuleSearchPolicy:
    """At most max_steps-1 searches, then final_answer with best-effort refinement."""

    def __init__(self, max_steps: int = 3, default_topk: int = 20):
        self.max_steps = max_steps
        self.default_topk = default_topk
        self._last_ndcg = 0.0
        self._best_query = ""
        self._best_ndcg = -1.0
        self._search_round = 0

    def reset(self, original_query: str) -> None:
        self._last_ndcg = 0.0
        self._best_query = original_query
        self._best_ndcg = -1.0
        self._search_round = 0

    def observe(self, *, query: str, ndcg: float) -> None:
        self._last_ndcg = ndcg
        self._search_round += 1
        if ndcg >= self._best_ndcg:
            self._best_ndcg = ndcg
            self._best_query = query

    def next_action(self, original_query: str, step_idx: int, remaining_steps: int) -> Dict[str, Any]:
        if remaining_steps <= 1:
            return {
                "tool": "final_answer",
                "final_query": self._best_query or original_query,
                "reason": "rule policy finalize with best query so far",
            }

        refined = _refine_query(original_query, self._search_round + 1, self._last_ndcg)
        return {
            "tool": "bm25_search",
            "query": refined,
            "topk": self.default_topk,
            "reason": f"rule refine step {self._search_round + 1}",
        }


def baseline_original_query_action(original_query: str, topk: int = 20) -> Dict[str, Any]:
    return {
        "tool": "bm25_search",
        "query": original_query,
        "topk": topk,
        "reason": "single-shot baseline",
    }
