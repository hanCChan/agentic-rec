"""Prompt templates for Qwen rollout policy (Phase 1.5 / 1.6)."""

from __future__ import annotations

from typing import Any, Dict, List


def format_search_history(search_history: List[Dict[str, Any]]) -> str:
    if not search_history:
        return "(none)"
    lines = []
    for i, item in enumerate(search_history, start=1):
        query = item.get("query", "")
        ndcg = item.get("ndcg_at_10")
        if ndcg is not None:
            lines.append(f"{i}. query={query!r}, ndcg_at_10={ndcg:.4f}")
        else:
            lines.append(f"{i}. query={query!r}")
    return "\n".join(lines)


def build_rollout_prompt(
    *,
    user_query: str,
    search_history: List[Dict[str, Any]],
    observation: str,
    max_steps: int,
    current_step: int = 1,
    remaining_steps: int | None = None,
    best_query_by_ndcg: str | None = None,
    best_ndcg_at_10: float | None = None,
) -> str:
    obs = observation.strip() if observation.strip() else "(none yet — you have not searched)"
    remaining = remaining_steps if remaining_steps is not None else max(max_steps - current_step + 1, 0)
    best_q = best_query_by_ndcg or user_query
    best_ndcg_str = f"{best_ndcg_at_10:.4f}" if best_ndcg_at_10 is not None else "unknown"

    last_step_rule = ""
    if remaining <= 1:
        last_step_rule = """
- THIS IS YOUR LAST STEP. You MUST output final_answer now.
- Do NOT output bm25_search on the last step.
- Set final_query to the best refined query you found."""
    elif remaining == 2:
        last_step_rule = """
- You have only 2 steps left. Prefer one more bm25_search, then you must finish with final_answer."""

    return f"""You are an e-commerce search agent.

Your task is to improve product search for a user shopping query.
You can interact with a BM25 search engine for at most {max_steps} steps.

Current step: {current_step} of {max_steps}
Remaining steps: {remaining}
Best query so far (by ndcg_at_10): {best_q!r} (ndcg_at_10={best_ndcg_str})

At each step, you must output exactly one valid JSON object.

Allowed actions:

1. Search:
{{"tool": "bm25_search", "query": "...", "topk": 20}}

2. Finish:
{{"tool": "final_answer", "final_query": "...", "reason": "..."}}

Rules:
- Output JSON only.
- Do not use Markdown.
- Do not wrap JSON in ``` blocks.
- Do not output explanations outside JSON.
- Keep queries concise and product-search oriented.
- Preserve important entities, brands, models, sizes, colors, materials, compatibility constraints, and negative constraints.
- If search results contain irrelevant accessories or wrong product types, refine the query to exclude them.
- If the current query is already good, finish with final_answer.
- Do not repeat the exact same query.
- Every episode MUST end with final_answer before steps run out.{last_step_rule}

User shopping query:
{user_query}

Search history:
{format_search_history(search_history)}

Current observation:
{obs}

Now output the next JSON action:"""
