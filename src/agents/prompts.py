"""Prompt templates for Qwen rollout policy (Phase 1.5)."""

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
) -> str:
    obs = observation.strip() if observation.strip() else "(none yet — you have not searched)"
    return f"""You are an e-commerce search agent.

Your task is to improve product search for a user shopping query.
You can interact with a BM25 search engine for at most {max_steps} steps.

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

User shopping query:
{user_query}

Search history:
{format_search_history(search_history)}

Current observation:
{obs}

Now output the next JSON action:"""
