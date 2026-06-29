"""Parse JSON tool actions for CommerceAgentEnv."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

ALLOWED_TOOLS = frozenset({"bm25_search", "final_answer"})


class ActionParseError(Exception):
    """Raised when an action string or dict cannot be parsed."""

    def __init__(self, message: str, raw: Optional[str] = None):
        super().__init__(message)
        self.raw = raw


def _extract_json_blob(text: str) -> str:
    text = text.strip()
    if not text:
        raise ActionParseError("empty action", raw=text)

    if text.startswith("{") and text.endswith("}"):
        return text

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)

    raise ActionParseError("no JSON object found", raw=text)


def normalize_query(query: str) -> str:
    return " ".join(query.strip().lower().split())


def parse_action(action: Any) -> Dict[str, Any]:
    """
    Parse agent action into a normalized dict.

    Expected schema:
      bm25_search: {"tool": "bm25_search", "query": str, "topk": int?, "reason": str?}
      final_answer: {"tool": "final_answer", "final_query": str, "reason": str?}
                    or {"tool": "final_answer", "query": str, ...}
    """
    if isinstance(action, dict):
        data = action
        raw = json.dumps(action, ensure_ascii=False)
    elif isinstance(action, str):
        raw = action
        try:
            data = json.loads(_extract_json_blob(action))
        except json.JSONDecodeError as exc:
            raise ActionParseError(f"JSON decode failed: {exc}", raw=action) from exc
    else:
        raise ActionParseError(f"unsupported action type: {type(action)!r}")

    if not isinstance(data, dict):
        raise ActionParseError("action must be a JSON object", raw=raw)

    tool = data.get("tool")
    if tool not in ALLOWED_TOOLS:
        raise ActionParseError(f"tool not allowed: {tool!r}", raw=raw)

    if tool == "bm25_search":
        query = data.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ActionParseError("bm25_search requires non-empty query", raw=raw)
        topk = data.get("topk", 20)
        if not isinstance(topk, int) or topk <= 0:
            raise ActionParseError("topk must be a positive integer", raw=raw)
        return {
            "tool": "bm25_search",
            "query": query.strip(),
            "topk": topk,
            "reason": data.get("reason"),
        }

    final_query = data.get("final_query") or data.get("query")
    if not isinstance(final_query, str) or not final_query.strip():
        raise ActionParseError("final_answer requires final_query or query", raw=raw)
    return {
        "tool": "final_answer",
        "final_query": final_query.strip(),
        "reason": data.get("reason"),
    }
