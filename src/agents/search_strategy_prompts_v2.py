"""
Phase 1.18h search strategy prompt V2 for collapse cases.

Stronger anti-collapse constraints while preserving the same 4 strategies
and unchanged action JSON schema.
"""

from __future__ import annotations

from typing import Dict, List

from src.agents.search_strategy_prompts import DEFAULT_STRATEGY_ORDER

GLOBAL_ANTI_COLLAPSE_RULES = """
You are assigned a specific search strategy.
Your rewritten query must be meaningfully different from the other strategies.
Do not simply copy the original query unless the strategy is exact_match.
Do not converge to the same final query for all strategies.
Preserve the user's core shopping intent.
Never violate explicit negative constraints such as without, no, not, non-, excluding.
""".strip()

SEARCH_STRATEGIES_V2: Dict[str, Dict[str, str]] = {
    "exact_match": {
        "description": "Stay closest to the user's original wording.",
        "instruction": (
            f"{GLOBAL_ANTI_COLLAPSE_RULES}\n\n"
            "Search strategy: exact_match.\n\n"
            "Produce a concise query that stays closest to the user's original wording.\n"
            "Keep the exact product type, quantity, material, size, color, compatibility, "
            "and negative constraints.\n"
            "Do not add broad category terms unless they are already implied.\n"
            "This strategy should be the most literal among all strategies.\n\n"
            "Example:\n"
            "Input: # 10 self-seal envelopes without window\n"
            "Output: #10 self seal envelopes without window"
        ),
    },
    "attribute_expansion": {
        "description": "Expand attributes without copying exact_match.",
        "instruction": (
            f"{GLOBAL_ANTI_COLLAPSE_RULES}\n\n"
            "Search strategy: attribute_expansion.\n\n"
            "Expand the query with product attributes that are useful for ecommerce search.\n"
            "Add terms such as business, security, adhesive, peel seal, gummed, pack, white, "
            "office, mailing, or replacement only if consistent with the query.\n"
            "Do not remove negative constraints.\n"
            "This strategy must add attribute terms beyond the exact wording.\n\n"
            "Example:\n"
            "Input: # 10 self-seal envelopes without window\n"
            "Output: #10 white business self seal security envelopes no window office mailing"
        ),
    },
    "broad_recall": {
        "description": "Broader category wording for recall.",
        "instruction": (
            f"{GLOBAL_ANTI_COLLAPSE_RULES}\n\n"
            "Search strategy: broad_recall.\n\n"
            "Use a broader product category to improve recall.\n"
            "Remove overly specific wording if it may hurt retrieval, but keep essential constraints.\n"
            "Use category-level words and common synonyms.\n"
            "This strategy should be broader than exact_match and shorter than attribute_expansion.\n\n"
            "Example:\n"
            "Input: # 10 self-seal envelopes without window\n"
            "Output: business envelopes no window self seal"
        ),
    },
    "constraint_preserving": {
        "description": "Emphasize negative and size constraints.",
        "instruction": (
            f"{GLOBAL_ANTI_COLLAPSE_RULES}\n\n"
            "Search strategy: constraint_preserving.\n\n"
            "Focus on preserving constraints exactly.\n"
            "Repeat or restate negative constraints using common ecommerce synonyms.\n"
            'For "without window", use terms like no window, windowless, security tint if appropriate.\n'
            "Do not broaden in a way that may retrieve products with windows.\n"
            "This strategy should emphasize constraints more than all other strategies.\n\n"
            "Example:\n"
            "Input: # 10 self-seal envelopes without window\n"
            "Output: #10 self seal windowless envelopes no window security tint"
        ),
    },
}

DEFAULT_STRATEGY_ORDER_V2: List[str] = list(DEFAULT_STRATEGY_ORDER)


def get_strategy_v2(name: str) -> Dict[str, str]:
    if name not in SEARCH_STRATEGIES_V2:
        raise ValueError(f"unknown search strategy v2: {name}")
    return SEARCH_STRATEGIES_V2[name]


def validate_strategies_v2(strategies: List[str]) -> None:
    unknown = [s for s in strategies if s not in SEARCH_STRATEGIES_V2]
    if unknown:
        raise ValueError(f"unknown strategies v2: {unknown}")
