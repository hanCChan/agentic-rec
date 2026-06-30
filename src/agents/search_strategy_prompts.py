"""
Phase 1.18d search strategy definitions for strategy-controlled rollout.

Each strategy injects a distinct rewrite objective into the rollout prompt.
The environment and action JSON schema remain unchanged.
"""

from __future__ import annotations

from typing import Dict, List

SEARCH_STRATEGIES: Dict[str, Dict[str, str]] = {
    "exact_match": {
        "description": "Preserve exact product intent with minimal generalization.",
        "instruction": (
            "Search strategy: exact_match.\n\n"
            "Rewrite the query to preserve the exact product intent.\n"
            "Keep important product type, brand, model, size, color, material, compatibility, "
            "and explicit constraints.\n"
            "Avoid adding speculative attributes that are not implied by the user query.\n"
            "Prefer concise product-search wording."
        ),
    },
    "attribute_expansion": {
        "description": "Expand implied product attributes with consistent ecommerce terms.",
        "instruction": (
            "Search strategy: attribute_expansion.\n\n"
            "Rewrite the query by expanding useful product attributes.\n"
            "You may add likely search attributes such as replacement, compatible, heavy duty, "
            "outdoor, waterproof, adjustable, pack, size, or material only when they are "
            "consistent with the user query.\n"
            "Do not change the product category.\n"
            "Do not violate negative constraints."
        ),
    },
    "broad_recall": {
        "description": "Broaden the query to improve recall while keeping core intent.",
        "instruction": (
            "Search strategy: broad_recall.\n\n"
            "Rewrite the query to improve recall.\n"
            "Use a broader product category while preserving the core intent.\n"
            "Remove overly specific wording if it may hurt retrieval.\n"
            "Keep essential constraints, but avoid unnecessary details."
        ),
    },
    "constraint_preserving": {
        "description": "Strictly preserve negative and compatibility constraints.",
        "instruction": (
            "Search strategy: constraint_preserving.\n\n"
            "Rewrite the query while strictly preserving all constraints.\n"
            "Pay special attention to negative constraints such as without, no, not, non-, "
            "excluding, replacement only, compatible with, under a price, size, color, "
            "or use-case constraints.\n"
            "Do not broaden the query in a way that violates constraints."
        ),
    },
}

DEFAULT_STRATEGY_ORDER: List[str] = [
    "exact_match",
    "attribute_expansion",
    "broad_recall",
    "constraint_preserving",
]


def get_strategy(name: str) -> Dict[str, str]:
    if name not in SEARCH_STRATEGIES:
        raise ValueError(f"unknown search strategy: {name}")
    return SEARCH_STRATEGIES[name]


def validate_strategies(strategies: List[str]) -> None:
    unknown = [s for s in strategies if s not in SEARCH_STRATEGIES]
    if unknown:
        raise ValueError(f"unknown strategies: {unknown}")
