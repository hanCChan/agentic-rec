"""Outcome metrics: NDCG and Recall."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence, Union


@lru_cache(maxsize=1)
def _ndcg_fn():
    from src.rec_r1_bridge import get_ndcg_at_k

    rec_r1_root = Path(__file__).resolve().parents[2] / "Rec-R1"
    return get_ndcg_at_k(rec_r1_root)


def compute_recall(retrieved: Sequence[str], targets: Union[str, Sequence[str]], k: int) -> float:
    retrieved_k = list(retrieved)[:k]
    if isinstance(targets, str):
        target_set = {targets}
    else:
        target_set = set(targets)
    if not target_set:
        return 0.0
    hits = sum(1 for doc in retrieved_k if doc in target_set)
    return hits / len(target_set)


def compute_ndcg(
    retrieved: Sequence[str],
    targets: Union[str, Sequence[str]],
    k: int,
    rel_scores: Iterable[float] | None = None,
) -> float:
    ndcg_at_k = _ndcg_fn()
    target_list: List[str]
    if isinstance(targets, str):
        target_list = [targets]
    else:
        target_list = list(targets)

    scores = list(rel_scores) if rel_scores is not None else [1.0] * len(target_list)
    return float(ndcg_at_k(list(retrieved), target_list, k, rel_scores=scores))
