"""
Phase 2.1: Clean 20-group smoke dataset builder.

Combines Phase 1.18h V2 candidates (excluding esci_val_3) with one Phase 1.18g
replacement to form a clean 20_g4 training candidate set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

FORBIDDEN_GROUP_IDS: Set[str] = {"esci_val_3"}
FORBIDDEN_LEARNABILITY_TYPES: Set[str] = {
    "bm25_retrieval_failure",
    "qrels_sparse_all_k_blind",
}


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _infer_learnability_type(replacement: Dict[str, Any]) -> str:
    best_rank = replacement.get("best_relevant_rank")
    if best_rank is not None and int(best_rank) <= 100:
        return "learnable_small_k"
    return "learnable_large_k"


def _format_replacement_row(replacement: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "group_id": replacement["group_id"],
        "original_query": replacement["original_query"],
        "source": "replacement_from_phase118g",
        "learnability_type": _infer_learnability_type(replacement),
        "num_relevant_docs": int(replacement.get("num_relevant_docs", 0)),
        "best_relevant_rank": replacement.get("best_relevant_rank"),
        "bm25_hit_at_1000": bool(replacement.get("bm25_hit_at_1000", True)),
        "recommended_for_phase2": True,
        "next_action": "keep_for_phase2",
        "strategy_prompt_version": "v2",
        "replacement_candidate_score": float(replacement.get("candidate_score", 0.0)),
        "target_items": list(replacement.get("target_items", [])),
    }


class Phase2SmokeDatasetBuilder:
    """Build and validate the Phase 2.1 clean 20-group smoke candidate set."""

    def __init__(
        self,
        target_groups: int = 20,
        exclude_group_ids: Optional[Sequence[str]] = None,
        drop_group_ids: Optional[Sequence[str]] = None,
    ):
        self.target_groups = target_groups
        self.exclude_group_ids: Set[str] = set(exclude_group_ids or []) | FORBIDDEN_GROUP_IDS
        self.drop_group_ids: Set[str] = set(drop_group_ids or [])

    def select_replacement(
        self,
        replacement_rows: List[Dict[str, Any]],
        existing_group_ids: Set[str],
        *,
        prefer_esci_val: bool = True,
    ) -> Dict[str, Any]:
        candidates = [
            row
            for row in replacement_rows
            if row.get("group_id") not in existing_group_ids
            and row.get("group_id") not in self.exclude_group_ids
        ]
        if not candidates:
            raise ValueError("no eligible replacement candidates found")

        if prefer_esci_val:
            val_candidates = [
                row for row in candidates if str(row.get("group_id", "")).startswith("esci_val_")
            ]
            if val_candidates:
                candidates = val_candidates

        candidates.sort(
            key=lambda r: (
                -float(r.get("candidate_score", 0.0)),
                str(r.get("group_id", "")),
            )
        )
        return candidates[0]

    def build_clean_set(
        self,
        candidate_v2_rows: List[Dict[str, Any]],
        replacement_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        kept: List[Dict[str, Any]] = []
        for row in candidate_v2_rows:
            gid = row.get("group_id")
            if gid in self.exclude_group_ids or gid in self.drop_group_ids:
                continue
            kept.append(dict(row))

        existing_ids = {r["group_id"] for r in kept}
        replacements_added: List[Dict[str, Any]] = []
        slots_needed = self.target_groups - len(kept)
        if slots_needed < 0:
            raise ValueError(
                f"too many candidate rows after filtering: {len(kept)} > target {self.target_groups}"
            )

        pool_ids = set(existing_ids)
        for _ in range(slots_needed):
            replacement = self.select_replacement(replacement_rows, pool_ids)
            replacement_row = _format_replacement_row(replacement)
            if len(replacements_added) == 0:
                replacement_row["replaces_group_id"] = "esci_val_3"
            replacements_added.append(replacement_row)
            kept.append(replacement_row)
            pool_ids.add(replacement_row["group_id"])

        kept.sort(key=lambda r: r["group_id"])

        validation = self.validate_clean_set(kept)
        if not validation["phase2_clean_set_ready"]:
            raise ValueError(f"clean set validation failed: {validation}")

        return {
            "rows": kept,
            "validation": validation,
            "replacements_added": replacements_added,
        }

    def validate_clean_set(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        group_ids = [r.get("group_id") for r in rows]
        unique_ids = set(group_ids)

        has_forbidden = any(gid in self.exclude_group_ids for gid in group_ids)
        has_bm25_failure = any(
            r.get("learnability_type") == "bm25_retrieval_failure" for r in rows
        )
        has_qrels_blind = any(
            r.get("learnability_type") == "qrels_sparse_all_k_blind" for r in rows
        )
        has_strategy_collapse = any(
            r.get("learnability_type") == "strategy_collapse"
            or r.get("v2_strategy_collapse") is True
            for r in rows
        )
        duplicate_group_ids = len(group_ids) != len(unique_ids)

        num_replacements = sum(
            1 for r in rows if r.get("source") == "replacement_from_phase118g"
        )

        ready = (
            len(rows) == self.target_groups
            and not has_forbidden
            and not has_bm25_failure
            and not has_qrels_blind
            and not has_strategy_collapse
            and not duplicate_group_ids
        )

        return {
            "num_groups": len(rows),
            "excluded_group_ids": sorted(self.exclude_group_ids),
            "num_replacements_added": num_replacements,
            "replacement_group_ids": [
                r["group_id"]
                for r in rows
                if r.get("source") == "replacement_from_phase118g"
            ],
            "has_bm25_failure": has_bm25_failure,
            "has_qrels_sparse_all_k_blind": has_qrels_blind,
            "has_strategy_collapse": has_strategy_collapse,
            "has_duplicate_group_ids": duplicate_group_ids,
            "has_excluded_group_ids": has_forbidden,
            "phase2_clean_set_ready": ready,
        }

    def write_outputs(
        self,
        rows: List[Dict[str, Any]],
        validation: Dict[str, Any],
        output_dir: str | Path,
    ) -> Dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        jsonl_path = out / "phase2_clean_20_groups.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fout:
            for row in rows:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

        summary_path = out / "phase2_clean_set_summary.json"
        summary = {
            "num_groups": validation["num_groups"],
            "excluded_group_ids": validation["excluded_group_ids"],
            "num_replacements_added": validation["num_replacements_added"],
            "replacement_group_ids": validation["replacement_group_ids"],
            "has_bm25_failure": validation["has_bm25_failure"],
            "has_qrels_sparse_all_k_blind": validation["has_qrels_sparse_all_k_blind"],
            "has_strategy_collapse": validation["has_strategy_collapse"],
            "phase2_clean_set_ready": validation["phase2_clean_set_ready"],
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return {"jsonl_path": jsonl_path, "summary_path": summary_path}


def load_clean_set_rows(path: str | Path) -> List[Dict[str, Any]]:
    return _load_jsonl(path)
