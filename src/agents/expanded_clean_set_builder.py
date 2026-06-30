"""
Phase 2.5b: Expanded clean set builder from ESCI val rescan.

Scans ESCI val with BM25 pre-filter, runs v2 strategy rollout on candidates,
applies Phase 1.18g/h gates, and splits train / held-out clean groups.
"""

from __future__ import annotations

import importlib.util
import json
import traceback
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Set

import pandas as pd

from src.agents.bm25_failure_cleanup import BM25FailureCleanup
from src.agents.commerce_agent_env import CommerceAgentEnv
from src.agents.periodic_fresh_eval import compute_strategy_distribution
from src.agents.qwen_rollout_policy import QwenRolloutPolicy
from src.agents.search_strategy_prompts_v2 import get_strategy_v2
from src.agents.strategy_episode_runner import StrategyEpisodeRunner
from src.tools.bm25_tool import BM25SearchTool

FORBIDDEN_GROUP_IDS: Set[str] = {"esci_val_3"}
DEFAULT_GATE_THRESHOLDS = {
    "retrieval_quality_spread_group_rate_min": 0.75,
    "zero_std_group_rate_max": 0.25,
    "penalty_only_spread_group_rate_max": 0.00,
    "parse_success_rate_min": 0.95,
    "invalid_action_rate_max": 0.05,
}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_all_esci_samples(parquet_path: Path) -> List[Dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    samples = []
    for idx, row in df.iterrows():
        targets = [str(x) for x in row["item_id"]]
        samples.append(
            {
                "qid": f"{row.get('data_source', 'esci')}_{idx}",
                "group_id": f"{row.get('data_source', 'esci')}_{idx}",
                "user_query": str(row["query"]),
                "original_query": str(row["query"]),
                "target_items": targets,
            }
        )
    return samples


class ExpandedCleanSetBuilder:
    """Build expanded train / held-out clean sets from ESCI val rescan."""

    def __init__(
        self,
        *,
        target_train_groups: int = 50,
        target_heldout_groups: int = 20,
        replacement_pool_size: int = 300,
        group_size: int = 4,
        exclude_group_ids: Optional[Set[str]] = None,
        gate_thresholds: Optional[Dict[str, float]] = None,
        k_list: Optional[List[int]] = None,
        candidate_name: str = "reward_largek_mix_1000",
    ):
        self.target_train_groups = target_train_groups
        self.target_heldout_groups = target_heldout_groups
        self.replacement_pool_size = replacement_pool_size
        self.group_size = group_size
        self.exclude_group_ids = set(exclude_group_ids or []) | FORBIDDEN_GROUP_IDS
        self.gate_thresholds = gate_thresholds or dict(DEFAULT_GATE_THRESHOLDS)
        self.k_list = k_list or [10, 50, 100, 1000]
        self.candidate_name = candidate_name
        self.cleanup = BM25FailureCleanup(k_list=self.k_list)

    def scan_bm25_pool(
        self,
        data_path: Path,
        search_tool: Any,
        *,
        max_pool: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        samples = load_all_esci_samples(data_path)
        eligible = [
            s
            for s in samples
            if s["group_id"] not in self.exclude_group_ids
        ]
        pool = self.cleanup.build_replacement_pool(eligible, search_tool, exclude_queries=[])
        limit = max_pool or self.replacement_pool_size
        return pool[:limit]

    def _load_v2_module(self, root: Path) -> Any:
        script_path = root / "scripts/smoke_strategy_prompt_v2.py"
        spec = importlib.util.spec_from_file_location("smoke_strategy_prompt_v2", script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {script_path}")
        v2_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(v2_module)
        return v2_module

    def _create_runner(
        self,
        *,
        root: Path,
        model_path: str,
        temperature: float,
        top_p: float,
        topk: int,
        seed: int,
    ) -> StrategyEpisodeRunner:
        rec_r1 = root / "Rec-R1"

        def env_factory() -> CommerceAgentEnv:
            search_tool = BM25SearchTool(rec_r1_root=rec_r1)
            return CommerceAgentEnv(
                search_tool=search_tool,
                max_steps=3,
                default_topk=topk,
            )

        policy = QwenRolloutPolicy(
            model_path=model_path,
            temperature=temperature,
            top_p=top_p,
            max_tokens=256,
        )
        return StrategyEpisodeRunner(
            env_factory=env_factory,
            policy=policy,
            strategies=["exact_match", "attribute_expansion", "broad_recall", "constraint_preserving"],
            max_steps=3,
            topk=topk,
            base_seed=seed,
            sampling_temperature=temperature,
            sampling_top_p=top_p,
            strategy_getter=get_strategy_v2,
            strategy_version="v2",
        )

    def rollout_single_group(
        self,
        sample: Dict[str, Any],
        *,
        runner: StrategyEpisodeRunner,
        v2_module: Any,
        seed: int,
    ) -> Dict[str, Any]:
        rollout_sample = {
            "qid": sample["group_id"],
            "user_query": sample["original_query"],
            "target_items": list(sample.get("target_items", [])),
        }
        try:
            group = runner.run_group(rollout_sample)
        except Exception as exc:
            return {
                "group_id": sample["group_id"],
                "passed": False,
                "reject_reason": str(exc),
                "learnability_type": "rollout_failed",
                "trace": traceback.format_exc(),
            }

        rollout_records = list(group["records"])
        gm = group["group_metrics"]
        parse_rate = float(gm.get("json_parse_success_rate", 0.0))
        invalid_rate = float(gm.get("invalid_action_rate", 0.0))

        tmp_dir = Path("/tmp") / f"phase25_gate_{sample['group_id']}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        rollout_path = tmp_dir / "rollout_records.jsonl"
        _write_jsonl(rollout_path, rollout_records)
        post = v2_module.run_post_analysis(
            rollout_path,
            tmp_dir,
            k_list=self.k_list,
            candidate_name=self.candidate_name,
        )

        coverage_rows = _load_jsonl(tmp_dir / "query_relevance_coverage.jsonl")
        metric_rows = _load_jsonl(tmp_dir / "group_metric_spread_by_k.jsonl")
        lk_rows = [
            r
            for r in _load_jsonl(tmp_dir / "large_k_candidate_group_reports.jsonl")
            if r.get("candidate_name") == self.candidate_name
        ]

        coverage = coverage_rows[0] if coverage_rows else {}
        metric = metric_rows[0] if metric_rows else {}
        large_k = lk_rows[0] if lk_rows else {}

        learnability = self.cleanup.classify_group_learnability(
            {
                "group_id": sample["group_id"],
                "original_query": sample["original_query"],
                "query_coverage": coverage,
                "metric_report": metric,
                "large_k_report": large_k,
            }
        )

        strategy_distribution = compute_strategy_distribution(rollout_records)
        reward_std = float(gm.get("reward_std", 0.0))
        candidate_spread = float(large_k.get("candidate_reward_spread", 0.0))

        passed = (
            learnability["keep_for_phase2"]
            and learnability["learnability_type"]
            not in {"bm25_retrieval_failure", "qrels_sparse_all_k_blind", "strategy_collapse"}
            and parse_rate >= self.gate_thresholds["parse_success_rate_min"]
            and invalid_rate <= self.gate_thresholds["invalid_action_rate_max"]
            and candidate_spread > 1e-6
        )

        print(
            f"[phase25] {sample['group_id']} "
            f"learnability={learnability['learnability_type']} "
            f"spread={candidate_spread:.4f} passed={passed}"
        )

        return {
            "group_id": sample["group_id"],
            "original_query": sample["original_query"],
            "target_items": list(sample.get("target_items", [])),
            "source": "esci_val_rescan",
            "learnability_type": learnability["learnability_type"],
            "num_relevant_docs": learnability.get("num_relevant_docs", 0),
            "best_relevant_rank": learnability.get("best_relevant_rank"),
            "bm25_hit_at_1000": learnability.get("bm25_hit_at_1000", True),
            "recommended_for_phase2": passed,
            "next_action": learnability.get("next_action"),
            "failure_reason": learnability.get("failure_reason", ""),
            "parse_success_rate": parse_rate,
            "invalid_action_rate": invalid_rate,
            "reward_std": reward_std,
            "candidate_reward_spread": candidate_spread,
            "strategy_distribution": strategy_distribution,
            "strategy_prompt_version": "v2",
            "replacement_candidate_score": float(sample.get("candidate_score", 0.0)),
            "passed": passed,
            "reject_reason": None if passed else learnability.get("failure_reason") or learnability["learnability_type"],
            "post_summary": post.get("large_k_summary", {}),
            "rollout_records": rollout_records,
            "group_metrics": gm,
        }

    def evaluate_set_gate(
        self,
        rows: List[Dict[str, Any]],
        rollout_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not rows:
            return {
                "num_groups": 0,
                "gate_passed": False,
                "blocking_reason": "empty set",
            }

        ng = len(rows) or 1
        zero_std = sum(1 for r in rows if float(r.get("reward_std", 0.0)) <= 1e-6)
        spread = sum(1 for r in rows if float(r.get("candidate_reward_spread", 0.0)) > 1e-6)

        parse_rates = [float(r.get("parse_success_rate", 0.0)) for r in rows]
        invalid_rates = [float(r.get("invalid_action_rate", 0.0)) for r in rows]

        bm25_fail = sum(
            1 for r in rows if r.get("learnability_type") == "bm25_retrieval_failure"
        )
        qrels_blind = sum(
            1 for r in rows if r.get("learnability_type") == "qrels_sparse_all_k_blind"
        )
        strategy_collapse = sum(
            1 for r in rows if r.get("learnability_type") == "strategy_collapse"
        )

        zero_std_rate = zero_std / ng
        spread_rate = spread / ng
        parse_rate = float(mean(parse_rates)) if parse_rates else 0.0
        invalid_rate = float(mean(invalid_rates)) if invalid_rates else 0.0
        strategy_distribution = compute_strategy_distribution(rollout_records)

        t = self.gate_thresholds
        gate_passed = (
            bm25_fail == 0
            and qrels_blind == 0
            and strategy_collapse == 0
            and spread_rate >= t["retrieval_quality_spread_group_rate_min"]
            and zero_std_rate <= t["zero_std_group_rate_max"]
            and parse_rate >= t["parse_success_rate_min"]
            and invalid_rate <= t["invalid_action_rate_max"]
            and strategy_distribution != {"unknown": 1.0}
        )

        blocking_reasons = []
        if bm25_fail:
            blocking_reasons.append(f"bm25_retrieval_failure={bm25_fail}")
        if qrels_blind:
            blocking_reasons.append(f"qrels_sparse_all_k_blind={qrels_blind}")
        if strategy_collapse:
            blocking_reasons.append(f"strategy_collapse={strategy_collapse}")
        if spread_rate < t["retrieval_quality_spread_group_rate_min"]:
            blocking_reasons.append(
                f"retrieval_quality_spread_group_rate={spread_rate:.3f} < {t['retrieval_quality_spread_group_rate_min']}"
            )
        if zero_std_rate > t["zero_std_group_rate_max"]:
            blocking_reasons.append(
                f"zero_std_group_rate={zero_std_rate:.3f} > {t['zero_std_group_rate_max']}"
            )
        if parse_rate < t["parse_success_rate_min"]:
            blocking_reasons.append(
                f"parse_success_rate={parse_rate:.3f} < {t['parse_success_rate_min']}"
            )
        if invalid_rate > t["invalid_action_rate_max"]:
            blocking_reasons.append(
                f"invalid_action_rate={invalid_rate:.3f} > {t['invalid_action_rate_max']}"
            )
        if strategy_distribution == {"unknown": 1.0}:
            blocking_reasons.append("strategy_distribution all unknown")

        return {
            "num_groups": len(rows),
            "gate_passed": gate_passed,
            "retrieval_quality_spread_group_rate": spread_rate,
            "zero_std_group_rate": zero_std_rate,
            "penalty_only_spread_group_rate": 0.0,
            "parse_success_rate": parse_rate,
            "invalid_action_rate": invalid_rate,
            "bm25_failure_count": bm25_fail,
            "qrels_sparse_all_k_blind_count": qrels_blind,
            "strategy_collapse_count": strategy_collapse,
            "strategy_distribution": strategy_distribution,
            "blocking_reason": "; ".join(blocking_reasons) if blocking_reasons else None,
        }

    def build(
        self,
        *,
        data_path: Path,
        output_dir: Path,
        model_path: str,
        temperature: float,
        top_p: float,
        topk: int,
        seed: int,
        root: Path,
        search_tool: Any,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        total_needed = self.target_train_groups + self.target_heldout_groups
        scan_log = out / "scan_progress.jsonl"
        rejected_log = out / "rejected_groups.jsonl"

        pool = self.scan_bm25_pool(data_path, search_tool)
        _write_jsonl(out / "bm25_replacement_pool.jsonl", pool)

        v2_module = self._load_v2_module(root)
        runner = self._create_runner(
            root=root,
            model_path=model_path,
            temperature=temperature,
            top_p=top_p,
            topk=topk,
            seed=seed,
        )
        clean_rows: List[Dict[str, Any]] = []
        all_rollout_records: List[Dict[str, Any]] = []
        scanned = 0

        for sample in pool:
            if len(clean_rows) >= total_needed:
                break
            scanned += 1
            gid = sample["group_id"]
            print(f"[phase25] scanning {gid} ({len(clean_rows)}/{total_needed} clean)")

            try:
                result = self.rollout_single_group(
                    sample,
                    runner=runner,
                    v2_module=v2_module,
                    seed=seed + scanned,
                )
            except Exception as exc:
                result = {
                    "group_id": gid,
                    "passed": False,
                    "reject_reason": str(exc),
                    "trace": traceback.format_exc(),
                }

            _append_jsonl(scan_log, {"group_id": gid, **{k: v for k, v in result.items() if k != "rollout_records"}})

            if result.get("passed"):
                row = {k: v for k, v in result.items() if k not in {"rollout_records", "group_metrics", "post_summary"}}
                clean_rows.append(row)
                all_rollout_records.extend(result.get("rollout_records", []))
                if progress_callback:
                    progress_callback(result)
            else:
                _append_jsonl(
                    rejected_log,
                    {
                        "group_id": gid,
                        "reject_reason": result.get("reject_reason"),
                        "learnability_type": result.get("learnability_type"),
                    },
                )

        train_rows = clean_rows[: self.target_train_groups]
        heldout_rows = clean_rows[
            self.target_train_groups : self.target_train_groups + self.target_heldout_groups
        ]

        train_ids = {r["group_id"] for r in train_rows}
        heldout_ids = {r["group_id"] for r in heldout_rows}
        overlap = train_ids & heldout_ids

        train_records = [r for r in all_rollout_records if r.get("group_id") in train_ids]
        heldout_records = [r for r in all_rollout_records if r.get("group_id") in heldout_ids]

        train_gate = self.evaluate_set_gate(train_rows, train_records)
        heldout_gate = self.evaluate_set_gate(heldout_rows, heldout_records)

        strategy_check = compute_strategy_distribution(all_rollout_records)

        enough = (
            len(train_rows) >= self.target_train_groups
            and len(heldout_rows) >= self.target_heldout_groups
            and not overlap
        )

        expanded_ready = (
            enough
            and train_gate["gate_passed"]
            and heldout_gate["gate_passed"]
        )

        summary = {
            "phase": "2.5b",
            "mode": "expanded_clean_set",
            "target_train_groups": self.target_train_groups,
            "target_heldout_groups": self.target_heldout_groups,
            "train_clean_groups": len(train_rows),
            "heldout_clean_groups": len(heldout_rows),
            "num_scanned_from_pool": scanned,
            "replacement_pool_size": len(pool),
            "train_heldout_overlap": sorted(overlap),
            "expanded_clean_set_ready": expanded_ready,
            "train_gate_passed": train_gate["gate_passed"],
            "heldout_gate_passed": heldout_gate["gate_passed"],
            "strategy_distribution_check": strategy_check,
            "bm25_failure_count": train_gate["bm25_failure_count"] + heldout_gate["bm25_failure_count"],
            "qrels_sparse_all_k_blind_count": (
                train_gate["qrels_sparse_all_k_blind_count"]
                + heldout_gate["qrels_sparse_all_k_blind_count"]
            ),
            "strategy_collapse_count": (
                train_gate["strategy_collapse_count"] + heldout_gate["strategy_collapse_count"]
            ),
            "candidate_name": self.candidate_name,
            "gate_thresholds": self.gate_thresholds,
        }

        if not enough:
            why_path = out / "why_not_enough_clean_groups.md"
            why_path.write_text(
                build_why_not_enough_report(summary, train_rows, heldout_rows, scanned),
                encoding="utf-8",
            )
            summary["blocking_reason"] = (
                f"only found {len(train_rows)} train + {len(heldout_rows)} heldout "
                f"(need {self.target_train_groups}+{self.target_heldout_groups})"
            )

        _write_jsonl(out / "train_clean_50_groups.jsonl", train_rows)
        _write_jsonl(out / "heldout_clean_20_groups.jsonl", heldout_rows)
        (out / "expanded_clean_set_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out / "train_clean_gate_summary.json").write_text(
            json.dumps(train_gate, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out / "heldout_clean_gate_summary.json").write_text(
            json.dumps(heldout_gate, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out / "strategy_distribution_check.json").write_text(
            json.dumps({"strategy_distribution": strategy_check}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        readme = out / "README.md"
        readme.write_text(
            build_readme(summary, train_gate, heldout_gate),
            encoding="utf-8",
        )

        return {
            "summary": summary,
            "train_rows": train_rows,
            "heldout_rows": heldout_rows,
            "train_gate": train_gate,
            "heldout_gate": heldout_gate,
        }


def build_why_not_enough_report(
    summary: Dict[str, Any],
    train_rows: List[Dict[str, Any]],
    heldout_rows: List[Dict[str, Any]],
    scanned: int,
) -> str:
    lines = [
        "# Why Not Enough Clean Groups",
        "",
        f"- target train: **{summary['target_train_groups']}**, found: **{len(train_rows)}**",
        f"- target heldout: **{summary['target_heldout_groups']}**, found: **{len(heldout_rows)}**",
        f"- scanned from BM25 pool: **{scanned}**",
        f"- replacement pool size: **{summary['replacement_pool_size']}**",
        "",
        "## Recommendation",
        "",
        "Do not start 200-step pilot until train/heldout targets are met.",
        "Consider increasing `--replacement-pool-size` or relaxing gate thresholds after review.",
    ]
    return "\n".join(lines) + "\n"


def build_readme(
    summary: Dict[str, Any],
    train_gate: Dict[str, Any],
    heldout_gate: Dict[str, Any],
) -> str:
    return (
        "# Phase 2.5b Expanded Clean Set\n\n"
        f"- train_clean_groups: **{summary['train_clean_groups']}**\n"
        f"- heldout_clean_groups: **{summary['heldout_clean_groups']}**\n"
        f"- expanded_clean_set_ready: **{summary['expanded_clean_set_ready']}**\n"
        f"- train_gate_passed: **{summary['train_gate_passed']}**\n"
        f"- heldout_gate_passed: **{summary['heldout_gate_passed']}**\n\n"
        "## Gate Summaries\n\n"
        f"Train spread rate: {train_gate.get('retrieval_quality_spread_group_rate')}\n\n"
        f"Heldout spread rate: {heldout_gate.get('retrieval_quality_spread_group_rate')}\n"
    )
