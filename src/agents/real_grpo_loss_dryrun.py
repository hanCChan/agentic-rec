"""
Phase 1.19: Real GRPO Loss Dry-Run with Large-K Quality Reward.

Uses Phase 1.18d strategy-controlled rollout groups and Phase 1.18f
reward_largek_mix_1000 to compute quality-only GRPO advantages and run
Phase 1.16 clipped policy loss dry-run. Does NOT train.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.agents.dataproto_mock import DataProtoMock
from src.agents.grpo_advantage_mock import GRPOAdvantageMock
from src.agents.grpo_loss_dryrun import GRPOLossDryRun, LOSS_DRYRUN_WARNING
from src.agents.real_dataproto_adapter import RealDataProtoAdapter
from src.agents.verl_batch_builder import VerlBatchBuilder, check_batch_shapes
from src.agents.verl_training_field_builder import VerlTrainingFieldBuilder, check_training_fields

REAL_GRPO_DRYRUN_WARNING = (
    "Real strategy-group GRPO loss dry-run only. "
    "No optimizer.step or trainer was invoked."
)

DEFAULT_CANDIDATE = "reward_largek_mix_1000"
EPS = 1e-6


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _spread(values: List[float]) -> float:
    if not values:
        return 0.0
    return max(values) - min(values)


def _std(values: List[float]) -> float:
    return pstdev(values) if len(values) > 1 else 0.0


def attach_group_metadata(
    fields: Dict[str, Any],
    records: List[Dict[str, Any]],
    group_size: int,
) -> None:
    fields["group_ids"] = [r["group_id"] for r in records]
    fields["group_indices"] = [r["group_index"] for r in records]
    fields["group_size"] = [group_size] * len(records)


class RealGRPOLossDryRun:
    """
    Phase 1.19 real GRPO loss dry-run with large-K quality-only rewards.

    Uses existing strategy-controlled rollout records and Phase 1.18f shaped
    rewards. Does NOT train or modify official environment reward.
    """

    def __init__(
        self,
        candidate_name: str = DEFAULT_CANDIDATE,
        normalize_by_std: bool = True,
        cliprange: float = 0.2,
        kl_coef: float = 0.01,
        loss_agg_mode: str = "token-mean",
        synthetic_logprob_delta: float = 0.02,
        seed: int = 42,
        eps: float = EPS,
    ):
        self.candidate_name = candidate_name
        self.normalize_by_std = normalize_by_std
        self.cliprange = cliprange
        self.kl_coef = kl_coef
        self.loss_agg_mode = loss_agg_mode
        self.synthetic_logprob_delta = synthetic_logprob_delta
        self.seed = seed
        self.eps = eps

    def load_inputs(
        self,
        rollout_path: str | Path,
        shaped_reward_path: str | Path,
        phase118f_summary_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        rollout_records = _load_jsonl(rollout_path)
        shaped_rows = _load_jsonl(shaped_reward_path)
        shaped_by_id = {row["sample_id"]: row for row in shaped_rows}

        phase118f_summary: Dict[str, Any] = {}
        if phase118f_summary_path and Path(phase118f_summary_path).exists():
            phase118f_summary = json.loads(Path(phase118f_summary_path).read_text(encoding="utf-8"))

        return {
            "rollout_records": rollout_records,
            "shaped_by_id": shaped_by_id,
            "phase118f_summary": phase118f_summary,
        }

    def merge_quality_rewards(
        self,
        rollout_records: List[Dict[str, Any]],
        shaped_by_id: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Replace sequence reward with quality-only large-K candidate reward."""
        merged: List[Dict[str, Any]] = []
        alignment_log: List[Dict[str, Any]] = []

        for record in rollout_records:
            sample_id = record.get("sample_id")
            shaped = shaped_by_id.get(sample_id)
            if shaped is None:
                raise KeyError(f"missing shaped reward for sample_id={sample_id}")

            if self.candidate_name not in shaped:
                raise KeyError(
                    f"candidate field `{self.candidate_name}` missing for sample_id={sample_id}"
                )

            quality_reward = float(shaped[self.candidate_name])
            member = dict(record)
            member["current_total_reward"] = float(record.get("reward", 0.0))
            member["penalty_component"] = float(shaped.get("penalty_component", 0.0))
            member["retrieval_quality_reward"] = quality_reward
            member["quality_reward_field"] = self.candidate_name
            member["reward"] = quality_reward

            metrics = dict(member.get("metrics", {}))
            metrics["total_reward"] = quality_reward
            metrics["quality_only_reward"] = quality_reward
            metrics["current_total_reward"] = member["current_total_reward"]
            metrics["penalty_component"] = member["penalty_component"]
            member["metrics"] = metrics

            extra = dict(member.get("extra_info", {}))
            extra["quality_only_reward"] = quality_reward
            extra["current_total_reward"] = member["current_total_reward"]
            extra["penalty_component"] = member["penalty_component"]
            member["extra_info"] = extra

            merged.append(member)
            alignment_log.append(
                {
                    "sample_id": sample_id,
                    "group_id": record.get("group_id"),
                    "strategy_name": record.get("strategy_name"),
                    "current_total_reward": member["current_total_reward"],
                    "penalty_component": member["penalty_component"],
                    "retrieval_quality_reward": quality_reward,
                    self.candidate_name: quality_reward,
                }
            )

        merged.sort(key=lambda r: (r.get("group_id", ""), r.get("group_index", 0)))
        return merged, alignment_log

    def compute_group_spread_stats(
        self,
        merged_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in merged_records:
            grouped.setdefault(row["group_id"], []).append(row)

        group_reports: List[Dict[str, Any]] = []
        for group_id in sorted(grouped.keys()):
            members = sorted(grouped[group_id], key=lambda r: r.get("group_index", 0))
            quality_rewards = [float(m["retrieval_quality_reward"]) for m in members]
            total_rewards = [float(m["current_total_reward"]) for m in members]
            penalties = [float(m["penalty_component"]) for m in members]

            q_std = _std(quality_rewards)
            q_spread = _spread(quality_rewards)
            p_spread = _spread(penalties)

            if q_spread > self.eps:
                spread_source = "retrieval_quality_spread"
            else:
                spread_source = "no_spread"

            group_reports.append(
                {
                    "group_id": group_id,
                    "group_size": len(members),
                    "quality_rewards": quality_rewards,
                    "current_total_rewards": total_rewards,
                    "penalty_values": penalties,
                    "quality_reward_std": q_std,
                    "quality_reward_spread": q_spread,
                    "penalty_spread": p_spread,
                    "zero_std_quality_reward": q_std <= self.eps,
                    "spread_source": spread_source,
                    "control_total_reward_spread": _spread(total_rewards),
                    "control_penalty_spread": p_spread,
                    "strategy_names": [m.get("strategy_name") for m in members],
                }
            )

        num_groups = len(group_reports)
        zero_std = sum(1 for r in group_reports if r["zero_std_quality_reward"])
        retrieval_spread = sum(
            1 for r in group_reports if r["spread_source"] == "retrieval_quality_spread"
        )

        return {
            "group_reports": group_reports,
            "num_groups": num_groups,
            "zero_std_group_rate": zero_std / num_groups if num_groups else 0.0,
            "retrieval_quality_spread_group_rate": retrieval_spread / num_groups
            if num_groups
            else 0.0,
            "penalty_only_spread_group_rate": 0.0,
        }

    def build_dataproto(
        self,
        merged_records: List[Dict[str, Any]],
        tokenizer_path: str,
        group_size: int,
        max_prompt_length: int,
        max_response_length: int,
        max_total_length: int,
    ) -> Tuple[Any, bool]:
        batch_builder = VerlBatchBuilder(
            tokenizer_path=tokenizer_path,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            max_total_length=max_total_length,
        )
        batch = batch_builder.build_batch(merged_records)
        check_batch_shapes(batch)

        pad_token_id = batch_builder.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = batch_builder.tokenizer.eos_token_id

        field_builder = VerlTrainingFieldBuilder(pad_token_id=pad_token_id)
        fields = field_builder.build_training_fields(batch)
        check_training_fields(fields)
        attach_group_metadata(fields, merged_records, group_size)

        mock_proto = DataProtoMock.from_fields(fields)
        mock_validate = self._validate_dataproto(fields, mock_proto)
        if not mock_validate["passed"]:
            raise ValueError(f"DataProtoMock validate failed: {mock_validate['errors']}")

        adapter = RealDataProtoAdapter()
        convert_result = adapter.to_real_dataproto(mock_proto)
        data_proto = convert_result["real_proto"] if convert_result["used_real_dataproto"] else mock_proto
        return data_proto, convert_result["used_real_dataproto"]

    def _validate_dataproto(self, fields: Dict[str, Any], mock_proto: DataProtoMock) -> Dict[str, Any]:
        """Validate DataProto; allow zero quality rewards on valid response rows."""
        checks = mock_proto.validate()
        if checks["passed"]:
            return checks

        allowed_zero_reward_errors = [
            e
            for e in checks["errors"]
            if "token_level_rewards nonzero rows" in e
        ]
        other_errors = [e for e in checks["errors"] if e not in allowed_zero_reward_errors]
        if other_errors:
            return {"passed": False, "errors": other_errors}

        token_level_rewards = fields["token_level_rewards"]
        response_attention_mask = fields["response_attention_mask"]
        for i in range(token_level_rewards.shape[0]):
            rlen = int(response_attention_mask[i].sum().item())
            if rlen <= 0:
                return {"passed": False, "errors": ["response_attention_mask empty row"]}
            last_idx = rlen - 1
            if token_level_rewards[i, last_idx].item() != fields["sequence_rewards"][i].item():
                return {
                    "passed": False,
                    "errors": ["token_level_rewards last token != sequence_rewards"],
                }

        return {"passed": True, "errors": [], "allowed_zero_quality_rewards": True}

    def run_advantage_and_loss(
        self,
        data_proto: Any,
        group_size: int,
    ) -> Dict[str, Any]:
        grpo_mock = GRPOAdvantageMock(
            group_size=group_size,
            normalize_by_std=self.normalize_by_std,
            synthetic_reward_jitter=0.0,
            seed=self.seed,
            eps=self.eps,
        )
        adv_output = grpo_mock.compute_group_advantages(data_proto)
        adv_check = grpo_mock.check_group_advantages(data_proto, adv_output)

        loss_dryrun = GRPOLossDryRun(
            cliprange=self.cliprange,
            kl_coef=self.kl_coef,
            loss_agg_mode=self.loss_agg_mode,
            synthetic_logprob_delta=self.synthetic_logprob_delta,
            seed=self.seed,
        )
        loss_inputs = loss_dryrun.build_mock_logprob_inputs(data_proto, adv_output)
        loss_output = loss_dryrun.compute_policy_loss(data_proto, loss_inputs)
        loss_check = loss_dryrun.check_loss_output(data_proto, loss_inputs, loss_output)

        return {
            "adv_output": adv_output,
            "adv_check": adv_check,
            "loss_inputs": loss_inputs,
            "loss_output": loss_output,
            "loss_check": loss_check,
        }

    def run(
        self,
        rollout_path: str | Path,
        shaped_reward_path: str | Path,
        tokenizer_path: str,
        phase118f_summary_path: str | Path | None = None,
        max_prompt_length: int = 1024,
        max_response_length: int = 2048,
        max_total_length: int = 3072,
    ) -> Dict[str, Any]:
        inputs = self.load_inputs(rollout_path, shaped_reward_path, phase118f_summary_path)
        rollout_records = inputs["rollout_records"]
        if not rollout_records:
            raise ValueError("no rollout records loaded")

        group_size = int(rollout_records[0].get("group_size", 4))
        merged_records, alignment_log = self.merge_quality_rewards(
            rollout_records,
            inputs["shaped_by_id"],
        )
        spread_stats = self.compute_group_spread_stats(merged_records)

        data_proto, used_real_dataproto = self.build_dataproto(
            merged_records,
            tokenizer_path=tokenizer_path,
            group_size=group_size,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            max_total_length=max_total_length,
        )

        run_output = self.run_advantage_and_loss(data_proto, group_size=group_size)
        adv_check = run_output["adv_check"]
        loss_check = run_output["loss_check"]
        loss_output = run_output["loss_output"]

        num_groups = spread_stats["num_groups"]
        summary = {
            "phase": "1.19",
            "num_groups": num_groups,
            "group_size": group_size,
            "num_rollout_records": len(merged_records),
            "reward_candidate": self.candidate_name,
            "quality_only_advantage": True,
            "penalties_in_advantage": False,
            "zero_std_group_rate": spread_stats["zero_std_group_rate"],
            "retrieval_quality_spread_group_rate": spread_stats[
                "retrieval_quality_spread_group_rate"
            ],
            "penalty_only_spread_group_rate": spread_stats["penalty_only_spread_group_rate"],
            "advantage_zero_std_group_rate": adv_check["zero_std_group_rate"],
            "mean_abs_sequence_advantage": adv_check["mean_abs_sequence_advantage"],
            "advantage_check_passed": adv_check["advantage_check_passed"],
            "loss_check_passed": loss_check["loss_check_passed"],
            "used_real_dataproto": used_real_dataproto,
            "policy_loss_finite": loss_check["policy_loss_finite"],
            "policy_loss_value": loss_output["policy_loss_value"],
            "clipfrac": loss_output["clipfrac"],
            "mean_valid_ratio": loss_output["mean_valid_ratio"],
            "mean_valid_kl": loss_output["mean_valid_kl"],
            "padding_loss_zero": loss_check["padding_loss_zero"],
            "padding_ratio_zero": loss_check["padding_ratio_zero"],
            "padding_kl_zero": loss_check["padding_kl_zero"],
            "loss_agg_mode": self.loss_agg_mode,
            "cliprange": self.cliprange,
            "kl_coef": self.kl_coef,
            "synthetic_logprob_delta": self.synthetic_logprob_delta,
            "is_training": False,
            "dryrun_warning": REAL_GRPO_DRYRUN_WARNING,
            "loss_dryrun_warning": LOSS_DRYRUN_WARNING,
        }

        if inputs["phase118f_summary"]:
            summary["phase118f_gate_passed"] = inputs["phase118f_summary"].get("gate_passed")
            summary["phase118f_recommended_candidate"] = inputs["phase118f_summary"].get(
                "recommended_candidate"
            )

        return {
            "summary": summary,
            "merged_records": merged_records,
            "alignment_log": alignment_log,
            "spread_stats": spread_stats,
            "adv_check": adv_check,
            "loss_check": loss_check,
            "loss_output": loss_output,
            "loss_shapes": {
                "policy_loss_mat": loss_check["policy_loss_mat_shape"],
                "ratio": loss_check["ratio_shape"],
                "token_kl": loss_check["token_kl_shape"],
                "kl_penalty": loss_check["kl_penalty_shape"],
            },
            "loss_stats": {
                "policy_loss_value": loss_output["policy_loss_value"],
                "clipfrac": loss_output["clipfrac"],
                "mean_valid_ratio": loss_output["mean_valid_ratio"],
                "mean_valid_kl": loss_output["mean_valid_kl"],
                "zero_std_group_rate": spread_stats["zero_std_group_rate"],
                "mean_abs_sequence_advantage": adv_check["mean_abs_sequence_advantage"],
            },
        }


def build_real_grpo_dryrun_report(result: Dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# Phase 1.19 Real GRPO Loss Dry-Run Report",
        "",
        "## Configuration",
        "",
        f"- reward_candidate: `{summary['reward_candidate']}`",
        f"- quality_only_advantage: **{summary['quality_only_advantage']}**",
        f"- penalties_in_advantage: **{summary['penalties_in_advantage']}**",
        "",
        "## Quality Spread (Large-K Reward)",
        "",
        f"- zero_std_group_rate: **{summary['zero_std_group_rate']:.2f}**",
        f"- retrieval_quality_spread_group_rate: **{summary['retrieval_quality_spread_group_rate']:.2f}**",
        f"- penalty_only_spread_group_rate: **{summary['penalty_only_spread_group_rate']:.2f}**",
        "",
        "## Checks",
        "",
        f"- advantage_check_passed: **{summary['advantage_check_passed']}**",
        f"- loss_check_passed: **{summary['loss_check_passed']}**",
        f"- used_real_dataproto: **{summary['used_real_dataproto']}**",
        "",
        "## Loss Dry-Run",
        "",
        f"- policy_loss_value: **{summary['policy_loss_value']:.6f}**",
        f"- clipfrac: **{summary['clipfrac']:.4f}**",
        f"- mean_valid_ratio: **{summary['mean_valid_ratio']:.4f}**",
        f"- mean_valid_kl: **{summary['mean_valid_kl']:.6f}**",
        f"- padding_loss_zero: **{summary['padding_loss_zero']}**",
        "",
        summary["dryrun_warning"],
        "",
        "## Per-Group Quality Rewards",
        "",
    ]
    for row in result["spread_stats"]["group_reports"]:
        lines.append(
            f"- `{row['group_id']}`: quality={row['quality_rewards']}, "
            f"spread={row['quality_reward_spread']:.4f}, source=`{row['spread_source']}`"
        )
    lines.extend(
        [
            "",
            "## Next Steps",
            "",
            "Phase 1.19 validates the loss path only. Before training:",
            "1. Phase 1.19b — scale gate check on 10_g4 / 20_g4",
            "2. Phase 1.18g — replace BM25 failure samples",
            "3. Phase 1.18h — fix strategy query collapse",
            "4. Phase 1.20 — no-update trainer dry-run",
        ]
    )
    return "\n".join(lines) + "\n"
