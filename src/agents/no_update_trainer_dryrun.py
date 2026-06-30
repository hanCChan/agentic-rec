"""
Phase 1.20: No-update VERL Trainer Dry-Run.

Validates strategy-controlled rollout + large-K quality reward data can enter
VERL trainer-facing DataProto and pass GRPO advantage / mini-batch / loss checks.

Does NOT call trainer.fit(), update_actor(), or optimizer.step().
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.agents.dataproto_mock import DataProtoMock, DRY_RUN_WARNING
from src.agents.grpo_advantage_mock import GRPOAdvantageMock, _get_batch, _get_non_tensor_list
from src.agents.grpo_loss_dryrun import GRPOLossDryRun, LOSS_DRYRUN_WARNING
from src.agents.real_dataproto_adapter import RealDataProtoAdapter
from src.agents.real_grpo_loss_dryrun import DEFAULT_CANDIDATE, RealGRPOLossDryRun
from src.agents.verl_batch_builder import VerlBatchBuilder, check_batch_shapes
from src.agents.verl_training_field_builder import VerlTrainingFieldBuilder, check_training_fields

NO_UPDATE_TRAINER_WARNING = (
    "Phase 1.20 no-update VERL trainer dry-run only. "
    "No trainer.fit(), update_actor(), or optimizer.step()."
)

TRAINER_BATCH_REQUIRED_KEYS = (
    "input_ids",
    "attention_mask",
    "position_ids",
    "prompts",
    "responses",
    "response_mask",
    "token_level_rewards",
    "old_log_probs",
    "ref_log_probs",
)

TRAINER_BATCH_POST_ADV_KEYS = ("advantages", "returns")

TRAINER_NON_TENSOR_REQUIRED_KEYS = (
    "uid",
    "sample_id",
    "group_id",
    "group_index",
    "strategy_name",
    "reward_candidate",
)

EPS = 1e-6


def _ensure_verl_path() -> None:
    root = Path(__file__).resolve().parents[2]
    rec_r1 = str(root / "Rec-R1")
    if rec_r1 not in sys.path:
        sys.path.insert(0, rec_r1)


class NoUpdateGuard:
    """Track forbidden trainer/update calls during Phase 1.20 dry-run."""

    def __init__(self) -> None:
        self.optimizer_step_called = False
        self.update_actor_called = False
        self.trainer_fit_called = False

    def mark_optimizer_step(self) -> None:
        self.optimizer_step_called = True

    def mark_update_actor(self) -> None:
        self.update_actor_called = True

    def mark_trainer_fit(self) -> None:
        self.trainer_fit_called = True

    def assert_no_update(self) -> None:
        assert not self.optimizer_step_called, "optimizer.step was called"
        assert not self.update_actor_called, "update_actor was called"
        assert not self.trainer_fit_called, "trainer.fit was called"

    def to_dict(self) -> Dict[str, Any]:
        passed = not any(
            [
                self.optimizer_step_called,
                self.update_actor_called,
                self.trainer_fit_called,
            ]
        )
        return {
            "trainer_fit_called": self.trainer_fit_called,
            "update_actor_called": self.update_actor_called,
            "optimizer_step_called": self.optimizer_step_called,
            "no_update_guard_passed": passed,
            "optimizer_step_guard_enabled": True,
        }


@contextmanager
def optimizer_step_guard(guard: NoUpdateGuard):
    """Monkeypatch Optimizer.step to fail fast if invoked."""
    original_step = torch.optim.Optimizer.step

    def guarded_step(self, closure=None):  # noqa: ANN001
        guard.mark_optimizer_step()
        raise RuntimeError("optimizer.step is forbidden in Phase 1.20 no-update dry-run")

    torch.optim.Optimizer.step = guarded_step
    try:
        yield
    finally:
        torch.optim.Optimizer.step = original_step


def attach_trainer_metadata(
    fields: Dict[str, Any],
    records: List[Dict[str, Any]],
    group_size: int,
    candidate_name: str,
) -> None:
    """Attach trainer-facing logprob placeholders and non-tensor metadata."""
    fields["old_log_probs"] = fields["mock_old_log_probs"].clone()
    fields["ref_log_probs"] = fields["mock_old_log_probs"].clone()
    fields["uid"] = [r["group_id"] for r in records]
    fields["sample_id"] = [r["sample_id"] for r in records]
    fields["group_id"] = [r["group_id"] for r in records]
    fields["group_index"] = [r["group_index"] for r in records]
    fields["strategy_name"] = [r.get("strategy_name", "") for r in records]
    fields["reward_candidate"] = [candidate_name] * len(records)
    fields["group_ids"] = list(fields["uid"])
    fields["group_indices"] = list(fields["group_index"])
    fields["group_size_list"] = [group_size] * len(records)
    fields["total_reward_diagnostic"] = [
        float(r.get("current_total_reward", r.get("reward", 0.0))) for r in records
    ]


def build_trainer_dataproto_mock(fields: Dict[str, Any]) -> DataProtoMock:
    """Build trainer-facing DataProtoMock with extended batch/non_tensor keys."""
    batch = {
        "input_ids": fields["input_ids"],
        "attention_mask": fields["attention_mask"],
        "position_ids": fields["position_ids"],
        "prompts": fields["prompts"],
        "responses": fields["responses"],
        "response_attention_mask": fields["response_attention_mask"],
        "response_mask": fields["response_mask"],
        "token_level_rewards": fields["token_level_rewards"],
        "sequence_rewards": fields["sequence_rewards"],
        "old_log_probs": fields["old_log_probs"],
        "ref_log_probs": fields["ref_log_probs"],
    }
    non_tensor_batch = {
        "uid": fields["uid"],
        "sample_id": fields["sample_id"],
        "group_id": fields["group_id"],
        "group_index": fields["group_index"],
        "strategy_name": fields["strategy_name"],
        "reward_candidate": fields["reward_candidate"],
        "sample_ids": fields["sample_ids"],
        "metrics": fields["metrics"],
        "extra_info": fields["extra_info"],
        "group_ids": fields["group_ids"],
        "group_indices": fields["group_indices"],
        "group_size": fields["group_size_list"],
        "total_reward_diagnostic": fields["total_reward_diagnostic"],
    }
    meta_info = {
        "phase": "1.20",
        "source": "agentic-rec no-update trainer dry-run",
        "prompt_lengths": fields["prompt_lengths"].tolist(),
        "response_lengths": fields["response_lengths"].tolist(),
        "dry_run_warning": DRY_RUN_WARNING,
        "no_update_warning": NO_UPDATE_TRAINER_WARNING,
    }
    return DataProtoMock(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)


class NoUpdateTrainerDryRun:
    """
    Phase 1.20 no-update VERL trainer dry-run.

    Validates trainer-facing DataProto + VERL GRPO advantage + mini-batch split
    + no-update loss path. Does NOT train.
    """

    def __init__(
        self,
        tokenizer_path: str,
        candidate_name: str = DEFAULT_CANDIDATE,
        train_batch_size: int = 20,
        rollout_n: int = 4,
        ppo_mini_batch_size: int = 20,
        micro_batch_size: int = 4,
        cliprange: float = 0.2,
        kl_coef: float = 0.01,
        loss_agg_mode: str = "token-mean",
        synthetic_logprob_delta: float = 0.02,
        seed: int = 42,
        eps: float = EPS,
    ):
        self.tokenizer_path = tokenizer_path
        self.candidate_name = candidate_name
        self.train_batch_size = train_batch_size
        self.rollout_n = rollout_n
        self.ppo_mini_batch_size = ppo_mini_batch_size
        self.micro_batch_size = micro_batch_size
        self.cliprange = cliprange
        self.kl_coef = kl_coef
        self.loss_agg_mode = loss_agg_mode
        self.synthetic_logprob_delta = synthetic_logprob_delta
        self.seed = seed
        self.eps = eps
        self.guard = NoUpdateGuard()
        self._real_grpo = RealGRPOLossDryRun(
            candidate_name=candidate_name,
            cliprange=cliprange,
            kl_coef=kl_coef,
            loss_agg_mode=loss_agg_mode,
            synthetic_logprob_delta=synthetic_logprob_delta,
            seed=seed,
            eps=eps,
        )

    def load_inputs(self, rollout_path: str, shaped_reward_path: str) -> dict:
        return self._real_grpo.load_inputs(rollout_path, shaped_reward_path)

    def build_trainer_facing_dataproto(
        self,
        inputs: dict,
        *,
        max_prompt_length: int = 1024,
        max_response_length: int = 2048,
        max_total_length: int = 3072,
    ) -> Tuple[Any, bool, List[Dict[str, Any]]]:
        rollout_records = inputs["rollout_records"]
        merged_records, _ = self._real_grpo.merge_quality_rewards(
            rollout_records,
            inputs["shaped_by_id"],
        )
        merged_records.sort(key=lambda r: (r.get("group_id", ""), r.get("group_index", 0)))

        expected = self.train_batch_size * self.rollout_n
        if len(merged_records) != expected:
            raise ValueError(
                f"expected {expected} records (train_batch_size={self.train_batch_size} "
                f"* rollout_n={self.rollout_n}), got {len(merged_records)}"
            )

        group_size = int(merged_records[0].get("group_size", self.rollout_n))

        batch_builder = VerlBatchBuilder(
            tokenizer_path=self.tokenizer_path,
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
        attach_trainer_metadata(fields, merged_records, group_size, self.candidate_name)

        mock_proto = build_trainer_dataproto_mock(fields)
        mock_validate = self._real_grpo._validate_dataproto(fields, mock_proto)
        if not mock_validate["passed"]:
            raise ValueError(f"DataProto validate failed: {mock_validate['errors']}")

        adapter = RealDataProtoAdapter()
        convert_result = adapter.to_real_dataproto(mock_proto)
        data_proto = convert_result["real_proto"] if convert_result["used_real_dataproto"] else mock_proto
        return data_proto, convert_result["used_real_dataproto"], merged_records

    def check_trainer_required_keys(self, data: Any) -> dict:
        batch = _get_batch(data)
        missing_batch = [k for k in TRAINER_BATCH_REQUIRED_KEYS if k not in batch]
        missing_non_tensor = [
            k for k in TRAINER_NON_TENSOR_REQUIRED_KEYS if k not in data.non_tensor_batch
        ]

        uid_values = _get_non_tensor_list(data, "uid")
        group_id_values = _get_non_tensor_list(data, "group_id")
        uid_matches_group = uid_values == group_id_values

        batch_size = batch["input_ids"].shape[0]
        uid_len_ok = len(uid_values) == batch_size

        unique_uids = sorted(set(uid_values))
        uid_group_sizes = {uid: uid_values.count(uid) for uid in unique_uids}
        all_groups_full = all(size == self.rollout_n for size in uid_group_sizes.values())
        num_groups_ok = len(unique_uids) == self.train_batch_size

        passed = (
            not missing_batch
            and not missing_non_tensor
            and uid_len_ok
            and uid_matches_group
            and all_groups_full
            and num_groups_ok
        )

        return {
            "trainer_required_keys_passed": passed,
            "missing_batch_keys": missing_batch,
            "missing_non_tensor_keys": missing_non_tensor,
            "batch_size": batch_size,
            "num_unique_uids": len(unique_uids),
            "uid_equals_group_id": uid_matches_group,
            "all_groups_size_equal_rollout_n": all_groups_full,
            "uid_group_sizes_sample": dict(list(uid_group_sizes.items())[:5]),
            "required_batch_keys_present": [k for k in TRAINER_BATCH_REQUIRED_KEYS if k in batch],
            "required_non_tensor_keys_present": [
                k for k in TRAINER_NON_TENSOR_REQUIRED_KEYS if k in data.non_tensor_batch
            ],
        }

    def compute_verl_grpo_advantage_no_update(self, data: Any) -> Tuple[Any, dict]:
        _ensure_verl_path()
        meta: Dict[str, Any] = {
            "used_verl_compute_advantage": False,
            "fallback_to_project_advantage": False,
            "fallback_reason": None,
            "norm_adv_by_std_in_grpo": True,
            "num_repeat": self.rollout_n,
        }

        try:
            from verl.trainer.ppo.ray_trainer import compute_advantage

            compute_advantage(
                data,
                adv_estimator="grpo",
                gamma=1.0,
                lam=1.0,
                num_repeat=self.rollout_n,
            )
            meta["used_verl_compute_advantage"] = True
            return data, meta
        except Exception as exc:
            meta["fallback_to_project_advantage"] = True
            meta["fallback_reason"] = f"{type(exc).__name__}: {exc}"

            grpo_mock = GRPOAdvantageMock(
                group_size=self.rollout_n,
                normalize_by_std=True,
                synthetic_reward_jitter=0.0,
                seed=self.seed,
                eps=self.eps,
            )
            adv_output = grpo_mock.compute_group_advantages(data)
            batch = _get_batch(data)
            batch["advantages"] = adv_output["token_level_advantages"]
            batch["returns"] = adv_output["token_level_advantages"].clone()
            return data, meta

    def _check_verl_advantage(self, data: Any) -> dict:
        batch = _get_batch(data)
        missing = [k for k in TRAINER_BATCH_POST_ADV_KEYS if k not in batch]
        if missing:
            return {
                "advantage_check_passed": False,
                "missing_keys": missing,
            }

        advantages = batch["advantages"]
        returns = batch["returns"]
        response_mask = batch["response_attention_mask"]
        uids = _get_non_tensor_list(data, "uid")

        assert advantages.shape == returns.shape
        assert advantages.shape == response_mask.shape
        assert torch.isfinite(advantages[response_mask.bool()]).all()

        unique_uids = sorted(set(uids))
        num_groups = len(unique_uids)
        sequence_advantages: List[float] = []
        zero_std_groups = 0

        for uid in unique_uids:
            idx = [i for i, u in enumerate(uids) if u == uid]
            group_adv = advantages[idx, 0].tolist()
            sequence_advantages.extend(group_adv)
            if len(group_adv) > 1 and pstdev(group_adv) <= self.eps:
                zero_std_groups += 1

        mean_abs = float(torch.abs(advantages[response_mask.bool()]).mean().item())
        zero_std_rate = zero_std_groups / num_groups if num_groups else 0.0

        return {
            "advantage_check_passed": True,
            "advantages_shape": list(advantages.shape),
            "returns_shape": list(returns.shape),
            "mean_abs_sequence_advantage": mean_abs,
            "zero_std_group_rate": zero_std_rate,
            "num_groups": num_groups,
            "padding_advantages_zero": bool((advantages[~response_mask.bool()] == 0).all().item()),
        }

    def check_minibatch_splits(self, data: Any) -> dict:
        batch = _get_batch(data)
        num_records = batch["input_ids"].shape[0]
        uids = _get_non_tensor_list(data, "uid")

        divisible_records = num_records % self.ppo_mini_batch_size == 0
        divisible_micro = self.ppo_mini_batch_size % self.micro_batch_size == 0
        num_ppo_minibatches = num_records // self.ppo_mini_batch_size if divisible_records else 0
        num_microbatches = (
            self.ppo_mini_batch_size // self.micro_batch_size if divisible_micro else 0
        )

        minibatch_reports: List[Dict[str, Any]] = []
        all_minibatches_ok = True

        for mb_idx in range(num_ppo_minibatches):
            start = mb_idx * self.ppo_mini_batch_size
            end = start + self.ppo_mini_batch_size
            mb_uids = uids[start:end]
            unique_mb_uids = sorted(set(mb_uids))
            uid_counts = {u: mb_uids.count(u) for u in unique_mb_uids}
            groups_full = all(c == self.rollout_n for c in uid_counts.values())
            covers_multiple = len(unique_mb_uids) >= 2

            mb_ok = groups_full and covers_multiple
            if not mb_ok:
                all_minibatches_ok = False

            micro_reports = []
            for micro_idx in range(num_microbatches):
                mstart = start + micro_idx * self.micro_batch_size
                mend = mstart + self.micro_batch_size
                micro_uids = uids[mstart:mend]
                micro_reports.append(
                    {
                        "micro_batch_index": micro_idx,
                        "unique_uids": len(set(micro_uids)),
                        "records": self.micro_batch_size,
                    }
                )

            minibatch_reports.append(
                {
                    "minibatch_index": mb_idx,
                    "record_range": [start, end],
                    "unique_uids": len(unique_mb_uids),
                    "uid_group_sizes": uid_counts,
                    "groups_full_size": groups_full,
                    "covers_multiple_uids": covers_multiple,
                    "minibatch_ok": mb_ok,
                    "micro_batches": micro_reports,
                }
            )

        passed = (
            divisible_records
            and divisible_micro
            and all_minibatches_ok
            and num_records == self.train_batch_size * self.rollout_n
        )

        return {
            "num_records": num_records,
            "ppo_mini_batch_size": self.ppo_mini_batch_size,
            "micro_batch_size": self.micro_batch_size,
            "num_ppo_minibatches": num_ppo_minibatches,
            "num_microbatches_per_minibatch": num_microbatches,
            "minibatch_check_passed": passed,
            "minibatch_reports": minibatch_reports,
        }

    def check_loss_inputs(self, data: Any) -> dict:
        batch = _get_batch(data)
        required = ["advantages", "returns", "old_log_probs", "ref_log_probs", "token_level_rewards"]
        missing = [k for k in required if k not in batch]
        mask = batch["response_attention_mask"]
        passed = not missing and batch["advantages"].shape == mask.shape
        return {
            "loss_input_check_passed": passed,
            "missing_keys": missing,
            "advantages_shape": list(batch["advantages"].shape) if "advantages" in batch else None,
            "old_log_probs_shape": list(batch["old_log_probs"].shape) if "old_log_probs" in batch else None,
        }

    def run_no_update_loss_dryrun(self, data: Any) -> dict:
        batch = _get_batch(data)
        response_mask = batch["response_attention_mask"].float()
        advantages = batch["advantages"].float() * response_mask
        old_log_probs = batch["old_log_probs"].float() * response_mask
        log_probs = (old_log_probs + self.synthetic_logprob_delta) * response_mask
        ref_log_probs = batch["ref_log_probs"].float() * response_mask

        loss_inputs = {
            "log_probs": log_probs,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs,
            "advantages": advantages,
            "response_mask": response_mask,
            "is_mock": True,
        }

        loss_dryrun = GRPOLossDryRun(
            cliprange=self.cliprange,
            kl_coef=self.kl_coef,
            loss_agg_mode=self.loss_agg_mode,
            synthetic_logprob_delta=self.synthetic_logprob_delta,
            seed=self.seed,
        )
        loss_output = loss_dryrun.compute_policy_loss(data, loss_inputs)
        loss_check = loss_dryrun.check_loss_output(data, loss_inputs, loss_output)
        return {
            "loss_inputs": {k: v for k, v in loss_inputs.items() if k != "log_probs"},
            "loss_output": {
                k: v
                for k, v in loss_output.items()
                if k
                not in {
                    "policy_loss",
                    "policy_loss_mat",
                    "ratio",
                    "clipped_ratio",
                    "log_ratio",
                    "token_kl",
                    "kl_penalty",
                    "kl_adjusted_token_rewards",
                }
            },
            "loss_check": loss_check,
        }

    def check_no_parameter_update(self) -> dict:
        self.guard.assert_no_update()
        return self.guard.to_dict()

    def _spread_stats_from_records(self, merged_records: List[Dict[str, Any]]) -> dict:
        return self._real_grpo.compute_group_spread_stats(merged_records)

    def _dataproto_shapes(self, data: Any) -> dict:
        batch = _get_batch(data)
        shapes = {k: list(v.shape) for k, v in batch.items()}
        non_tensor = {
            k: len(v) if hasattr(v, "__len__") else 1 for k, v in data.non_tensor_batch.items()
        }
        return {"batch_shapes": shapes, "non_tensor_lengths": non_tensor}

    def run(
        self,
        rollout_path: str,
        shaped_reward_path: str,
        output_dir: str,
        max_prompt_length: int = 1024,
        max_response_length: int = 2048,
        max_total_length: int = 3072,
    ) -> dict:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        with optimizer_step_guard(self.guard):
            inputs = self.load_inputs(rollout_path, shaped_reward_path)
            data_proto, used_real_dataproto, merged_records = self.build_trainer_facing_dataproto(
                inputs,
                max_prompt_length=max_prompt_length,
                max_response_length=max_response_length,
                max_total_length=max_total_length,
            )

            key_check = self.check_trainer_required_keys(data_proto)
            data_proto, adv_meta = self.compute_verl_grpo_advantage_no_update(data_proto)
            adv_check = self._check_verl_advantage(data_proto)
            adv_check.update(adv_meta)

            minibatch_check = self.check_minibatch_splits(data_proto)
            loss_input_check = self.check_loss_inputs(data_proto)
            loss_result = self.run_no_update_loss_dryrun(data_proto)
            guard_result = self.check_no_parameter_update()

            spread_stats = self._spread_stats_from_records(merged_records)
            shapes = self._dataproto_shapes(data_proto)

            loss_check = loss_result["loss_check"]
            loss_output = loss_result["loss_output"]

            summary = {
                "phase": "1.20",
                "mode": "no_update_trainer_dryrun",
                "num_groups": self.train_batch_size,
                "group_size": self.rollout_n,
                "num_rollout_records": len(merged_records),
                "reward_candidate": self.candidate_name,
                "quality_only_advantage": True,
                "penalties_in_advantage": False,
                "used_real_dataproto": used_real_dataproto,
                "used_verl_compute_advantage": adv_meta["used_verl_compute_advantage"],
                "fallback_to_project_advantage": adv_meta["fallback_to_project_advantage"],
                "trainer_required_keys_passed": key_check["trainer_required_keys_passed"],
                "advantage_check_passed": adv_check.get("advantage_check_passed", False),
                "minibatch_check_passed": minibatch_check["minibatch_check_passed"],
                "loss_check_passed": loss_check["loss_check_passed"],
                "zero_std_group_rate": spread_stats["zero_std_group_rate"],
                "retrieval_quality_spread_group_rate": spread_stats[
                    "retrieval_quality_spread_group_rate"
                ],
                "penalty_only_spread_group_rate": spread_stats["penalty_only_spread_group_rate"],
                "mean_abs_sequence_advantage": adv_check.get("mean_abs_sequence_advantage", 0.0),
                "policy_loss_finite": loss_check["policy_loss_finite"],
                "clipfrac": loss_output["clipfrac"],
                "mean_valid_ratio": loss_output["mean_valid_ratio"],
                "mean_valid_kl": loss_output["mean_valid_kl"],
                "padding_loss_zero": loss_check["padding_loss_zero"],
                "padding_ratio_zero": loss_check["padding_ratio_zero"],
                "padding_kl_zero": loss_check["padding_kl_zero"],
                "trainer_fit_called": guard_result["trainer_fit_called"],
                "update_actor_called": guard_result["update_actor_called"],
                "optimizer_step_called": guard_result["optimizer_step_called"],
                "no_update_guard_passed": guard_result["no_update_guard_passed"],
                "optimizer_step_guard_enabled": guard_result["optimizer_step_guard_enabled"],
                "is_training": False,
                "safe_for_phase2": False,
                "next_recommendation": (
                    "Proceed to Phase 1.18g/1.18h cleanup and then Phase 2 smoke training "
                    "only after residual collapse cases are handled."
                ),
                "dryrun_warning": NO_UPDATE_TRAINER_WARNING,
                "loss_dryrun_warning": LOSS_DRYRUN_WARNING,
            }
            if adv_meta.get("fallback_reason"):
                summary["fallback_reason"] = adv_meta["fallback_reason"]

            (out / "trainer_facing_dataproto_shapes.json").write_text(
                json.dumps(shapes, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (out / "trainer_required_key_check.json").write_text(
                json.dumps(key_check, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (out / "verl_advantage_check.json").write_text(
                json.dumps(adv_check, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (out / "minibatch_split_check.json").write_text(
                json.dumps(minibatch_check, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (out / "no_update_loss_stats.json").write_text(
                json.dumps(
                    {"loss_output": loss_output, "loss_check": loss_check, "loss_input_check": loss_input_check},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (out / "no_update_guard.json").write_text(
                json.dumps(guard_result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (out / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (out / "no_update_trainer_dryrun_report.md").write_text(
                build_no_update_trainer_report(summary, key_check, adv_check, minibatch_check, loss_check),
                encoding="utf-8",
            )
            readme = out / "README.md"
            if not readme.exists():
                readme.write_text(
                    "# Phase 1.20 No-update VERL Trainer Dry-Run\n\n"
                    "Validates trainer-facing DataProto + VERL GRPO advantage + mini-batch "
                    "split + no-update loss path. No training.\n",
                    encoding="utf-8",
                )

        return {
            "summary": summary,
            "key_check": key_check,
            "adv_check": adv_check,
            "minibatch_check": minibatch_check,
            "loss_check": loss_check,
            "loss_output": loss_output,
            "guard_result": guard_result,
            "spread_stats": spread_stats,
        }


def build_no_update_trainer_report(
    summary: dict,
    key_check: dict,
    adv_check: dict,
    minibatch_check: dict,
    loss_check: dict,
) -> str:
    lines = [
        "# Phase 1.20 No-update VERL Trainer Dry-Run Report",
        "",
        "## Mode",
        "",
        f"- mode: `{summary['mode']}`",
        f"- is_training: **{summary['is_training']}**",
        f"- reward_candidate: `{summary['reward_candidate']}`",
        "",
        "## DataProto / Trainer Path",
        "",
        f"- used_real_dataproto: **{summary['used_real_dataproto']}**",
        f"- used_verl_compute_advantage: **{summary['used_verl_compute_advantage']}**",
        f"- fallback_to_project_advantage: **{summary['fallback_to_project_advantage']}**",
        f"- trainer_required_keys_passed: **{summary['trainer_required_keys_passed']}**",
        "",
        "## Checks",
        "",
        f"- advantage_check_passed: **{summary['advantage_check_passed']}**",
        f"- minibatch_check_passed: **{summary['minibatch_check_passed']}**",
        f"- loss_check_passed: **{summary['loss_check_passed']}**",
        f"- no_update_guard_passed: **{summary['no_update_guard_passed']}**",
        "",
        "## No-Update Guard",
        "",
        f"- trainer_fit_called: **{summary['trainer_fit_called']}**",
        f"- update_actor_called: **{summary['update_actor_called']}**",
        f"- optimizer_step_called: **{summary['optimizer_step_called']}**",
        "",
        "## Quality Spread",
        "",
        f"- zero_std_group_rate: **{summary['zero_std_group_rate']:.2f}**",
        f"- retrieval_quality_spread_group_rate: **{summary['retrieval_quality_spread_group_rate']:.2f}**",
        "",
        "## Loss Dry-Run",
        "",
        f"- policy_loss_finite: **{summary['policy_loss_finite']}**",
        f"- clipfrac: **{summary['clipfrac']:.4f}**",
        f"- mean_valid_ratio: **{summary['mean_valid_ratio']:.4f}**",
        f"- mean_valid_kl: **{summary['mean_valid_kl']:.6f}**",
        "",
        "## Mini-batch",
        "",
        f"- num_records: **{minibatch_check['num_records']}**",
        f"- num_ppo_minibatches: **{minibatch_check['num_ppo_minibatches']}**",
        f"- num_microbatches_per_minibatch: **{minibatch_check['num_microbatches_per_minibatch']}**",
        "",
        summary["dryrun_warning"],
        "",
        "## Next Steps",
        "",
        summary["next_recommendation"],
    ]
    if summary.get("fallback_reason"):
        lines.extend(["", f"Fallback reason: `{summary['fallback_reason']}`"])
    return "\n".join(lines) + "\n"
