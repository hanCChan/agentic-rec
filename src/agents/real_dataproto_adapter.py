"""
Phase 1.11 Real DataProto Adapter.

Attempts to convert DataProtoMock into verl.protocol.DataProto.
Gracefully falls back to DataProtoMock if verl/tensordict are unavailable or incompatible.

Does not launch GRPO training or VERL trainer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.agents.dataproto_mock import DataProtoMock


class RealDataProtoAdapter:
    """Convert DataProtoMock to real verl.protocol.DataProto when available."""

    def __init__(self):
        self.verl_available = False
        self.tensordict_available = False
        self.error: Optional[str] = None
        self._DataProto = None
        self._TensorDict = None
        self._import_report = self.check_imports()

    def check_imports(self) -> Dict[str, Any]:
        """Try importing verl.protocol.DataProto and tensordict.TensorDict."""
        report: Dict[str, Any] = {
            "verl_import_ok": False,
            "tensordict_import_ok": False,
            "data_proto_class": None,
            "tensor_dict_class": None,
            "error": None,
        }

        try:
            from tensordict import TensorDict

            self._TensorDict = TensorDict
            self.tensordict_available = True
            report["tensordict_import_ok"] = True
            report["tensor_dict_class"] = f"{TensorDict.__module__}.{TensorDict.__name__}"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            report["error"] = self.error
            return report

        try:
            from verl.protocol import DataProto

            self._DataProto = DataProto
            self.verl_available = True
            report["verl_import_ok"] = True
            report["data_proto_class"] = f"{DataProto.__module__}.{DataProto.__name__}"
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            report["error"] = self.error

        return report

    @staticmethod
    def _non_tensors_to_numpy(non_tensor_batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        converted: Dict[str, np.ndarray] = {}
        for key, val in non_tensor_batch.items():
            if isinstance(val, np.ndarray):
                converted[key] = val.astype(object, copy=False) if val.dtype != object else val
            elif isinstance(val, list):
                converted[key] = np.array(val, dtype=object)
            else:
                converted[key] = np.array([val], dtype=object)
        return converted

    def to_real_dataproto(self, mock_proto: DataProtoMock) -> Dict[str, Any]:
        """Convert DataProtoMock to real DataProto, or fallback on failure."""
        result: Dict[str, Any] = {
            "real_proto": None,
            "used_real_dataproto": False,
            "fallback_to_mock": True,
            "error": self.error,
        }

        if not self.verl_available or not self.tensordict_available:
            if result["error"] is None:
                result["error"] = "verl or tensordict import failed"
            return result

        try:
            DataProto = self._DataProto
            batch_size = mock_proto.batch_size()
            non_tensors = self._non_tensors_to_numpy(mock_proto.non_tensor_batch)

            if hasattr(DataProto, "from_dict"):
                real_proto = DataProto.from_dict(
                    tensors=mock_proto.batch,
                    non_tensors=non_tensors,
                    meta_info=dict(mock_proto.meta_info),
                )
            else:
                TensorDict = self._TensorDict
                td = TensorDict(mock_proto.batch, batch_size=[batch_size])
                real_proto = DataProto(
                    batch=td,
                    non_tensor_batch=non_tensors,
                    meta_info=dict(mock_proto.meta_info),
                )

            result["real_proto"] = real_proto
            result["used_real_dataproto"] = True
            result["fallback_to_mock"] = False
            result["error"] = None
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["real_proto"] = None
            result["used_real_dataproto"] = False
            result["fallback_to_mock"] = True

        return result

    def inspect_real_dataproto(
        self,
        real_proto: Any,
        mock_proto: DataProtoMock,
    ) -> Dict[str, Any]:
        """Validate real DataProto fields against mock expectations."""
        report: Dict[str, Any] = {
            "real_dataproto_check_passed": False,
            "errors": [],
        }

        def fail(msg: str) -> None:
            report["errors"].append(msg)

        try:
            assert real_proto.batch is not None, "real_proto.batch missing"
            assert real_proto.non_tensor_batch is not None, "real_proto.non_tensor_batch missing"
            assert real_proto.meta_info is not None, "real_proto.meta_info missing"

            batch_size = int(real_proto.batch.batch_size[0])
            mock_batch_size = mock_proto.batch_size()
            if batch_size != mock_batch_size:
                fail(f"batch_size {batch_size} != mock {mock_batch_size}")

            mock_input_shape = list(mock_proto.batch["input_ids"].shape)
            real_input_shape = list(real_proto.batch["input_ids"].shape)
            if real_input_shape != mock_input_shape:
                fail(f"input_ids shape {real_input_shape} != mock {mock_input_shape}")

            mock_resp_shape = list(mock_proto.batch["responses"].shape)
            real_resp_shape = list(real_proto.batch["responses"].shape)
            if real_resp_shape != mock_resp_shape:
                fail(f"responses shape {real_resp_shape} != mock {mock_resp_shape}")

            sample_ids = real_proto.non_tensor_batch.get("sample_ids")
            if sample_ids is None:
                fail("non_tensor_batch missing sample_ids")
            elif len(sample_ids) != batch_size:
                fail(f"sample_ids length {len(sample_ids)} != batch_size {batch_size}")

            if "warning" not in real_proto.meta_info:
                fail("meta_info missing warning")

            report["real_dataproto_check_passed"] = len(report["errors"]) == 0
            report["batch_size"] = batch_size
            report["batch_keys"] = sorted(list(real_proto.batch.keys()))
            report["non_tensor_keys"] = sorted(list(real_proto.non_tensor_batch.keys()))
            report["meta_info_keys"] = sorted(list(real_proto.meta_info.keys()))
            report["input_ids_shape"] = real_input_shape
            report["responses_shape"] = real_resp_shape
        except AssertionError as exc:
            fail(str(exc))
            report["real_dataproto_check_passed"] = False

        return report
