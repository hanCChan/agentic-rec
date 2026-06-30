# Phase 1.11 Smoke — 10 records

Real `verl.protocol.DataProto` compatibility check（DataProtoMock → TensorDict → DataProto）。

## 指标（summary.json）

| 指标 | 值 |
|------|-----|
| mock_validate_passed | true |
| verl_import_ok | true |
| tensordict_import_ok | true |
| used_real_dataproto | true |
| fallback_to_mock | false |
| real_dataproto_check_passed | true |

## 产物

- `summary.json`
- `compatibility_report.json`

## 复现

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec
python scripts/smoke_real_dataproto_compat.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase111_real_dataproto_compat_10
```
