# Phase 1.12 Smoke — 10 records

Actor logprob interface mock（真实 DataProto + mock old_log_probs/entropys shape check）。

## 指标（summary.json）

| 指标 | 值 |
|------|-----|
| used_real_dataproto | true |
| actor_input_check_passed | true |
| logprob_shape_check_passed | true |
| missing_actor_keys | [] |
| is_mock | true |
| old_log_probs_shape | [10, 1402] |

## 产物

- `summary.json`
- `actor_logprob_request_shapes.json`
- `mock_logprob_output_shapes.json`

## 复现

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec
python scripts/smoke_actor_logprob_mock.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase112_actor_logprob_mock_10
```
