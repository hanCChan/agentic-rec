# Phase 1.9 Smoke — 10 records

VERL training fields mock（Phase 1.7 rollout → Phase 1.8 batch → Phase 1.9 fields）。

## 指标（summary.json）

| 指标 | 值 |
|------|-----|
| batch_size | 10 |
| prompts_shape | [10, 59] |
| responses_shape | [10, 1402] |
| shape_check_passed | true |
| reward_mean | 0.033 |
| num_nonzero_token_rewards | 10 |

## 产物

- `summary.json`
- `training_fields_shapes.json`

## 复现

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec
python scripts/smoke_verl_training_fields.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase19_verl_training_fields_mock_10
```
