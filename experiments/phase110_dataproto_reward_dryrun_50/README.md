# Phase 1.10 Smoke — 50 records

DataProtoMock + CommerceRewardFn dry-run（Phase 1.7 rollout → Phase 1.8/1.9 → DataProto-like payload）。

## 指标（summary.json）

| 指标 | 值 |
|------|-----|
| batch_size | 50 |
| dataproto_validate_passed | true |
| actor_input_check_passed | true |
| reward_fn_check_passed | true |
| reward_mean | 0.036 |
| num_nonzero_token_rewards | 50 |
| missing_actor_keys | [] |

## 产物

- `summary.json`
- `dataproto_shapes.json`
- `reward_fn_output.json`

## 复现

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec
python scripts/smoke_dataproto_reward_dryrun.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_50/rollout_records.jsonl \
  --output-dir experiments/phase110_dataproto_reward_dryrun_50
```
