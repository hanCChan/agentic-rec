# Phase 1.15 Collapse 诊断 — 10 base × group 4, no jitter

`--no-synthetic-jitter`：组内 reward 相同，预期 `zero_std_group_rate=1.0`，advantage collapse。

## 复现

```bash
python scripts/smoke_grpo_advantage_mock.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase115_grpo_advantage_mock_10_g4_collapse \
  --num-base-records 10 --group-size 4 --no-synthetic-jitter
```
