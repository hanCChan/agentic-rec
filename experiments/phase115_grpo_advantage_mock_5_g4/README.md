# Phase 1.15 Smoke — 5 base × group 4

Synthetic GRPO grouped advantage dry-run。

## 复现

```bash
python scripts/smoke_grpo_advantage_mock.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase115_grpo_advantage_mock_5_g4 \
  --num-base-records 5 --group-size 4
```
