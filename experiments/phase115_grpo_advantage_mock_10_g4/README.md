# Phase 1.15 Smoke — 10 base × group 4

Synthetic GRPO grouped advantage dry-run（`zero_std_group_rate=0`，`mean_abs_sequence_advantage≈0.89`）。

## 复现

```bash
python scripts/smoke_grpo_advantage_mock.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase115_grpo_advantage_mock_10_g4 \
  --num-base-records 10 --group-size 4
```
