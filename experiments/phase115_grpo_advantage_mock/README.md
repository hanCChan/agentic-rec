# Phase 1.15：GRPO Advantage Mock / Grouped Reward Dry-Run

构造 synthetic grouped rollout（同一 prompt 多 response），计算 GRPO 组内归一化 advantage。**不训练、不接 GRPO trainer。**

## 运行

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

python scripts/smoke_grpo_advantage_mock.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase115_grpo_advantage_mock_10_g4 \
  --num-base-records 10 \
  --group-size 4 \
  --synthetic-reward-jitter 0.02
```

## Collapse 诊断

```bash
python scripts/smoke_grpo_advantage_mock.py \
  --output-dir experiments/phase115_grpo_advantage_mock_10_g4_collapse \
  --num-base-records 10 \
  --group-size 4 \
  --no-synthetic-jitter \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl
```

## 说明

Synthetic group 仅用于 advantage shape dry-run。Phase 1.16 再做 GRPO loss dry-run。
