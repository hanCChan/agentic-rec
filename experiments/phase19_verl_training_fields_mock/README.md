# Phase 1.9：VERL Training Fields Mock

在 Phase 1.8 tokenized batch 基础上，补齐 VERL/GRPO 训练前置字段：`position_ids`、`prompts`、`responses`、`token_level_rewards`，以及 `mock_old_log_probs` / `mock_advantages` / `mock_returns` 占位。**不训练 GRPO，不重算真实 logprob。**

## 运行

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

python scripts/smoke_verl_training_fields.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --tokenizer-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
  --output-dir experiments/phase19_verl_training_fields_mock_10
```

## 产物

- `summary.json`
- `training_fields_shapes.json`

## 说明

`mock_old_log_probs` / `mock_advantages` / `mock_returns` 仅为 shape 占位，**不可用于训练**。Phase 1.10 再做 Rec-R1/VERL DataProto 最小对齐 dry-run。
