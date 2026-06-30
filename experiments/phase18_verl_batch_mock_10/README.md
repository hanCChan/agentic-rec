# Phase 1.8 Smoke — 10 records

VERL batch mock shape check（Qwen2.5-3B tokenizer，Phase 1.7 rollout 输入）。

## 指标（summary.json）

| 指标 | 值 |
|------|-----|
| batch_size | 10 |
| seq_len | 1457 |
| shape_check_passed | true |
| reward_mean | 0.033 |
| avg_prompt_length | 55.3 |
| avg_response_length | 1211.8 |
| num_response_truncated | 0 |

## 产物

- `summary.json`
- `batch_shapes.json`
- `batch_meta.json`

## 复现

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec
python scripts/smoke_verl_batch_mock.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase18_verl_batch_mock_10
```
