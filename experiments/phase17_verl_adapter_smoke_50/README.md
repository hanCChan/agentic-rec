# Phase 1.7 Smoke — 50 samples

VERL-like rollout adapter smoke test（Qwen2.5-3B，max_steps=3）。

## 指标（summary.json）

| 指标 | 值 |
|------|-----|
| num_rollout_records | 50 |
| finish_rate | 1.0 |
| llm_finish_rate | 0.0 |
| auto_finish_rate | 1.0 |
| parse_success_rate | 1.0 |
| invalid_action_rate | 0.0 |
| avg_reward | 0.036 |
| avg_ndcg_at_10 | 0.092 |
| avg_search_calls | 1.96 |

## 产物

- `rollout_records.jsonl`
- `summary.json`

## 复现

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec
CUDA_VISIBLE_DEVICES=2 python scripts/smoke_verl_rollout_adapter.py \
  --num-samples 50 \
  --output-dir experiments/phase17_verl_adapter_smoke_50
```
