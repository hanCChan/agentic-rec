# Phase 1.7：VERL Rollout Adapter 骨架

将 `CommerceAgentEnv + QwenRolloutPolicy` episode 转为 VERL-like rollout record。**不训练 GRPO。**

## 运行

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

CUDA_VISIBLE_DEVICES=2 python scripts/smoke_verl_rollout_adapter.py \
  --num-samples 10 \
  --max-steps 3 \
  --model-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
  --output-dir experiments/phase17_verl_adapter_smoke_10
```

## Rollout Record 字段

| 字段 | 说明 |
|------|------|
| `sample_id` | ESCI qid |
| `prompt` | Actor 初始 prompt（任务 + user query） |
| `response` | 多步 `<step_i>` + `<observation_i>` 串联 |
| `reward` | episode `total_reward` (float) |
| `trajectory` | 完整 CommerceAgentEnv 轨迹 |
| `metrics` | ndcg / finish / search_calls 等标量 |
| `extra_info` | debug 用 best_query / final_query 等 |

## 产物

- `rollout_records.jsonl`
- `summary.json`

## 说明

本阶段是 **VERL-like 数据结构**，尚未接入 VERL rollout worker / logprob / token mask。Phase 1.8 再接 Rec-R1/VERL 入口。
