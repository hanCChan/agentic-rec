# Phase 1.5：Qwen Rollout Smoke

LLM（Qwen2.5-3B-Instruct + vLLM）作为 JSON policy，驱动 CommerceAgentEnv。**不训练，不接 GRPO。**

## 运行命令

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

# 5 条 smoke
CUDA_VISIBLE_DEVICES=2 python scripts/smoke_qwen_rollout.py \
  --num-samples 5 \
  --max-steps 3 \
  --model-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
  --output-dir experiments/phase15_qwen_rollout_smoke_5

# 10 条 smoke
CUDA_VISIBLE_DEVICES=2 python scripts/smoke_qwen_rollout.py \
  --num-samples 10 \
  --max-steps 3 \
  --model-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
  --output-dir experiments/phase15_qwen_rollout_smoke_10
```

## 结果摘要（2026-06-29）

| 指标 | 5 条 | 10 条 |
|------|------|-------|
| parse_success_rate | **1.00** | **1.00** |
| invalid_action_rate | 0.00 | 0.00 |
| finish_rate | 0.00 | 0.00 |
| avg_search_calls | 2.80 | 2.60 |
| avg_ndcg_at_10 | 0.074 | 0.092 |
| avg_total_reward | -0.172 | -0.159 |
| avg_output_tokens | 76.4 | 82.0 |

**验收：** parse ≥ 0.60 ✅ | invalid ≤ 0.40 ✅ | trajectory + summary 落盘 ✅

**观察：** JSON 解析 100% 成功，但模型几乎从不调用 `final_answer`（3 步全用于 search，触发 `max_steps_without_final`）。下一步应在 prompt 中强化「剩余 1 步必须 final_answer」，再接入 VERL rollout wrapper。

## 产物

- `trajectory.jsonl` — 含 `prompt` / `raw_output` / `parse_ok` / 逐步 reward
- `summary.json` — 聚合指标
