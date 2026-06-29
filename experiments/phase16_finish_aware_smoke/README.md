# Phase 1.6：Finish-Aware Rollout Fix

在 Phase 1.5 基础上修复 episode 无法 `final_answer` 的问题。

## 改动

| 项 | 说明 |
|----|------|
| Prompt | 增加 `current_step` / `remaining_steps` / `best_query_by_ndcg` |
| Auto-finalize | 最后 1 步不调用 LLM，自动 `final_answer(best_query)` |
| Parser | `search` → `bm25_search` 别名（轻量 wrapper） |
| Summary | 区分 `llm_finish_rate` / `auto_finish_rate` / `finish_rate` |

## 运行

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

CUDA_VISIBLE_DEVICES=2 python scripts/smoke_qwen_rollout.py \
  --num-samples 10 \
  --max-steps 3 \
  --model-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
  --output-dir experiments/phase16_finish_aware_smoke_10
```

## 结果（2026-06-29，10 条）

| 指标 | 值 | 验收 |
|------|-----|------|
| parse_success_rate | **1.00** | ✅ ≥ 0.95 |
| invalid_action_rate | **0.00** | ✅ ≤ 0.05 |
| finish_rate | **1.00** | ✅ ≥ 0.90 |
| llm_finish_rate | 0.00 | LLM 仍倾向 search |
| auto_finish_rate | **1.00** | 最后一步脚本兜底 |
| avg_search_calls | **1.90** | ✅ ≤ 2.20 |
| avg_ndcg_at_10 | 0.092 | 非本阶段目标 |

## 下一步

VERL rollout wrapper（仍不训练 GRPO）。
