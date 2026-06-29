# Phase 1 实验结果

本目录存放 **CommerceAgentEnv** smoke test 与后续 agentic RL 实验的可复现输出（JSON/JSONL，体积小，纳入 Git）。

## phase1_env_smoke（2026-06-29）

**命令：**

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec
python scripts/smoke_agent_env.py --num-samples 20
```

**配置：**

| 项 | 值 |
|----|-----|
| max_steps | 3 |
| tools | `bm25_search`, `final_answer` |
| action | JSON |
| reward | NDCG@10 + λ·ΣΔNDCG − penalties |
| 数据 | `Rec-R1/data/esci/inst/sparse/subset_smoke/val.parquet` |

**结果摘要（20 条 val smoke）：**

| 模式 | avg NDCG@10 | avg total_reward | avg search_calls |
|------|-------------|------------------|------------------|
| baseline_single_shot | 0.127 | 0.141 | 1.0 |
| rule_multi_step | 0.127 | 0.078 | 1.75 |

**说明：** rule policy 为启发式冒烟策略，尚未优化检索质量；Phase 1 验收目标是 **env 全链路跑通**（tool loop、reward 分解、trajectory 落盘），不是 beat baseline。

**产物：**

- `baseline_trajectory.jsonl` — 原始 query 单次检索 + final_answer
- `rule_policy_trajectory.jsonl` — 最多 3 步 rule policy 轨迹
- `summary.json` — 聚合指标

## 下一步

- 接入 Qwen2.5-3B rollout（仍不训 GRPO，先测 LLM policy）
- 再包装 env 接入 VERL GRPO
