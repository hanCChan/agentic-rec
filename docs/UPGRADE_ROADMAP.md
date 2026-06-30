# Agentic Commerce-R1 升级路线图

> 从 Rec-R1 复现 → 面向电商搜索的可训练 Agentic RL 检索决策系统

## 定位一句话

训练 LLM Agent 学会 **query 理解、检索策略选择、证据筛选、查询重写、结果反思与排序决策**；推荐只是场景，主线是 **Agentic RL + Search/RAG + 工业检索优化**。

## 当前 vs 目标

| 维度 | 当前（已跑通） | 目标 |
|------|---------------|------|
| 交互 | single-shot query rewrite | multi-turn tool-use agent |
| 检索 | 固定 BM25 | BM25 + dense + hybrid + rerank（检索器固定） |
| Reward | 最终 NDCG/Recall | outcome + process（ΔNDCG、evidence、cost） |
| 策略 | 自由生成 query | 可选 strategy action（SAGE） |
| Memory | 无 | compact JSON state（MemSearcher 轻量版） |
| 多模态 | 无 | SQID + image caption（Phase 6） |
| 诊断 | 无 | rollout drift / collapse（RAGEN） |

## 分阶段实施（8 周参考）

| 阶段 | 内容 | 参考论文 | 产出 |
|------|------|---------|------|
| **0** | 读论文 + 答疑 | 全部 `papers/` | `QUESTIONS_BEFORE_UPGRADE.md` 确认 |
| **1** | Multi-turn tool-use env | Search-R1 | `commerce_agent_env.py` + smoke |
| **2** | Dense + hybrid 检索 | BGE-M3 | `dense_tool.py`, Faiss index |
| **3** | Process reward | SmartSearch, OThink-SRR1 | `process_reward.py` |
| **4** | Strategy action | SAGE | `strategy_reward.py` |
| **5** | Compact memory | MemSearcher | `memory_state.py` |
| **6** | 多模态 corpus | SQID | caption + visual attrs |
| **7** | 诊断 + 主实验 | RAGEN | ablation 表格 + 案例分析 |
| **8** | 文档 + 面试材料 | — | README / 技术报告 |

## 目标目录结构（确认后创建）

```text
agentic-rec/
├── papers/                    # 参考论文 PDF
├── docs/
│   ├── UPGRADE_ROADMAP.md     # 本文件
│   └── QUESTIONS_BEFORE_UPGRADE.md
├── src/                       # 新增（与 Rec-R1 并列）
│   ├── agents/
│   ├── tools/
│   ├── reward/
│   ├── retriever/
│   ├── multimodal/
│   └── diagnostics/
├── scripts/
│   ├── train_agentic_grpo_3b.sh
│   └── eval_agentic_search.sh
└── Rec-R1/                    # 现有 VERL + Rec-R1
```

## GPU 规划（6×A100-80G）

| 阶段 | 模型 | GPU 分配 |
|------|------|---------|
| MVP | Qwen2.5-3B | 4 train + 2 vLLM/retrieval |
| 主实验 | Qwen2.5-7B / Qwen3-8B | 同上，rollout batch 减小 |
| 多模态 | 离线 VL caption | 不占 RL GPU |

## 实验矩阵（最终报告）

1. **主效果**：BM25 original → prompt → Rec-R1 → Search-R1-style → Ours (+ hybrid + process)  
2. **工具消融**：BM25 / Dense / Hybrid / +Rerank  
3. **Reward 消融**：outcome only → +ΔNDCG → +evidence → +cost  
4. **策略消融**：no strategy → fixed → agent-selected  
5. **多模态**：text only → +caption → +visual attrs  

## 不做的事

- 从零预训练大模型  
- 端到端训练检索器  
- 完整在线推荐系统  
- 一开始 14B+  
- 复杂 multi-agent 协作  

## 下一步

1. ~~阅读 `papers/README.md` 建议顺序~~
2. ~~回复 `docs/QUESTIONS_BEFORE_UPGRADE.md` 第十一节 5 个优先问题~~ ✅ 已确认
3. ~~确认 MVP 后开始阶段 1 代码~~ ✅ 见 `experiments/phase1_env_smoke/`
4. ~~**进行中**：Qwen rollout policy smoke~~ ✅ 见 `experiments/phase15_qwen_rollout_smoke/`
5. ~~**下一步**：prompt 强化 final_answer → VERL rollout wrapper~~ ✅ Phase 1.6 finish-aware 完成
6. ~~**下一步**：VERL rollout wrapper 骨架（仍不训 GRPO）~~ ✅ Phase 1.7 完成，见 `experiments/phase17_verl_adapter_smoke_10/`
7. ~~**下一步**：Phase 1.8 — VERL batch mock / reward tensor / token mask shape check~~ ✅ 见 `experiments/phase18_verl_batch_mock_10/`
8. ~~**下一步**：Phase 1.9 — VERL training fields mock / field alignment~~ ✅ 见 `experiments/phase19_verl_training_fields_mock_10/`
9. ~~**下一步**：Phase 1.10 — DataProto / reward_fn dry-run~~ ✅ 见 `experiments/phase110_dataproto_reward_dryrun_10/`
10. ~~**下一步**：Phase 1.11 — 真实 `verl.DataProto` import 兼容性检查~~ ✅ 见 `experiments/phase111_real_dataproto_compat_10/`
11. ~~**下一步**：Phase 1.12 — Actor LogProb dry-run 设计（仍不调用 actor.forward）~~ ✅ 见 `experiments/phase112_actor_logprob_mock_10/`
12. ~~**下一步**：Phase 1.13 — 真实 actor logprob dry-run（单 batch、no grad、仍不接 trainer）~~ ✅ 见 `experiments/phase113_actor_logprob_dryrun_2/`
13. ~~**下一步**：Phase 1.14 — Reference LogProb / KL Dry-Run 设计~~ ✅ 见 `experiments/phase114_ref_kl_dryrun_2/`
14. ~~**下一步**：Phase 1.15 — GRPO Advantage Mock / Grouped Reward Dry-Run~~ ✅ 见 `experiments/phase115_grpo_advantage_mock_10_g4/`
15. ~~**下一步**：Phase 1.16 — GRPO Loss Dry-Run~~ ✅ 见 `experiments/phase116_grpo_loss_dryrun_10_g4/`
16. ~~**下一步**：Phase 1.17 — Real Multi-Sample Rollout Smoke~~ ✅ 见 `experiments/phase117_multisample_rollout_5_g4/`
17. ~~**下一步**：Phase 1.18 — Real GRPO Loss Dry-Run（或若 `zero_std_group_rate` 高则先做 Phase 1.18a Rollout Diversity Diagnostics）~~ ✅ 见 `experiments/phase118a_rollout_diagnostics_5_g4/`
18. ~~**下一步**：Phase 1.18b — Reward Sensitivity Diagnostics~~ ✅ 见 `experiments/phase118b_reward_sensitivity_5_g4/`
19. ~~**下一步**：Phase 1.18c — Reward Shaping Proposal + Dry-Run~~ ✅ 见 `experiments/phase118c_reward_shaping_dryrun_5_g4/`
20. ~~**下一步**：Phase 1.18c — Reward Shaping Proposal + Dry-Run~~ ✅ 见 `experiments/phase118c_reward_shaping_dryrun_5_g4/`
21. ~~**下一步**：Phase 1.18d — Rollout Diversity Prompt / Search Strategy Fix~~ ✅ 见 `experiments/phase118d_strategy_rollout_5_g4/`
22. ~~**下一步**：Phase 1.18d — Rollout Diversity Prompt / Search Strategy Fix~~ ✅ 见 `experiments/phase118d_strategy_rollout_5_g4/`
23. ~~**下一步**：Phase 1.19a — Strategy Rollout Reward Decomposition~~ ✅ 见 `experiments/phase119a_strategy_reward_decomposition_5_g4/`
24. **下一步**：Phase 1.19b 或 1.19 — 取决于 1.19a gate 结果
