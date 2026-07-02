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
23. ~~**下一步**：Phase 1.19a — Strategy Rollout Reward Decomposition~~ ✅ 见 `experiments/phase119a_strategy_reward_decomposition_5_g4/`（gate **未通过**：`retrieval_quality_spread=0.20`，`penalty_only_spread=0.80`，`zero_std_quality_only=0.80`）
24. ~~**下一步（阻塞 Phase 1.19）**：Phase 1.18e — Qrels / Metric Blindness~~ ✅ 见 `experiments/phase118e_qrels_metric_blindness_5_g4/`
25. ~~**下一步**：Phase 1.18f — Large-K Reward Candidate Dry-Run~~ ✅ 见 `experiments/phase118f_large_k_reward_dryrun_5_g4/`
26. ~~**下一步**：Phase 1.19 — Real GRPO Loss Dry-Run with reward_largek_mix_1000~~ ✅ 见 `experiments/phase119_real_grpo_loss_dryrun_5_g4_largek1000/`
27. ~~**进行中**：Phase 1.19b — 10_g4 / 20_g4 scale gate check~~ ✅ 见 `experiments/phase119b_scale_gate_check/`（**stable gate passed**）
28. ~~**下一步**：Phase 1.20 — No-update VERL Trainer Dry-Run~~ ✅ 见 `experiments/phase120_no_update_trainer_dryrun_20_g4/`
29. ~~**下一步**：Phase 1.18g — BM25 Failure / Unlearnable Sample Cleanup~~ ✅ 见 `experiments/phase118g_bm25_failure_cleanup_20_g4/`
30. ~~**下一步**：Phase 1.18h — Strategy Prompt V2 for Collapse Cases~~ ✅ 见 `experiments/phase118h_strategy_prompt_v2_20_g4/`（19 组 gate 通过）；`esci_val_3` targeted 未修复 → **用 1.18g replacement 替换**
31. ~~**下一步**：Phase 2.1 — Tiny GRPO Smoke Training~~ ✅ 见 `experiments/phase21_tiny_grpo_smoke/`（commit `7bb5ade`；clean 20_g4，1-step，`optimizer_step_called=true`，checkpoint `SMOKE_ONLY_DO_NOT_PROMOTE`）
32. ~~**下一步**：Phase 2.2 — 3-Step GRPO Stability Smoke~~ ✅ 见 `experiments/phase22_3step_grpo_stability_smoke/`（commit `27b7bf0`；修复 KL 诊断：`approx_kl_nonnegative`，ref logprob snapshot；3/3 steps 稳定）
33. ~~**下一步**：Phase 2.3 — 10-Step Controlled GRPO Smoke~~ ✅ 见 `experiments/phase23_10step_grpo_controlled_smoke/`（commit `5475563`；lr=1e-6 与 5e-7 均通过；`both_stable=true`，推荐 lr=5e-7）
34. ~~**下一步**：Phase 2.4a — 50-Step Pilot GRPO Training Plan~~ ✅ 见 `docs/PHASE2_4_50STEP_PILOT_PLAN.md`（commit `6d9953c`）
35. ~~**下一步**：Phase 2.4 — 50-step controlled pilot~~ ✅ 见 `experiments/phase24_50step_grpo_pilot/`（commit `286b12c`；lr=5e-7，`pilot_passed=true`，fresh eval 0.373→0.395）
36. ~~**下一步**：Phase 2.5 — 扩 clean set + held-out + 200-step pilot 计划~~ ✅ Phase 2.5a-b clean set ready（commit `c7c0d7c`）
37. ~~**下一步**：Phase 2.5c — 200-step GRPO pilot（lr=5e-7, kl_coef=0.01）~~ ⚠️ 117/200 step KL stop（`approx_kl=0.202`）；见 `experiments/phase25_200step_grpo_pilot/lr_5e-7/kl_stop_report.md`
38. ~~**下一步**：Phase 2.5d-A — eval hook fix + KL-controlled rerun（kl_coef=0.02）~~ ⚠️ 同样 117/200 KL stop（`approx_kl=0.204`）；kl_coef 加倍未改变 stop horizon
39. **下一步**：Phase 2.5e — KL/loss wiring audit（验证 kl_coef 是否进入 actor loss）；audit 通过后再跑 config B

## Claim Boundary（当前口径）

```text
Rec-R1 范式延伸 + Agentic Search 工程化 + pilot 级 GRPO 验证
≠ 完整 Rec-R1 benchmark 复现并超越

Phase 2.4: pilot 工程稳定 + curated 20 groups fresh eval early signal
Phase 2.5c: 200-step pilot KL drift at step 117 — not reward/JSON collapse; fix KL control before scaling
Phase 2.5d-A: kl_coef=0.02 reproduced same 117-step KL stop — suggests KL loss wiring audit needed
Phase 2.5e+: audit kl_coef → total_loss → gradient; then config B or periodic re-rollout
```
