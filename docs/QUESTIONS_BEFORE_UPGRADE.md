# Agentic Commerce-R1 升级：实施前待澄清问题

> 状态：**先读论文、先答疑、再动手**  
> 参考论文：`papers/` 目录（11 篇 PDF）  
> 当前基线：single-shot query rewrite + BM25 + NDCG/Recall + GRPO（已跑通）

---

## 一、我对升级方向的理解（请确认）

当前仓库是 **Rec-R1 复现**，本质是：

```text
用户 query → LLM 一次输出 rewritten query → BM25 → NDCG/Recall reward → GRPO
```

目标升级为 **Agentic Commerce-R1**：

```text
用户意图 → Agent 多步 plan → 选策略 → 调 BM25/dense/hybrid/rerank 工具
         → 观察 topK → refine 证据 → 必要时再搜 → 最终决策
         → outcome + process reward → GRPO
```

**核心转变**：从「query rewriter」到「可训练的多工具检索决策 Agent」。推荐/电商只是场景，技术标签是 Search-R1 + process reward + hybrid retrieval + memory + diagnostics。

若以上理解有偏差，请先纠正再进入实现。

---

## 二、架构层：必须先拍板的问题

### Q1. Multi-turn 环境接在哪里？

当前 Rec-R1 reward 在 `verl/utils/reward_score/esci.py`，rollout 是 vLLM 单次生成。

**选项 A**：在 VERL rollout 内嵌 tool loop（类似 Search-R1，改 `main_ppo` + custom rollout）  
**选项 B**：独立 `CommerceAgentEnv`，VERL 只负责 policy 更新，环境在外部 Ray actor  
**选项 C**：先用 Agent Lightning 思路解耦 agent 执行与 RL（读 `Agent-Lightning.pdf` 后决定）

> **疑问**：Search-R1 与 Rec-R1 都基于 VERL，你更倾向 A（改 verl 深）还是 B（环境独立、侵入小）？6×A100 下 rollout 与 retrieval service 如何分 GPU？

---

### Q2. Retrieved token masking 是否必须？

Search-R1 强调：检索回来的 token **不参与梯度**，否则训练不稳定。

当前 Rec-R1 是 BM25 reward 在 Python 侧算分，模型不 ingest 完整检索结果到 context（只生成 query）。

一旦改成 multi-turn + `<observation>` 注入商品列表：

> **疑问**：是否严格实现 Search-R1 式 masking？observation 部分全部 mask loss？还是 observation 用短摘要（refine 后）且仍 mask？

---

### Q3. 最大 tool 步数与 episode 长度

MemSearcher / OThink-SRR1 都惩罚过多检索。

> **疑问**：ESCI 上单 query 允许最多几步？建议初版 `max_steps=3` 还是 `5`？超出步数 reward 如何截断（硬 penalty vs 强制 `<final>`）？

---

### Q4. Action 格式：JSON vs XML tag

方案里混用了 JSON action 和 Rec-R1 现有 `<answer>` XML。

> **疑问**：是否统一为 JSON tool call（便于 parser + 无效 action penalty）？还是保留 XML 以兼容现有 GRPO 数据格式？

---

## 三、检索层：dense / hybrid 的工程边界

### Q5. Dense retriever 选型

方案提 BGE-M3 + Faiss。论文 `BGE-M3.pdf` 支持 dense + sparse + multi-vector。

> **疑问**：初版只用 **dense embedding + Faiss** 是否足够？还是直接上 BGE-M3 的 hybrid 分数（省一个 α 融合层）？索引建在 ESCI 全量语料还是 subset 涉及 ASIN？

---

### Q6. Hybrid α 谁定？

方案：`score = α * bm25 + (1-α) * dense`。

> **疑问**：α 是 Agent **可调用参数**（`hybrid_search` 带 alpha）还是训练外固定超参？若 Agent 可选 α，action space 是否过大？

---

### Q7. Rerank 是否进 RL loop

Cross-encoder rerank 延迟高。

> **疑问**：Phase 1 是否 **只做 offline rerank 评估**，不进 rollout？还是用小 cross-encoder（如 bge-reranker-base）作为可选 tool？

---

## 四、Reward 设计：最核心也最容易踩坑

### Q8. Outcome vs Process 如何合并进 GRPO？

GRPO 是 group-relative，通常一个 trajectory 一个 scalar reward。

方案提出：

```text
R_final = 0.5*NDCG@10 + 0.3*Recall@50 + ...
R_step  = ΔNDCG + EvidencePrecision - cost - repeat
```

> **疑问**：  
> - 过程奖励是 **逐步累加** 到 trajectory 末尾，还是 **只在最后一步** 加权和？  
> - Search-R1 / SmartSearch 具体怎么做 credit assignment？需对照 `SmartSearch.pdf` §reward。  
> - ESCI 标签是 query-document 相关性，中间步 ΔNDCG 用 **当前步 query** 算还是 **累积 best query** 算？

---

### Q9. Evidence Precision 如何定义

OThink-SRR1 奖励「准确证据识别」。

ESCI 有 E/S/C/I 四档标签。

> **疑问**：EvidencePrecision = topK 中 Exact+Substitute 占比？是否区分 E vs S 权重？refine 步骤的 evidence 指 **模型显式列出的商品属性** 还是 **检索 hit 集合**？

---

### Q10. Invalid / Repeated action penalty

> **疑问**：无效 JSON、未知 tool、空 query 的 penalty 量级相对 NDCG（0~1）设多少？重复 query 是 exact match 还是 embedding 相似度 > 0.95 算重复？

---

### Q11. Strategy reward 是否独立

SAGE 有 Strategic Credit Shaping。

> **疑问**：strategy 标签是 **离散分类 head** 还是 **生成式**（模型输出 strategy 字段）？策略选错但 NDCG 高，要不要惩罚？

---

## 五、Memory：轻量版的范围

### Q12. 先做哪种 memory 场景

MemSearcher 是全对话 memory；方案建议轻量 JSON state。

ESCI 原始数据 **以单 query 为主**，不天然多轮。

> **疑问**：  
> - 是否先用 **合成多轮**（从单 query 拆 constraint 轮）做 memory 实验？  
> - 还是直接接 Amazon Review 序列 / 自建 dialog？  
> - Memory 更新由 **规则 parser** 还是 **模型生成 memory block**？

---

## 六、多模态：Phase 边界

### Q13. SQID 数据获取

`SQID.pdf` 描述 ESCI 图像增强版（~19 万商品图）。

> **疑问**：SQID 是否有公开 HF 数据集可直接拉？还是需单独申请？若拿不到，是否用 **caption 合成**（Qwen2.5-VL 离线打标）在 ESCI 子集上做 demo？

---

### Q14. 多模态进 RL 还是只进 corpus

> **疑问**：Phase 6 是否 **只增强商品文档**（image caption → BM25/dense 文本索引），Agent 仍纯文本？还是支持用户上传图片 query（太重，建议否）？

---

## 七、训练与资源：6×A100 的可行配置

### Q15. 阶段 1 最小可验证实验（MVP）

建议 MVP 范围：

```text
✓ multi-turn (max 3 steps)
✓ tools: bm25_search only
✓ reward: final NDCG + ΔNDCG step penalty
✗ dense / hybrid / memory / multimodal（后续）
```

> **疑问**：你是否同意 **先 MVP 再叠功能**？还是坚持第一次 commit 就要 hybrid + process + strategy 全套？

---

### Q16. 基座模型路线

方案：3B/4B 跑通 → 7B/8B 主实验。

> **疑问**：继续 **Qwen2.5-3B-Instruct**（已下载）还是换 **Qwen3-4B**？换基座需重跑 smoke，是否接受？

---

### Q17. 与现有 Rec-R1 checkpoint 的关系

已有 `esci-qwen3b-grpo-smoke` checkpoint（global_step_16）。

> **疑问**：multi-turn agent 是否 **冷启动** 新训？还是用 single-shot checkpoint **warm start**（可能带错误格式 bias）？

---

## 八、评估与诊断：什么算「做完」

### Q18. 主对比表最少几行

方案列了 6 组方法对比。

> **疑问**：面试 MVP 是否 **4 行足够**：BM25 original / prompt rewrite / Rec-R1 single-shot / Agentic Commerce-R1 (ours)？

---

### Q19. Diagnostics 必做指标

RAGEN 提 collapse、drift。

> **疑问**：训练过程中是否 **强制 wandb/console 打**：avg tool calls、repeat rate、invalid rate、reward variance、KL？先实现 `rollout_analyzer.py` 离线分析是否可接受？

---

## 九、仓库与 GitHub 策略

### Q20. 代码组织

方案建议新建 `src/agents/`、`src/tools/` 等于 `Rec-R1/` 并列。

> **疑问**：新代码放 **`agentic-rec/src/`**（wrapper 层）还是 **`Rec-R1/src/`**（深改上游）？倾向少改 verl 内核。

---

### Q21. 论文与文档是否 push GitHub

`papers/*.pdf` 合计 ~22MB。

> **疑问**：PDF 是否 push 到 GitHub（可能超 100MB 单文件限制但总量 OK）？还是只保留 README 链接 + 本地目录？

---

## 十、建议的「答疑 → 实施」顺序

| 轮次 | 先回答的问题 | 再做什么 |
|------|-------------|---------|
| **Round 1** | Q1 Q2 Q4 Q15 Q16 | 定 MVP 范围 + 环境接口草图 |
| **Round 2** | Q8 Q9 Q10 | 定 reward 公式 + 对照 SmartSearch/OThink 论文 |
| **Round 3** | Q5 Q6 Q7 | 定 dense/hybrid 是否进 MVP |
| **Round 4** | Q12 Q13 | 定 memory / multimodal 是否 Phase 2 |
| **Round 5** | Q18 Q19 Q20 | 定评估与 repo 结构，开始写代码 |

---

## 十一、请你优先回复的 5 个问题（最高优先级）

1. **MVP 边界**：是否同意「3 步 multi-turn + BM25 only + final+ΔNDCG reward」作为第一版可跑通目标？  
2. **环境接入方式**：VERL 内嵌 tool loop（A）还是独立 Ray env（B）？  
3. **Action 格式**：JSON tool call 还是 XML？  
4. **基座模型**：Qwen2.5-3B 继续用还是换 Qwen3-4B？  
5. **ESCI 单 query 数据**：多轮 memory 是否先用合成 dialog，还是换数据集？

---

请逐条回复或标注「按你推荐默认值即可」。确认后再进入 **阶段 1 代码实现**（不改训练，先搭 agent env + smoke test）。
