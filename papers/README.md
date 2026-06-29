# 参考论文库

本目录存放 **Agentic Commerce-R1** 升级方案涉及的参考论文 PDF，供设计与实现前精读。

## 论文清单

| 文件 | arXiv | 角色 | 对应升级阶段 |
|------|-------|------|-------------|
| [Search-R1.pdf](./Search-R1.pdf) | [2503.09516](https://arxiv.org/abs/2503.09516) | 多轮 search agent + retrieved token masking | 阶段 1：multi-turn tool-use |
| [Rec-R1.pdf](./Rec-R1.pdf) | [2503.24289](https://arxiv.org/abs/2503.24289) | 黑盒 NDCG/Recall reward、当前基线 | 已有复现，作为起点 |
| [OThink-SRR1.pdf](./OThink-SRR1.pdf) | [2604.19766](https://arxiv.org/abs/2604.19766) | Search-Refine-Reason、GRPO-IR | 阶段 3：refine + 过程奖励 |
| [SmartSearch.pdf](./SmartSearch.pdf) | [2601.04888](https://arxiv.org/abs/2601.04888) | Process Reward-Guided Query Refinement | 阶段 3：ΔNDCG 过程奖励 |
| [SAGE.pdf](./SAGE.pdf) | [2506.19783](https://arxiv.org/abs/2506.19783) | 策略化 query rewriting | 阶段 4：strategy action |
| [MemSearcher.pdf](./MemSearcher.pdf) | [2511.02805](https://arxiv.org/abs/2511.02805) | Compact memory + multi-context GRPO | 阶段 5：轻量 memory |
| [RAGEN.pdf](./RAGEN.pdf) | [2504.20073](https://arxiv.org/abs/2504.20073) | Agent RL 训练诊断、collapse 分析 | 阶段 7：diagnostics |
| [BGE-M3.pdf](./BGE-M3.pdf) | [2402.03216](https://arxiv.org/abs/2402.03216) | Dense / hybrid 检索基座 | 阶段 2：dense + hybrid |
| [SQID.pdf](./SQID.pdf) | [2405.15190](https://arxiv.org/abs/2405.15190) | 图文多模态商品搜索数据 | 阶段 6：多模态扩展 |
| [Qwen3.pdf](./Qwen3.pdf) | [2505.09388](https://arxiv.org/abs/2505.09388) | 更强基座模型选型参考 | 第二阶段 7B/8B |
| [Agent-Lightning.pdf](./Agent-Lightning.pdf) | [2508.03680](https://arxiv.org/abs/2508.03680) | Agent 执行与 RL 训练解耦 | 架构参考（可选） |

## 相关代码仓库（未下载 PDF）

| 项目 | URL | 说明 |
|------|-----|------|
| Search-R1 | https://github.com/PeterGriffinJin/Search-R1 | 多轮 search agent 参考实现 |
| Rec-R1 | https://github.com/linjc16/Rec-R1 | 当前工程上游 |
| VERL | https://github.com/volcengine/verl | RL 训练框架 |

## 建议阅读顺序

1. **Rec-R1** → 确认当前 baseline 能力与 reward 接口
2. **Search-R1** → 理解 multi-turn 环境与 masking
3. **SmartSearch + OThink-SRR1** → 过程奖励设计
4. **SAGE** → 策略动作空间
5. **MemSearcher** → memory 模块（轻量版）
6. **BGE-M3** → dense index 构建
7. **RAGEN** → 训练监控指标
8. **SQID + Qwen3** → 扩展实验与模型选型

## 下载方式（重下）

```bash
cd /data1/hcc/agentic-rec/papers
curl -L -o Search-R1.pdf https://arxiv.org/pdf/2503.09516.pdf
# ... 其余见上表 arXiv 链接
```
