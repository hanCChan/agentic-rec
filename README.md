# agentic-rec

基于 [Rec-R1](https://github.com/linjc16/Rec-R1) 的 **Agentic 强化学习推荐** 复现工程。目标是用 GRPO 训练 LLM，使其学会根据用户意图生成检索查询，并通过 BM25 检索 + NDCG/Recall 奖励形成闭环，同时覆盖「大模型 RL 算法」与「生成式推荐」两条研究线。

> 上游论文：*Rec-R1: Bridging Generative Large Language Models and User-Centric Recommendation Systems via Reinforcement Learning*（TMLR 2025）  
> 本仓库在官方 Rec-R1 之上，补充了国内可用的数据镜像、环境脚本、冒烟/全量训练脚本与中文说明。

---

## 项目结构

```
agentic-rec/
├── README.md              # 本说明（中文）
├── DATA.md                # 数据下载与镜像说明
├── env.sh                 # 一键激活 conda + Java + HF 缓存
├── run_full_train.sh      # 后台启动 ESCI 全量 GRPO 训练
└── Rec-R1/                # Rec-R1 主代码（含 verl 框架）
    ├── verl/              # VERL 强化学习训练框架
    ├── src/Lucene/        # BM25 稀疏检索（Pyserini）
    ├── src/dataset/       # 数据集预处理脚本
    ├── scripts/train/     # 训练脚本（含本仓库新增的 smoke/full）
    ├── scripts/eval/      # 查询生成评测
    ├── scripts/eval_search/ # 检索指标评测（Recall@K, NDCG@K）
    └── data/              # 训练 parquet（小文件随仓库；大语料需本地下载）
```

**不纳入 Git 的本地产物**（见 `.gitignore`）：`checkpoints/`、`database/`（Lucene 索引）、`outputs/`、日志、原始语料 `data/esci/raw/` 等。

---

## 核心思路

| 环节 | 说明 |
|------|------|
| **任务** | 给定用户购物查询/意图，LLM 输出 `` 推理 + `<answer>` 检索查询 |
| **检索** | 固定 BM25 索引（Pyserini/Lucene），不训练检索器 |
| **奖励** | 用 NDCG@K / Recall@K 等排序指标作为 outcome reward |
| **算法** | GRPO（Group Relative Policy Optimization），基于 VERL + vLLM rollout |
| **基座** | Qwen2.5-3B-Instruct（可换其他 Instruct 模型） |

与 Search-R1 类似：模型学会**何时、如何改写查询**以提升检索效果；与纯 SFT 不同，奖励直接来自推荐/检索指标。

---

## 环境要求

- **GPU**：建议 ≥2 张 A100-80GB（本机实测 8×A100，训练默认用 GPU 2,3）
- **Python**：3.9（conda 环境名 `recr1`）
- **关键依赖**：PyTorch 2.4、vLLM 0.6.3、Ray、VERL（`pip install -e Rec-R1`）、flash-attn、pyserini（需 **Java 17**）、faiss-gpu

### 一键激活

```bash
source /data1/hcc/agentic-rec/env.sh
```

`env.sh` 会激活 `recr1` 环境，设置 `JAVA_HOME`（BM25 索引）、`HF_HOME`（模型缓存到大盘）等。

### 从零安装（新机器）

```bash
conda create -n recr1 python=3.9 -y
conda activate recr1
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3 ray
cd Rec-R1 && pip install -e . --no-deps
pip install "tensordict<0.6" "transformers<4.48" hydra-core codetiming pybind11 accelerate dill pandas wandb
pip install flash-attn --no-build-isolation   # 或安装对应 wheel
pip install pyserini faiss-gpu
conda install -c conda-forge openjdk=17 -y
```

详细版本与踩坑记录见上游 [Rec-R1 README](Rec-R1/README.md)。

---

## 数据准备

首选 **Amazon ESCI**（真实购物 query + 商品元数据 + E/S/C/I 相关性标签）。

| 用途 | 路径 | 说明 |
|------|------|------|
| RL 训练/验证/测试 | `Rec-R1/data/esci/inst/sparse/subset/*.parquet` | 仓库已带 |
| 冒烟小子集 | `Rec-R1/data/esci/inst/sparse/subset_smoke/` | 本仓库新增 |
| BM25 商品语料 | `Rec-R1/database/esci/jsonl_docs/esci_metadata.jsonl` | 需本地构建 |
| Lucene 索引 | `Rec-R1/database/esci/pyserini_index/` | 需本地构建 |

国内无法访问 Google Drive 时，请用 HuggingFace 镜像重建语料，步骤见 [DATA.md](./DATA.md)。

```bash
source env.sh
cd Rec-R1
# 下载语料 → 转格式 → 建索引（详见 DATA.md）
python src/Lucene/esci/0_build_corpus_from_hf.py
bash src/Lucene/esci/2_build_database.sh
```

基座模型默认路径：`/data1/hcc/.hf_home/Qwen2.5-3B-Instruct`（可通过 `BASE_MODEL` 覆盖）。

---

## 训练

### 阶段 0：冒烟测试（验证全链路）

验证 rollout → BM25 reward → GRPO 更新是否正常（小子集 + 1 epoch）：

```bash
source env.sh
cd Rec-R1
export CUDA_VISIBLE_DEVICES=2,3
export N_GPUS=2
bash scripts/train/train-esci_3b_smoke.sh
```

日志：`Rec-R1/exp_log/esci-qwen3b-grpo-smoke-*.log`

### 阶段 1：全量 GRPO 训练

```bash
# 前台
source env.sh && cd Rec-R1 && bash scripts/train/train-esci_3b_full.sh

# 或后台
bash run_full_train.sh
tail -f full_train.log
```

-  checkpoint 目录：`Rec-R1/checkpoints/recr1-esci/esci-qwen3b-grpo-full/actor/global_step_*`
- 每 50 step 保存；本仓库 VERL 无自动 resume，中断后可用最新 checkpoint 作 `BASE_MODEL` 继续（优化器状态会丢失）

主要超参（`train-esci_3b_full.sh`）：`lr=1e-6`，`rollout.n=12`，`total_epochs=20`，KL 系数 `0.001`。

---

## 评测

**Step 1 — 用训练好的模型生成查询：**

```bash
bash scripts/eval/esci/inst_gen/sparse/rec-r1.sh Video_Games
```

**Step 2 — BM25 检索并计算 Recall@K / NDCG@K：**

```bash
bash scripts/eval_search/esci/sparse/eval_search.sh
```

结果目录：`Rec-R1/results/esci/`。

---

## 本仓库相对上游的改动

| 文件 | 说明 |
|------|------|
| `env.sh` | 统一 conda/Java/HF 缓存路径 |
| `DATA.md` | 国内 HF 镜像替代 GDrive |
| `run_full_train.sh` | 后台全量训练入口 |
| `Rec-R1/scripts/train/train-esci_3b_smoke.sh` | ESCI 冒烟 GRPO |
| `Rec-R1/scripts/train/train-esci_3b_full.sh` | ESCI 全量 GRPO + 定期存 checkpoint |
| `Rec-R1/src/Lucene/esci/0_build_corpus_from_hf.py` | 从 HF 语料重建 ESCI metadata |
| `Rec-R1/data/esci/inst/sparse/subset_smoke/` | 极小训练子集 |

---

## 扩展路线

1. **Amazon C4**：更大商品语料，脚本见 `scripts/train/train_rec-amazon_c4_3b.sh`
2. **Amazon Review / All Beauty**：序列推荐出口，数据见 `data/amazon_review/`
3. **Dense 检索**：见 `Rec-R1/src/Dense/` 与 `verl/utils/reward_score_dense/`

---

## 升级方向：Agentic Commerce-R1

当前仓库是 **Rec-R1 复现（已跑通）**。下一步升级为面向电商搜索的多工具 Agentic RL 检索决策系统，详见：

- [升级路线图](docs/UPGRADE_ROADMAP.md)
- [实施前待澄清问题](docs/QUESTIONS_BEFORE_UPGRADE.md)
- [参考论文索引](papers/README.md)（PDF 本地 `papers/`，不纳入 Git）

### Phase 1 已完成：CommerceAgentEnv smoke test

独立多步 BM25 环境（JSON action + process reward），**未改 GRPO 主链路**。

```bash
source env.sh
python scripts/smoke_agent_env.py --num-samples 20
```

结果见 [`experiments/phase1_env_smoke/`](experiments/phase1_env_smoke/README.md)。

### Phase 1.5 已完成：Qwen2.5-3B rollout smoke

LLM JSON policy + vLLM 驱动 CommerceAgentEnv，**不训练 GRPO**。

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/smoke_qwen_rollout.py --num-samples 10
```

结果见 [`experiments/phase15_qwen_rollout_smoke/`](experiments/phase15_qwen_rollout_smoke/README.md)（parse_success_rate=1.0，10 条样本）。

### Phase 1.6 已完成：Finish-Aware Rollout Fix

Prompt 增强 + 最后一步 auto-finalize + `llm/auto/finish_rate` 指标。

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/smoke_qwen_rollout.py --num-samples 10
```

结果见 [`experiments/phase16_finish_aware_smoke/`](experiments/phase16_finish_aware_smoke/README.md)（finish_rate=1.0，parse=1.0）。

### Phase 1.7 已完成：VERL Rollout Adapter 骨架

将 `CommerceAgentEnv + QwenRolloutPolicy` episode 包装为 VERL-like rollout record（`prompt/response/reward/trajectory/metrics/extra_info`）。**不训练 GRPO，未接入 VERL worker。**

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/smoke_verl_rollout_adapter.py \
  --num-samples 10 \
  --max-steps 3 \
  --model-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
  --output-dir experiments/phase17_verl_adapter_smoke_10
```

结果见 [`experiments/phase17_verl_adapter_smoke/`](experiments/phase17_verl_adapter_smoke/README.md)（10 条：finish_rate=1.0，invalid=0.0，num_rollout_records=10）。

### Phase 1.8 已完成：VERL Batch Mock / Shape Check

读取 Phase 1.7 `rollout_records.jsonl`，用 Qwen tokenizer 构造 mock batch（`input_ids` / `attention_mask` / `response_mask` / `rewards`）。**不训练 GRPO，不调用 vLLM/env/BM25。**

```bash
python scripts/smoke_verl_batch_mock.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --tokenizer-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
  --output-dir experiments/phase18_verl_batch_mock_10
```

结果见 [`experiments/phase18_verl_batch_mock/`](experiments/phase18_verl_batch_mock/README.md)（shape_check_passed=true）。

### Phase 1.9 已完成：VERL Training Fields Mock

在 Phase 1.8 batch 上补齐 `position_ids`、`prompts`、`responses`、`token_level_rewards` 及 mock logprob/advantage 占位。**不训练 GRPO，不重算真实 logprob。**

```bash
python scripts/smoke_verl_training_fields.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase19_verl_training_fields_mock_10
```

结果见 [`experiments/phase19_verl_training_fields_mock/`](experiments/phase19_verl_training_fields_mock/README.md)（shape_check_passed=true）。

### Phase 1.10 已完成：DataProto / Reward Function Dry-Run

将 Phase 1.9 training fields 映射为 `DataProtoMock`，运行 `CommerceRewardFn` dry-run 与 actor input field check。**不训练 GRPO，不调用 actor.forward。**

```bash
python scripts/smoke_dataproto_reward_dryrun.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase110_dataproto_reward_dryrun_10
```

结果见 [`experiments/phase110_dataproto_reward_dryrun/`](experiments/phase110_dataproto_reward_dryrun/README.md)（三项 check 均通过）。

### Phase 1.11 已完成：Real DataProto Compatibility Check

尝试将 `DataProtoMock` 转为真实 `verl.protocol.DataProto`；环境不支持时 graceful fallback。**不训练 GRPO。**

```bash
python scripts/smoke_real_dataproto_compat.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase111_real_dataproto_compat_10
```

结果见 [`experiments/phase111_real_dataproto_compat/`](experiments/phase111_real_dataproto_compat/README.md)。

### Phase 1.12 已完成：Actor LogProb Interface Mock

侦察 verl `compute_log_prob` 字段，构造 actor-logprob-ready request，生成 mock logprob 并做 shape check。**不调用 actor.forward，不训练。**

```bash
python scripts/smoke_actor_logprob_mock.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase112_actor_logprob_mock_10
```

结果见 [`experiments/phase112_actor_logprob_mock/`](experiments/phase112_actor_logprob_mock/README.md)（`is_mock=true`）。

### Phase 1.13 已完成：Real Actor LogProb Dry-Run

HuggingFace `AutoModelForCausalLM` + `torch.no_grad()` 计算真实 response logprob。**不训练、不接 GRPO。**

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/smoke_actor_logprob_dryrun.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase113_actor_logprob_dryrun_2 \
  --num-records 2
```

结果见 [`experiments/phase113_actor_logprob_dryrun/`](experiments/phase113_actor_logprob_dryrun/README.md)。

### Phase 1.14 已完成：Reference LogProb / KL Dry-Run

验证 actor/old/ref logprobs、token KL、ratio 与 mask 对齐。**不训练、不接 GRPO。**

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/smoke_ref_kl_dryrun.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase114_ref_kl_dryrun_2 \
  --num-records 2 --shared-ref
```

结果见 [`experiments/phase114_ref_kl_dryrun/`](experiments/phase114_ref_kl_dryrun/README.md)（shared-ref 下 KL≈0，ratio≈1）。

### Phase 1.15 已完成：GRPO Advantage Mock / Grouped Reward Dry-Run

Synthetic grouped rollout + 组内 reward 归一化 advantage。**不训练、不接 GRPO trainer。**

```bash
python scripts/smoke_grpo_advantage_mock.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase115_grpo_advantage_mock_10_g4 \
  --num-base-records 10 --group-size 4
```

结果见 [`experiments/phase115_grpo_advantage_mock/`](experiments/phase115_grpo_advantage_mock/README.md)。

### Phase 1.16 已完成：GRPO Loss Dry-Run

PPO/GRPO clipped policy loss + KL penalty 独立 dry-run（mock log_probs）。**不训练、不接 GRPO trainer。**

```bash
python scripts/smoke_grpo_loss_dryrun.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase116_grpo_loss_dryrun_10_g4 \
  --num-base-records 10 --group-size 4
```

结果见 [`experiments/phase116_grpo_loss_dryrun/`](experiments/phase116_grpo_loss_dryrun/README.md)。

---

## 引用

```bibtex
@article{lin2025rec,
  title={Rec-R1: Bridging Generative Large Language Models and User-Centric Recommendation Systems via Reinforcement Learning},
  author={Lin, Jiacheng and Wang, Tian and Qian, Kun},
  journal={arXiv preprint arXiv:2503.24289},
  year={2025}
}
```

## 致谢

- [Rec-R1 / linjc16](https://github.com/linjc16/Rec-R1)
- [VERL](https://github.com/volcengine/verl)
- [Pyserini](https://github.com/castorini/pyserini)
