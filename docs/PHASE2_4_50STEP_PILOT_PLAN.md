# Phase 2.4：50-Step Pilot GRPO Training Plan

> **状态：Phase 2.4a — 计划文档（不跑训练）**  
> 前置：Phase 2.3 10-step controlled smoke 已通过（lr=1e-6 与 5e-7，`both_stable=true`）

## 1. 实验目标

Phase 2.4 是**第一个真正 pilot**，不是扩大版 10-step smoke。目标不是追求最终 NDCG 提升，而是回答 6 个问题：

1. **50 step 内 KL 是否仍然可控？**（`approx_kl_nonnegative ≤ 0.2`）
2. **JSON / tool-use 格式是否仍稳定？**（`parse_success_rate ≥ 0.95`）
3. **`reward_largek_mix_1000` 是否没有崩？**（fresh eval 下降 ≤ 30%）
4. **fresh rollout eval 是否有方向性变化？**（行为真的在变，而非 stale batch 过拟合）
5. **strategy 分布（broad_recall / exact_match 等）是否异常偏移？**
6. **smoke checkpoint 是否可保存、可回滚、不可 promote？**

### Rec-R1 对齐

Rec-R1 的核心是用**固定黑盒推荐/检索系统反馈**做闭环 RL 优化，不依赖 GPT-4o 蒸馏数据。本 pilot 沿用：

- 固定 BM25 检索器
- 固定 qrels 指标
- `reward_largek_mix_1000` 作为 outcome reward
- penalty **不进** advantage
- 无 oracle / overlap diagnostic reward

### verl / GRPO 对齐

训练侧沿用 Phase 2.1–2.3 已验证链路：

- `adv_estimator=grpo`（group-relative advantage）
- KL 放在 actor loss（`kl_coef=0.01`）
- `loss_agg_mode=token-mean`
- `cliprange=0.2`
- mini-batch / micro-batch 与 Phase 2.3 相同

---

## 2. 不做什么（硬性边界）

```text
❌ 不跑 full dataset training
❌ 不直接 200-step
❌ 不 promote checkpoint（全部 SMOKE_ONLY_DO_NOT_PROMOTE）
❌ 不覆盖 base model / Qwen2.5-3B-Instruct
❌ 不回 Phase 1 修 esci_val_3
❌ 不改 reward 公式
❌ 不把 penalty 放回 advantage
❌ 不用 oracle / overlap reward
❌ 不一边改 reward 一边改训练配置
❌ 不在 Phase 2.4 同时做 LR ablation（1e-6 留作后续对照）
```

**Phase 2.4 只改一个变量：**

```text
update_steps: 10 → 50
learning_rate: 固定 5e-7（Phase 2.3 LR sweep 推荐）
其余保持 Phase 2.3 稳定配置
```

---

## 3. 输入数据

### Clean set（复用，不重建）

| 路径 | 说明 |
|------|------|
| `experiments/phase21_tiny_grpo_smoke/phase2_clean_20_groups.jsonl` | 20 groups，已排除 `esci_val_3`，replacement `esci_val_52/57` |
| `experiments/phase21_tiny_grpo_smoke/phase2_clean_set_summary.json` | `phase2_clean_set_ready=true` 门禁 |
| `experiments/phase21_tiny_grpo_smoke/preflight_v2_rollout_20_g4/` | 固定 preflight rollout（训练 batch 数据源） |

### 训练 rollout 策略

```text
训练：固定 clean 20_g4 preflight rollout（80 records = 20×4）
      保证可控、可复现，避免 pilot 初期变量爆炸

评估：每 10 step 做 fresh rollout eval（同一 clean 20_g4，当前 checkpoint 重新生成）
      判断模型行为是否真的变了，避免 stale batch 过拟合假象
```

**不在 Phase 2.4 做 online re-rollout 训练**（那是 Phase 2.5+ 的事）。

---

## 4. 训练配置（默认主实验）

```yaml
phase: "2.4"
mode: "50step_grpo_pilot"

# 与 Phase 2.3 相同，仅 steps 和 lr 变化
max_update_steps: 50
learning_rate: 5.0e-7
kl_coef: 0.01
cliprange: 0.2
train_batch_size: 20
rollout_n: 4
ppo_mini_batch_size: 20
micro_batch_size: 4
max_prompt_length: 1024
max_response_length: 2048
max_total_length: 3072
loss_agg_mode: token-mean
max_grad_norm: 1.0

reward_candidate: reward_largek_mix_1000
penalties_in_advantage: false
diagnostic_oracle_reward_used: false

checkpoint_steps: [10, 25, 50]
eval_steps: [0, 10, 25, 50]
checkpoint_label: SMOKE_ONLY_DO_NOT_PROMOTE
checkpoint_promoted: false

model_path: /data1/hcc/.hf_home/Qwen2.5-3B-Instruct
tokenizer_path: /data1/hcc/.hf_home/Qwen2.5-3B-Instruct
data_path: Rec-R1/data/esci/inst/sparse/subset/val.parquet
```

### LR 选择理由

| LR | Phase 2.3 max_approx_kl | 结论 |
|----|-------------------------|------|
| 1e-6 | 0.028 | 稳定但 mild_drift |
| **5e-7** | **0.0046** | **更低 KL，适合第一个 50-step pilot** |

Phase 2.4 默认 **lr=5e-7**。1e-6 留作 Phase 2.5+ 对照。

---

## 5. GPU 分配（4 卡）

用户分配 **4×GPU**。Phase 2.4 沿用 Phase 2.3 单卡 actor 训练路径，4 卡用于隔离训练/eval 与预留 headroom：

```text
CUDA_VISIBLE_DEVICES=0,1,2,3

GPU 0 — Actor GRPO 训练（optimizer.step，与 Phase 2.3 相同 device=cuda:0）
GPU 1 — Fresh rollout eval（step 0/10/25/50 时加载 checkpoint 做 v2 rollout）
GPU 2 — BM25 / pyserini index（eval 期间检索，与 Phase 1 rollout 一致）
GPU 3 — 预留（OOM 回退 / 并行 pre-eval warmup，Phase 2.4 默认 idle）
```

运行命令（Phase 2.4 正式跑时）：

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_50step_grpo_pilot.py \
  --output-dir experiments/phase24_50step_grpo_pilot/lr_5e-7 \
  --learning-rate 5e-7 \
  --max-update-steps 50 \
  --save-steps 10 25 50 \
  --eval-steps 0 10 25 50 \
  --train-gpu 0 \
  --eval-gpu 1
```

磁盘预估（Qwen2.5-3B bf16 × 3 checkpoints）：

```text
~18 GB checkpoints + ~2 GB metrics/logs/eval ≈ 25 GB
建议 output_dir 所在分区剩余 ≥ 40 GB
```

---

## 6. Reward 定义

与 Phase 1.19 / 2.1–2.3 完全一致，**不改公式**：

```text
reward = reward_largek_mix_1000
  = mix(NDCG@1000, Recall@1000, MRR@1000)  # large-K quality signal

advantage = GRPO group-relative normalization on reward only
penalties (format, invalid action, etc.) → diagnostic only, NOT in advantage
```

Baseline reward（step-0 fresh eval）：Phase 2.3 lr=5e-7 post-eval ≈ **0.389**（用于 30% collapse 检测）。

---

## 7. KL / Grad / JSON Hard Stop

沿用 Phase 2.2 修复：**hard stop 用 `approx_kl_nonnegative`，不用 signed gap**。

每步记录：

```text
signed_logprob_gap_mean = actor_logprob_mean - ref_logprob_mean
signed_logprob_gap_abs_mean
approx_kl_nonnegative = E[exp(r) - 1 - r]
ref_logprob_mean
actor_logprob_mean
```

### 训练 step hard stop

| 条件 | 动作 |
|------|------|
| `nan_detected` | 立即停止 |
| `loss` NaN/Inf | 立即停止 |
| `grad_norm` NaN/Inf | 立即停止 |
| `grad_norm > 10.0` | 立即停止 |
| `approx_kl_nonnegative > 0.2` | 立即停止 |
| `json_format_ok == false` | 立即停止 |
| `checkpoint` 保存失败 | 立即停止 |
| `abs(signed_logprob_gap_mean) > 5.0` | **warn only**（除非 approx_kl 也高） |

### Fresh eval hard stop（eval_steps 触发）

| 条件 | 动作 |
|------|------|
| `parse_success_rate < 0.95` | 立即停止 |
| `invalid_action_rate > 0.05` | 立即停止 |
| `json_format_ok == false` | 立即停止 |
| `mean_reward` 相对 baseline 下降 **> 30%** | 立即停止 |
| strategy 分布偏移 > 50%（任一 strategy 占比从 >10% 变 <5% 或反向） | warn + 记录，不自动 stop（人工审查） |

---

## 8. Checkpoint 规则

### 保存策略

```text
checkpoints/pilot_step_10/   ← SMOKE_ONLY_DO_NOT_PROMOTE
checkpoints/pilot_step_25/   ← SMOKE_ONLY_DO_NOT_PROMOTE
checkpoints/pilot_step_50/   ← SMOKE_ONLY_DO_NOT_PROMOTE
```

不需要每步保存。磁盘不足时：**只保存 step 50**，manifest 中说明。

### checkpoint_manifest.json

```json
{
  "checkpoint_promoted": false,
  "checkpoint_label": "SMOKE_ONLY_DO_NOT_PROMOTE",
  "save_steps": [10, 25, 50],
  "checkpoints": [
    {"step": 10, "path": "checkpoints/pilot_step_10", "label": "SMOKE_ONLY_DO_NOT_PROMOTE"},
    {"step": 25, "path": "checkpoints/pilot_step_25", "label": "SMOKE_ONLY_DO_NOT_PROMOTE"},
    {"step": 50, "path": "checkpoints/pilot_step_50", "label": "SMOKE_ONLY_DO_NOT_PROMOTE"}
  ]
}
```

### 回滚

任意 checkpoint 可加载做 fresh eval 或从 step N 继续（Phase 2.4 默认不 resume，只验证可加载）。

---

## 9. Eval Protocol

### Eval 时间点

| Step | 类型 | 目的 |
|------|------|------|
| 0 | baseline fresh eval | base model 行为基线（训练前） |
| 10 | fresh eval | 早期行为漂移检测 |
| 25 | fresh eval | 中期稳定性 |
| 50 | fresh eval | pilot 最终行为 |

### 每次 eval 输出

```text
parse_success_rate
finish_rate
invalid_action_rate
json_format_ok
zero_std_group_rate
retrieval_quality_spread_group_rate
mean_reward_largek_mix_1000
mean_ndcg1000
mean_recall1000
mean_mrr1000
strategy_distribution  # exact_match / attribute_expansion / broad_recall / constraint_preserving
```

### 对比基线

| 阶段 | mean_reward | parse |
|------|-------------|-------|
| Phase 2.1 (1-step) | 0.391 | 1.0 |
| Phase 2.2 (3-step) | 0.396 | 1.0 |
| Phase 2.3 (10-step, lr=5e-7) | 0.389 | 1.0 |
| Phase 2.4 step-0 | TBD | TBD |
| Phase 2.4 step-50 | TBD | TBD |

---

## 10. 输出目录结构

```text
experiments/phase24_50step_grpo_pilot/
├── README.md
├── pilot_config.yaml                    # 默认配置快照
├── lr_5e-7/
│   ├── pilot_train_config.yaml
│   ├── pilot_train_metrics.jsonl        # 50 行，每 step 一行
│   ├── pilot_train_summary.json
│   ├── checkpoint_manifest.json
│   ├── curve_analysis.json
│   ├── pilot_report.md
│   ├── eval_step_0/post_eval_summary.json
│   ├── eval_step_10/post_eval_summary.json
│   ├── eval_step_25/post_eval_summary.json
│   ├── eval_step_50/post_eval_summary.json
│   └── checkpoints/
│       ├── pilot_step_10/
│       ├── pilot_step_25/
│       └── pilot_step_50/
└── dry_config_check.json                # Phase 2.4b 产物
```

---

## 11. 成功标准（pilot_passed）

```text
actual_update_steps = 50
optimizer_steps_called = 50
nan_detected = false
oom_detected = false
kl_exploded = false
loss_finite_all_steps = true
grad_norm_finite_all_steps = true
max_approx_kl_nonnegative ≤ 0.2
max_grad_norm ≤ 10.0
parse_success_rate ≥ 0.95（step-50 fresh eval）
invalid_action_rate ≤ 0.05
json_format_ok = true
mean_reward 下降 ≤ 30%（相对 step-0 baseline）
checkpoint_saved = true
checkpoint_promoted = false
pilot_passed = true
```

**不要求：**

```text
不要求 reward 提升
不要求 NDCG 提升
不要求 safe_for_full_training = true
不要求 checkpoint 可用于正式推理
```

---

## 12. 失败处理

### 情况 C：KL 或 JSON 崩

```text
→ Phase 2.4c：Training Stability Fix
  lr = 2e-7
  kl_coef = 0.02
  max_response_length 缩短
  更强 format preservation
```

### 情况 D：训练 reward 升但 fresh eval 降

```text
→ stale batch 过拟合
  不在 Phase 2.4 继续训
  Phase 2.5 改 periodic re-rollout
  扩大 eval set / held-out groups
```

---

## 13. Phase 2.5 决策规则

### 情况 A：50-step 稳定，fresh eval 不崩

```text
Phase 2.5：200-step pilot
先扩 clean groups：20 → 50 或 100
```

### 情况 B：50-step 稳定，reward 没变化

```text
工程稳定但策略未学
→ reward / strategy ablation
→ larger clean set
→ fresh rollout 数据重采样
```

---

## 14. 执行顺序

```text
Phase 2.4a ✅ 写本计划文档（当前）
    ↓
Phase 2.4b    dry-config check（scripts/run_50step_grpo_pilot.py --dry-config-check）
              验证：clean set、preflight、磁盘、GPU、config、eval hook
    ↓
Phase 2.4     50-step controlled pilot training（4 GPU，lr=5e-7）
    ↓
Phase 2.5     200-step pilot 或 clean set 扩到 50/100 groups
    ↓
主实验矩阵   reward ablation / strategy ablation / K ablation / LR ablation
```

### Phase 2.4b dry-config check 命令

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

python scripts/run_50step_grpo_pilot.py \
  --dry-config-check \
  --output-dir experiments/phase24_50step_grpo_pilot
```

### Phase 2.4 正式训练命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_50step_grpo_pilot.py \
  --output-dir experiments/phase24_50step_grpo_pilot/lr_5e-7 \
  --learning-rate 5e-7 \
  --max-update-steps 50 \
  --save-steps 10 25 50 \
  --eval-steps 0 10 25 50
```

---

## 15. 代码清单

| 文件 | 用途 |
|------|------|
| `docs/PHASE2_4_50STEP_PILOT_PLAN.md` | 本计划 |
| `src/agents/grpo_pilot_monitor.py` | Pilot 监控（训练 + eval hard stop） |
| `scripts/run_50step_grpo_pilot.py` | dry-config check + pilot 训练入口 |
| `experiments/phase24_50step_grpo_pilot/` | 实验产物目录 |

复用（不大改）：

- `src/agents/tiny_grpo_smoke_trainer.py`
- `src/agents/controlled_grpo_smoke_trainer.py`
- `src/agents/grpo_stability_monitor.py`
- `src/agents/grpo_curve_analyzer.py`

---

## 16. 与 Phase 2.3 的核心差异

| 维度 | Phase 2.3 | Phase 2.4 |
|------|-----------|-----------|
| steps | 10 | 50 |
| 默认 lr | 1e-6 / 5e-7 sweep | **5e-7 固定** |
| eval | 仅 post-train 一次 | **step 0/10/25/50 fresh eval** |
| 目标 | 工程稳定性 | **行为是否变化 + 稳定性** |
| GPU | 1 卡 | **4 卡（train/eval 隔离）** |
| checkpoint | smoke_step_* | pilot_step_* |
| 性质 | smoke | **第一个 pilot** |

---

**一句话：Phase 2.4 用 lr=5e-7、固定 preflight batch 训练 50 step，每 10 step fresh eval，验证 KL/格式/reward 不崩且行为有方向性变化；checkpoint 全部 SMOKE_ONLY_DO_NOT_PROMOTE。**
