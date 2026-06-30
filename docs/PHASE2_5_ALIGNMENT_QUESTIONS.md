# Phase 2.5 Alignment Questions

> 本文档记录 Phase 2.4 完成后的口径对齐问题、已知缺陷与 Phase 2.5 开放决策。
> 配套对外表述见 [EXTERNAL_CLAIMS_GUIDE.md](./EXTERNAL_CLAIMS_GUIDE.md)。

**项目定位（统一采用）：**

```text
Rec-R1 范式延伸 + Agentic Search 工程化 + pilot 级 GRPO 训练验证
```

**禁止定位：**

```text
完整 Rec-R1 benchmark 复现并超越
```

---

## 1. Fixed-batch training reward vs fresh rollout eval reward

### 事实

Phase 2.4 训练使用 **固定 preflight rollout batch**（80 records = 20 groups × 4 strategies）。
`pilot_train_metrics.jsonl` 中 `mean_reward` **50 步均为 0.3726**，不随 step 变化。

Fresh rollout eval（checkpoint 加载后重新生成 trajectory）结果：

| Eval step | mean_reward_largek_mix_1000 |
|-----------|----------------------------|
| 0 | 0.373 |
| 10 | 0.385 |
| 25 | 0.390 |
| 50 | 0.395 |

相对 step-0 约 **+5.9%**。

### 对外表述（采用）

```text
Fresh rollout eval reward improved from 0.373 to 0.395 on the curated clean 20-group pilot set.
Fixed preflight in-batch training reward remained flat; the +5.9% signal comes from fresh eval, not the training reward curve.
```

### 禁止表述

```text
Training reward improved by 5.9%.
GRPO online training reward increased over 50 steps.
```

### 原因

GRPO 完整 online loop 应为：generate → reward → advantage → update → **re-rollout**。
当前 Phase 2.1–2.4 训练 batch 固定，in-batch reward 主要用于 loss/KL 稳定性验证；
策略行为变化需通过 **fresh rollout eval** 观测。

---

## 2. VERL-compatible vs full VERL trainer

### 事实

已验证：

```text
real verl.DataProto
verl compute_advantage (group-relative)
mini-batch / micro-batch / token-mean loss aggregation
GRPO clipped policy loss + KL in actor loss
```

当前训练实现：

```text
TinyGrpoSmokeTrainer / ControlledGrpoSmokeTrainer
HF AutoModelForCausalLM + hand-written optimizer.step
NOT RayPPOTrainer / Ray worker / vLLM rollout-in-loop
```

### 对外表述（采用）

```text
VERL-compatible DataProto / GRPO advantage path validated.
Training uses a custom pilot trainer with HF actor; full VERL Ray trainer integration is Phase 3+.
```

### 禁止表述

```text
Full VERL distributed GRPO trainer completed.
Identical to Rec-R1 official training stack.
```

---

## 3. Strategy distribution monitoring bug

### 事实

Phase 2.4 四次 fresh eval 的 `strategy_distribution` 均为：

```json
{"unknown": 1.0}
```

原因：`run_50step_grpo_pilot.py` 中 `run_fresh_eval()` 读取 `record.get("strategy")` /
`record.get("search_strategy")`，与 rollout record 实际字段不一致。

### 对外表述（采用）

```text
Strategy-conditioned rollout is used to construct GRPO groups.
Strategy distribution drift monitoring is planned but not yet validated in Phase 2.4 eval (field mapping bug).
```

### 禁止表述

```text
We verified broad_recall / exact_match strategy ratio shifts after training.
Strategy drift monitoring passed.
```

### 优先级

**P0** — Phase 2.5 前修复 eval 字段映射。

---

## 4. Agentic rollout vs fixed-batch GRPO

### 事实

已实现：

```text
multi-step agentic search environment
strategy-conditioned BM25 tool-use rollout
JSON action parsing + finish-aware episodes
```

Phase 2 pilot 训练：

```text
fixed preflight rollout batch for GRPO update
NOT per-step online re-rollout
```

Eval 时 fresh rollout 能反映 checkpoint 行为变化，但训练 loop 本身不是 online closed-loop。

### 对外表述（采用）

```text
Agentic multi-step search rollout capability is implemented.
Phase 2 pilot uses fixed-batch GRPO for stability; online re-rollout training is Phase 2.5+/Phase 3.
```

---

## 5. Clean 20-group curation bias

### 事实

`phase2_clean_20_groups.jsonl` 经过：

```text
exclude esci_val_3 (unlearnable)
drop esci_val_6 (seed=42 collapse)
replacement esci_val_52, esci_val_57
Phase 1.18g BM25 failure cleanup
Phase 1.18h strategy V2 gate
large-K reward spread gate
```

### 对外表述（采用）

```text
Phase 2.4 +5.9% is a heavily curated in-domain pilot signal on 20 clean groups.
It must not be extrapolated to ESCI full validation, Rec-R1 benchmark settings, or production search.
```

---

## 6. Rec-R1 positioning

### 采用

```text
Rec-R1-style closed-loop retrieval RL extension with agentic search engineering.
```

### 禁止

```text
Rec-R1 reproduction and outperformance.
We beat Rec-R1 on official benchmarks.
```

Rec-R1 论文在 product search / sequential recommendation 上做完整 benchmark 与 baseline 对比；
本项目尚未做官方 baseline 对齐与大规模实验。

---

## 7. Known issues (Phase 2.4)

| Issue | Priority | Status | Action |
|-------|----------|--------|--------|
| `strategy_distribution=unknown` | P0 | Open | Fix eval field mapping before Phase 2.5 |
| Eval step 10/25/50 runs **after** all 50 training steps, not mid-training | P0 | Open | Periodic eval hook in Phase 2.5 |
| `checkpoint_path` in summary still says `smoke_step_50` | P1 | Open | Rename to `pilot_step_50` in summary |
| `invalid_action_rate=0.0` false-positive stop | P0 | **Fixed** | `grpo_pilot_monitor.py` commit `286b12c` |
| Local checkpoints deleted in cleanup (~70GB) | P1 | Done | Define retention policy for Phase 2.5 |
| Phase 1 experiment dirs archived to RESULTS_INDEX only | Doc | Done | See `docs/RESULTS_INDEX.md` |
| GPU plan 0–3 vs actual 1–4 (GPU 0 occupied by vLLM) | Doc | Done | Hardware constraint, not config change |

---

## 8. Phase 2.5 open questions (with recommendations)

### Q1. 扩组方式

**问题：** 从旧 candidate pool 扩，还是重新扫 ESCI val？

**建议：** 重新扫描 ESCI val，沿用 Phase 1.18g learnability gate：

```text
ESCI val scan
→ BM25 top1000 coverage
→ qrels sparsity check
→ strategy V2 rollout
→ reward_largek_mix_1000 spread gate
→ select 50–100 clean train groups
```

不要只从 Phase 1.18h candidate pool 扩（已有历史筛选偏差）。

### Q2. Held-out 划分

**问题：** 从 20 组拆 15/5，还是新增 held-out？

**建议：** 新增 held-out，never train：

```text
train_clean: 50 groups (or 40 if resource-limited)
heldout_clean: 20 groups (or 10)
```

20 组拆 15/5 统计意义太弱。

### Q3. 200-step 是否仍 fixed preflight batch？

**问题：** 继续 fixed batch 会否 overfit？

**建议：**

```text
train: fixed clean train batch (control variable)
eval: periodic fresh rollout every 25/50 steps on train + heldout
Phase 3: online rollout → reward → update loop
```

### Q4. Baseline 优先级

**建议顺序：**

```text
1. Prompt-only Qwen strategy V2 (no training)
2. Rec-R1-style single-shot query rewrite GRPO (no multi-step agent)
3. SFT JSON/tool-use format baseline
4. reward ablation / strategy ablation
```

最有说服力对比：**Agentic strategy GRPO vs single-shot Rec-R1-style GRPO**。

### Q5. +5.9% 统计意义

**问题：** 20 groups × 4 records 是否足够写 formal conclusion？

**建议：**

```text
bootstrap confidence interval over groups
≥3 random seeds
report mean ± std
```

资源受限时至少 bootstrap over heldout groups；否则只能称 early signal。

---

## 9. Scientific claim levels

```text
L1 Engineering: controllable pilot runs, monitored, rollback-safe ✅
L2 Signal: fresh eval early positive trend on curated clean 20 groups ✅
L3 Scientific result: requires Phase 2.5 (50–100 groups, held-out, baselines, seeds) ❌
```

---

## 10. Phase 2.5 recommended sequence

```text
Phase 2.5a: 本文档 + EXTERNAL_CLAIMS_GUIDE ✅
Phase 2.5b: Fix strategy_distribution eval bug
Phase 2.5c: Expand clean set (50 train + 20 heldout) from ESCI val rescan
Phase 2.5d: Write 200-step pilot plan (fixed train batch + periodic fresh eval)
Phase 2.5e: Prompt-only baseline
Phase 2.5f: 200-step pilot (do not auto-run without plan approval)
```
