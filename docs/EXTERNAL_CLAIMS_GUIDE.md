# External Claims Guide

> 对外表述规范：简历、面试、报告、答辩、GitHub README 统一口径。
> 技术细节与开放问题见 [PHASE2_5_ALIGNMENT_QUESTIONS.md](./PHASE2_5_ALIGNMENT_QUESTIONS.md)。

**一句话定位：**

```text
Rec-R1 范式延伸 + Agentic Search 工程化 + pilot 级 GRPO 训练验证
```

---

## Can say（可以说）

- Completed a **Rec-R1-style closed-loop retrieval RL** engineering prototype for agentic e-commerce search.
- Completed staged **1 / 3 / 10 / 50-step** GRPO smoke and pilot validation on a curated clean set.
- **50-step pilot** was stable: `max_approx_kl=0.049`, `max_grad_norm=0.422`, `parse_success_rate=1.0`, no NaN/OOM/early stop.
- **Fresh rollout eval reward** improved from **0.373 to 0.395** (~+5.9%) on the curated clean **20-group** pilot set.
- Fixed preflight **in-batch training reward** stayed flat; the positive signal comes from **fresh eval**, not the training reward curve.
- The project **extends** Rec-R1-style RL with: agentic tool-use, strategy-controlled rollout, large-K retrieval reward (`reward_largek_mix_1000`), penalty/quality decoupling, and pilot-grade monitoring (KL, grad, JSON, checkpoint guard).
- **VERL-compatible** DataProto / GRPO advantage path is validated; training uses a custom pilot trainer (Phase 3+ targets full VERL Ray integration).
- Engineering completeness exceeds a typical "GRPO demo that runs one step without crashing."

---

## Cannot say（不能说）

- **Outperforms Rec-R1** or beats Rec-R1 official benchmarks.
- Completed **official Rec-R1 benchmark reproduction**.
- **Proves generalization** on ESCI full validation or production search.
- **Training reward improved 5.9%** (misleading — improvement is from fresh eval).
- **Strategy distribution drift** has been validated (Phase 2.4 eval bug: all `unknown`).
- **Full VERL distributed trainer** has been completed.
- Smoke/pilot checkpoints are production-ready models (they are `SMOKE_ONLY_DO_NOT_PROMOTE`).

---

## Scientific claim levels

| Level | Claim | Status |
|-------|-------|--------|
| **L1 Engineering** | Controllable pilot training runs, monitored, rollback-safe | ✅ Established |
| **L2 Signal** | Fresh eval shows early positive trend on curated clean 20 groups | ✅ Observed |
| **L3 Scientific result** | Generalization, benchmark superiority, formal effect size | ❌ Not yet — needs Phase 2.5 |

Phase 2.4 establishes **L1 + L2**. It does **not** establish L3.

---

## Phase 2.4 key numbers (for accurate citation)

```text
actual_update_steps     = 50/50
pilot_passed            = true
max_approx_kl_nonnegative = 0.049  (< 0.2 threshold)
max_grad_norm           = 0.422   (< 10 threshold)
parse_success_rate      = 1.0     (all fresh evals)
checkpoint_promoted     = false
checkpoint_label        = SMOKE_ONLY_DO_NOT_PROMOTE

Fresh eval mean_reward_largek_mix_1000:
  step 0  = 0.373
  step 10 = 0.385
  step 25 = 0.390
  step 50 = 0.395  (~+5.9% vs step 0)

In-batch training mean_reward (fixed preflight):
  all steps ≈ 0.373  (flat)
```

---

## Recommended interview wording（中文面试表述）

```text
我做的不是简单复现 Rec-R1，而是在 Rec-R1 的 closed-loop retrieval RL 思路上做了 Agentic Search 扩展。
模型不是一次性生成 query，而是通过 strategy-conditioned tool-use 进行多步 BM25 搜索，
再用 large-K retrieval metric（NDCG@1000 + Recall@1000 + MRR@1000）构造 quality-only GRPO reward，
并且把 process penalty 从 advantage 里解耦出去。

工程上我完成了从 rollout、reward、advantage、DataProto、loss 到 1/3/10/50-step pilot 的分阶段验证。
50-step pilot 在 KL、grad、JSON 格式和 checkpoint 安全上都稳定；
fresh rollout eval 在 curated clean 20 groups 上有从 0.373 到 0.395 的轻微正向信号。

但我不会声称超过 Rec-R1，因为：
(1) 训练 batch 仍是 fixed preflight，不是 full online re-rollout；
(2) 样本是 heavily curated 的 20 groups，不能外推泛化；
(3) 还需要更大 clean set、held-out eval、prompting baseline 和 ablation。
```

---

## Paper / report wording（英文摘要级）

```text
We present an agentic commerce-search RL prototype that extends Rec-R1-style
closed-loop retrieval optimization with multi-step tool-use, strategy-controlled
group rollout, and large-K quality rewards. Using a VERL-compatible DataProto
pipeline and staged GRPO pilot training (1/3/10/50 steps), we demonstrate
engineering stability and an early positive fresh-eval trend (+5.9% reward on a
curated 20-group pilot set). We do not claim benchmark superiority over Rec-R1;
formal conclusions require scaled clean sets, held-out evaluation, baselines, and
multi-seed analysis (Phase 2.5).
```

---

## README / GitHub one-liner

```text
Rec-R1-style agentic search RL prototype: strategy tool-use + large-K GRPO reward +
staged 1→50-step pilot validation. Early fresh-eval signal on curated clean set;
not a full Rec-R1 benchmark reproduction.
```

---

## Related documents

| Document | Purpose |
|----------|---------|
| [RESULTS_INDEX.md](./RESULTS_INDEX.md) | Commit hash + metrics evidence chain |
| [PHASE2_4_50STEP_PILOT_PLAN.md](./PHASE2_4_50STEP_PILOT_PLAN.md) | Phase 2.4 experiment protocol |
| [PHASE2_5_ALIGNMENT_QUESTIONS.md](./PHASE2_5_ALIGNMENT_QUESTIONS.md) | Open questions + known issues |
| [ARTIFACT_CLEANUP_MANIFEST.md](./ARTIFACT_CLEANUP_MANIFEST.md) | What was kept/deleted locally |
| [UPGRADE_ROADMAP.md](./UPGRADE_ROADMAP.md) | Phase timeline |

---

## Claim boundary (current)

```text
Phase 2.4 establishes pilot-level engineering stability and early fresh-eval signal
on a curated clean 20-group set. It does NOT establish full benchmark performance
or outperformance over Rec-R1.
```
