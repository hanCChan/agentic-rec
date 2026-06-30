# Phase 2.4：50-Step Pilot GRPO

> **当前阶段：2.4a — 计划文档已就绪，尚未跑训练**

## 执行顺序

```text
Phase 2.4a ✅ 计划文档 docs/PHASE2_4_50STEP_PILOT_PLAN.md
Phase 2.4b    dry-config check（不训练）
Phase 2.4     50-step pilot（4 GPU，lr=5e-7）
```

## Phase 2.4b：Dry Config Check

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

python scripts/run_50step_grpo_pilot.py \
  --dry-config-check \
  --output-dir experiments/phase24_50step_grpo_pilot
```

## Phase 2.4：Pilot Training（待执行）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_50step_grpo_pilot.py \
  --output-dir experiments/phase24_50step_grpo_pilot/lr_5e-7 \
  --learning-rate 5e-7 \
  --max-update-steps 50 \
  --save-steps 10 25 50 \
  --eval-steps 0 10 25 50
```

## 关键约束

- checkpoint 全部 `SMOKE_ONLY_DO_NOT_PROMOTE`
- 训练用固定 preflight batch；eval 用 fresh rollout
- hard stop：`approx_kl_nonnegative > 0.2`，fresh eval reward 下降 > 30%
- 只改变量：`update_steps 10→50`，`lr=5e-7`

## 前置结果

Phase 2.3 已通过（commit `5475563`），lr=5e-7 max_approx_kl=0.0046。
