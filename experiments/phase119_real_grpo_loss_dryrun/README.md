# Phase 1.19 Real GRPO Loss Dry-Run

Strategy-controlled real groups + `reward_largek_mix_1000` quality-only advantages + GRPO clipped loss dry-run.

## Run

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec
python scripts/smoke_real_grpo_loss_dryrun.py \
  --rollout-path experiments/phase118d_strategy_rollout_5_g4/rollout_records.jsonl \
  --shaped-reward-path experiments/phase118f_large_k_reward_dryrun_5_g4/large_k_shaped_record_rewards.jsonl \
  --candidate-name reward_largek_mix_1000 \
  --tokenizer-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
  --output-dir experiments/phase119_real_grpo_loss_dryrun_5_g4_largek1000
```

## Boundaries

- No training / no optimizer.step
- Penalties do NOT enter GRPO advantage
- Does NOT re-rollout
