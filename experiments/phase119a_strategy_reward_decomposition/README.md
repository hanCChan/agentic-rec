# Phase 1.19a Strategy Rollout Reward Decomposition

Decomposes Phase 1.18d strategy-controlled rollout rewards into retrieval quality vs penalty components.

No training, no reward changes, no GRPO loss.

## Run

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

python scripts/analyze_strategy_reward_decomposition.py \
  --rollout-path experiments/phase118d_strategy_rollout_5_g4/rollout_records.jsonl \
  --group-summary-path experiments/phase118d_strategy_rollout_5_g4/group_summaries.jsonl \
  --output-dir experiments/phase119a_strategy_reward_decomposition_5_g4
```

## Gate to Phase 1.19

Proceed to GRPO loss dry-run only if:

- `retrieval_quality_spread_group_rate >= 0.6`
- `penalty_only_spread_group_rate <= 0.2`
