# Phase 1.18b Reward Sensitivity Diagnostics

Analyzes why multi-sample GRPO groups lack reward spread (NDCG vs Recall/MRR vs penalties vs topK overlap).

No training, no reward formula changes.

## Run

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

python scripts/analyze_reward_sensitivity.py \
  --rollout-path experiments/phase117_multisample_rollout_5_g4/rollout_records.jsonl \
  --group-diagnostics-path experiments/phase118a_rollout_diagnostics_5_g4/group_diagnostics.jsonl \
  --output-dir experiments/phase118b_reward_sensitivity_5_g4 \
  --topk-list 10 50 100
```

## Outputs

- `query_metric_diagnostics.jsonl`
- `group_reward_sensitivity.jsonl`
- `reward_sensitivity_summary.json`
- `reward_recommendations.md`
- `summary.json`
