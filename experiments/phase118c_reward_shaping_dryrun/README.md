# Phase 1.18c Reward Shaping Proposal Dry-Run

Offline evaluation of 6 candidate reward formulas on Phase 1.17 multi-sample rollouts.

No training, no official reward changes.

## Run

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

python scripts/dryrun_reward_shaping.py \
  --rollout-path experiments/phase117_multisample_rollout_5_g4/rollout_records.jsonl \
  --group-sensitivity-path experiments/phase118b_reward_sensitivity_5_g4/group_reward_sensitivity.jsonl \
  --query-metrics-path experiments/phase118b_reward_sensitivity_5_g4/query_metric_diagnostics.jsonl \
  --output-dir experiments/phase118c_reward_shaping_dryrun_5_g4
```

## Outputs

- `shaped_record_rewards.jsonl`
- `candidate_group_reports.jsonl`
- `candidate_comparison.json` / `.md`
- `reward_shaping_recommendations.md`
- `summary.json`
