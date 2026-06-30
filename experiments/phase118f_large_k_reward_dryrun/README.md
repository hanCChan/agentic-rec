# Phase 1.18f Large-K Reward Candidate Dry-Run

Offline evaluation of global large-K retrieval-quality reward candidates using Phase 1.18d rollouts and Phase 1.18e metric-by-K diagnostics.

## Run

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec
python scripts/dryrun_large_k_reward.py \
  --rollout-path experiments/phase118d_strategy_rollout_5_g4/rollout_records.jsonl \
  --metric-by-k-path experiments/phase118e_qrels_metric_blindness_5_g4/metric_by_k_diagnostics.jsonl \
  --group-metric-spread-path experiments/phase118e_qrels_metric_blindness_5_g4/group_metric_spread_by_k.jsonl \
  --decomposition-path experiments/phase119a_strategy_reward_decomposition_5_g4/group_reward_source_report.jsonl \
  --output-dir experiments/phase118f_large_k_reward_dryrun_5_g4
```

## Gate (non-diagnostic candidates)

- `retrieval_quality_spread_group_rate >= 0.6`
- `penalty_only_spread_group_rate <= 0.2`
- `zero_std_group_rate <= 0.5`
