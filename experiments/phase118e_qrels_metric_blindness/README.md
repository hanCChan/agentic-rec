# Phase 1.18e Qrels / Metric Blindness

Diagnostic analysis of ESCI qrels coverage and multi-K IR metric spread for Phase 1.18d strategy-controlled rollouts.

## Inputs

- `experiments/phase118d_strategy_rollout_5_g4/rollout_records.jsonl`
- `experiments/phase118d_strategy_rollout_5_g4/group_summaries.jsonl` (optional)
- `experiments/phase119a_strategy_reward_decomposition_5_g4/group_reward_source_report.jsonl` (optional)

## Run

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec
python scripts/analyze_qrels_metric_blindness.py \
  --rollout-path experiments/phase118d_strategy_rollout_5_g4/rollout_records.jsonl \
  --group-summary-path experiments/phase118d_strategy_rollout_5_g4/group_summaries.jsonl \
  --output-dir experiments/phase118e_qrels_metric_blindness_5_g4 \
  --k-list 10 50 100 1000
```

## Outputs

See `experiments/phase118e_qrels_metric_blindness_5_g4/` after running.
