# Phase 1.18a Rollout Diversity / Reward Variance Diagnostics

Analyzes Phase 1.17 real multi-sample rollout records to explain reward variance vs trajectory diversity.

No training, no re-rollout.

## Run

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

python scripts/analyze_multisample_rollout.py \
  --rollout-path experiments/phase117_multisample_rollout_5_g4/rollout_records.jsonl \
  --group-summary-path experiments/phase117_multisample_rollout_5_g4/group_summaries.jsonl \
  --output-dir experiments/phase118a_rollout_diagnostics_5_g4
```

## Outputs

- `group_diagnostics.jsonl` — per-group reward / query / trajectory analysis
- `diagnosis_summary.json` — aggregate category counts and main diagnosis
- `case_studies.md` — illustrative examples
- `summary.json` — compact summary for reporting
