# Phase 1.17 Real Multi-Sample Rollout Smoke

Real Qwen multi-sample rollouts with BM25 rewards and GRPO group advantages.
No training was performed.

## Run

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

CUDA_VISIBLE_DEVICES=2 python scripts/smoke_multisample_rollout.py \
  --num-base-records 5 \
  --group-size 4 \
  --model-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
  --output-dir experiments/phase117_multisample_rollout_5_g4 \
  --temperature 0.7 \
  --top-p 0.95 \
  --seed 42
```

## Outputs

Each run directory contains:

- `rollout_records.jsonl` — G real trajectories per base query
- `group_summaries.jsonl` — per-group diversity and reward stats
- `group_advantage_shapes.json` / `group_advantage_stats.json`
- `summary.json`
