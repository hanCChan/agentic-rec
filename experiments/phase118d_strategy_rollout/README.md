# Phase 1.18d Strategy-Controlled Rollout Smoke

Four fixed search strategies per query (exact_match, attribute_expansion, broad_recall, constraint_preserving).

No training, no reward formula changes.

## Run

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

CUDA_VISIBLE_DEVICES=2 python scripts/smoke_strategy_multisample_rollout.py \
  --num-base-records 5 \
  --group-size 4 \
  --model-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
  --output-dir experiments/phase118d_strategy_rollout_5_g4 \
  --temperature 0.7 \
  --top-p 0.95 \
  --seed 42
```

## Outputs

- `rollout_records.jsonl` — one record per strategy per query
- `group_summaries.jsonl` — per-group strategy metrics
- `strategy_summary.json` — cross-group strategy comparison
- `summary.json`
