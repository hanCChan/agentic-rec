# Phase 1.19b Scale Gate Check

Validate `reward_largek_mix_1000` gate stability at 10_g4 and 20_g4.

## Run

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec
CUDA_VISIBLE_DEVICES=2 python scripts/run_scale_gate_check.py \
  --scales 10 20 \
  --group-size 4 \
  --model-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
  --output-root experiments/phase119b_scale_gate_check
```

## Stable gate (Phase 1.20 prerequisite)

Both 10_g4 and 20_g4 must pass large-K gate with loss dry-run checks.
