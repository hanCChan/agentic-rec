# Phase 1.13：Real Actor LogProb Dry-Run

使用 HuggingFace `AutoModelForCausalLM` 在 `torch.no_grad()` 下计算真实 response-token logprob。**不训练、不接 GRPO、不 optimizer.step。**

## 运行

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

CUDA_VISIBLE_DEVICES=2 python scripts/smoke_actor_logprob_dryrun.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --model-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
  --tokenizer-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
  --output-dir experiments/phase113_actor_logprob_dryrun_2 \
  --num-records 2 \
  --dtype bfloat16 \
  --device cuda
```

## 产物

- `summary.json`
- `real_logprob_shapes.json`
- `real_logprob_stats.json`

## 说明

默认 `--num-records 2` 小 batch，避免 OOM。Phase 1.14 再做 ref_logprob / KL dry-run 设计。
