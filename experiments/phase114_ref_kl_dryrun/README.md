# Phase 1.14：Reference LogProb / KL Dry-Run

验证 actor/old/ref logprobs、token-level KL、ratio 与 `response_attention_mask` 对齐。**不训练、不接 GRPO。**

默认 `--shared-ref`：只加载一份 Qwen2.5-3B，ref=logprobs clone actor，KL≈0、ratio≈1。

## 运行

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

CUDA_VISIBLE_DEVICES=2 python scripts/smoke_ref_kl_dryrun.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase114_ref_kl_dryrun_2 \
  --num-records 2 \
  --shared-ref
```

## 产物

- `summary.json`
- `kl_shapes.json`
- `kl_stats.json`

## 说明

Phase 1.15 再做 GRPO advantage mock / grouped reward dry-run。
