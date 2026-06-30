# Phase 1.14 Dry-Run — 1 record

Reference logprob / KL dry-run（shared-ref，KL≈0，ratio≈1）。

## 复现

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/smoke_ref_kl_dryrun.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase114_ref_kl_dryrun_1 \
  --num-records 1 --shared-ref
```
