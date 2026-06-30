# Phase 1.14 Dry-Run — 4 records

Reference logprob / KL dry-run（shared-ref，无 OOM）。

## 复现

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/smoke_ref_kl_dryrun.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase114_ref_kl_dryrun_4 \
  --num-records 4 --shared-ref
```
