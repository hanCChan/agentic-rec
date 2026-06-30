# Phase 1.14 Dry-Run — 2 records

Reference logprob / KL dry-run（shared-ref）。

| 指标 | 值 |
|------|-----|
| kl_check_passed | true |
| mean_valid_kl | 0.0 |
| mean_valid_ratio | 1.0 |

## 复现

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/smoke_ref_kl_dryrun.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase114_ref_kl_dryrun_2 \
  --num-records 2 --shared-ref
```
