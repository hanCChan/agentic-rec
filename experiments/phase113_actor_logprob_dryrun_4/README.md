# Phase 1.13 Dry-Run — 4 records

真实 Qwen2.5-3B logprob dry-run（`torch.no_grad()`，HuggingFace causal LM）。

## 指标（summary.json）

| 指标 | 值 |
|------|-----|
| real_logprob_check_passed | true |
| used_real_dataproto | true |
| mean_valid_logprob | -1.32 |
| 无 OOM | ✅ |

## 复现

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/smoke_actor_logprob_dryrun.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase113_actor_logprob_dryrun_4 \
  --num-records 4
```
