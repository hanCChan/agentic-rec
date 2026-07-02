# Phase 2.5d-A KL Stop Report

## Stop Summary

| Field | Value |
|-------|-------|
| Target steps | **200** |
| Actual steps | **117** |
| learning_rate | **5e-7** |
| kl_coef | **0.02** |
| Stop reason | **KL hard stop** |
| max_approx_kl_nonnegative | **0.2042** |
| KL threshold | **0.2** |
| pilot_passed | **false** |
| failure_type | **kl_guard_stop** |

## Stability Signals

- **NaN / OOM**: none
- **max_grad_norm**: 0.439 (well below 10.0)
- **JSON / parse**: stable (parse ≥ 0.997 at all eval points)
- **invalid_action_rate**: ≤ 0.002
- **heldout collapse**: false (baseline 0.5453 → stop 0.5427, drop ~0.5%)
- **overfit hard-stop**: false (overfit_risk logged at step 117 but below threshold)

## Comparison to Phase 2.5c

| | 2.5c | 2.5d-A |
|--|------|--------|
| kl_coef | 0.01 | **0.02** |
| stop step | **117** | **117** |
| max KL | 0.202 | 0.204 |

Doubling `kl_coef` did **not** change the KL stop horizon. This suggests config-level KL control did not affect the training trajectory — possible KL-loss wiring or KL-scale issue (Phase 2.5e audit).

## Fresh Eval Curve

| step | train_reward | heldout_reward | parse (train/heldout) |
|-----:|-------------:|---------------:|:----------------------|
| 0 | 0.5471 | 0.5453 | 0.998 / 1.0 |
| 25 | 0.5631 | 0.5482 | 1.0 / 1.0 |
| 50 | 0.5547 | 0.5465 | 0.998 / 1.0 |
| 100 | 0.5539 | 0.5618 | 1.0 / 1.0 |
| 117 (final stop) | 0.5500 | 0.5427 | 1.0 / 1.0 |

Eval outputs: `eval_step_0/`, `eval_step_25/`, `eval_step_50/`, `eval_step_100/`, `final_stop_eval_step_117/`.

## GPU Layout

```text
CUDA_VISIBLE_DEVICES = 4,5,6,7
train_gpus = 0,1 → physical 4,5
eval_gpus = 2,3 → physical 6,7 (vLLM TP=2, isolated subprocess)
```

## Diagnosis

Long-horizon **KL drift** on fixed preflight batch under `lr=5e-7`. Not OOM, NaN, JSON collapse, or heldout overfit.

**Key anomaly:** `kl_coef` 0.01 → 0.02 did not extend stop step. Next: Phase 2.5e KL/loss wiring audit before config B.

## Next Step

**Phase 2.5e**: Verify `kl_coef` enters actor `total_loss` and affects gradients. Do **not** run config B until audit passes.
