# Phase 2.5c KL Guard Stop Report

## Stop Summary

| Field | Value |
|-------|-------|
| stopped_at_step | **117** / 200 |
| stopped_by | **KL hard stop** |
| approx_kl_nonnegative | **0.202** |
| threshold | **0.2** |
| pilot_passed | **false** |
| failure_type | **kl_guard_stop** |

## Stability Signals

- **NaN / OOM**: none
- **grad_norm**: stable (max 0.443, well below 10.0)
- **JSON / parse**: stable (parse ≥ 0.99 at all eval points)
- **heldout reward**: did not collapse (0.550 → 0.564 at step 100)
- **train reward**: changed little (0.548 → 0.545 at step 100)

## Diagnosis

Long-horizon **KL drift** on fixed preflight batch: actor policy gradually diverged from frozen reference policy. Training reward on stale trajectories stayed flat while `approx_kl_nonnegative` accumulated until step 117.

This is **not** model collapse, JSON format failure, reward collapse, heldout overfit, or engineering error.

## Fresh Eval Curve (completed points)

| step | train_reward | heldout_reward |
|-----:|-------------:|---------------:|
| 0 | 0.5481 | 0.5500 |
| 50 | 0.5515 | 0.5517 |
| 100 | 0.5447 | 0.5642 |

Note: step 25 eval missing (eval hook was tied to checkpoint save step — fixed in Phase 2.5d).

## Configuration

```text
learning_rate = 5e-7
kl_coef = 0.01
max_update_steps = 200
train groups = 50
heldout groups = 20
save_steps = [50, 100, 200]
eval_steps = [0, 25, 50, 100, 200]  # step 25 not executed due to hook bug
```

## Next Step

**Phase 2.5d**: KL-controlled rerun with stronger KL constraint (`kl_coef=0.02`, lr=5e-7). Do not relax KL threshold. Do not promote checkpoint.
