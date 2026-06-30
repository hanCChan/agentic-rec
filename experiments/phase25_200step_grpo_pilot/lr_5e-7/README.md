# Phase 2.5c 200-Step GRPO Pilot

- pilot_passed: **false**
- failure_type: **kl_guard_stop**
- stopped_at_step: **117** / 200
- early_stop: **false** (KL guard in trainer, not eval hook)
- train groups: **50**, heldout groups: **20**
- checkpoint_label: **SMOKE_ONLY_DO_NOT_PROMOTE**
- checkpoint_promoted: **false**

## Key Metrics

```text
approx_kl_nonnegative at stop = 0.202 > 0.2
max_grad_norm = 0.443
NaN / OOM = false
heldout reward did not collapse
diagnosis = long-horizon KL drift without JSON/reward collapse
```

## Artifacts

- `pilot_200step_summary.json` — full run summary
- `pilot_200step_train_metrics.jsonl` — per-step training metrics
- `periodic_eval_summary.json` — fresh eval history (steps 0, 50, 100)
- `train_vs_heldout_curve.md` — reward curve table
- `overfit_risk_report.md` — overfit analysis
- `kl_stop_report.md` — KL stop diagnosis and next steps

See `docs/RESULTS_INDEX.md` for mainline context.
