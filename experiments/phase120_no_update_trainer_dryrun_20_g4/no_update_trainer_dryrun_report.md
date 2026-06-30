# Phase 1.20 No-update VERL Trainer Dry-Run Report

## Mode

- mode: `no_update_trainer_dryrun`
- is_training: **False**
- reward_candidate: `reward_largek_mix_1000`

## DataProto / Trainer Path

- used_real_dataproto: **True**
- used_verl_compute_advantage: **True**
- fallback_to_project_advantage: **False**
- trainer_required_keys_passed: **True**

## Checks

- advantage_check_passed: **True**
- minibatch_check_passed: **True**
- loss_check_passed: **True**
- no_update_guard_passed: **True**

## No-Update Guard

- trainer_fit_called: **False**
- update_actor_called: **False**
- optimizer_step_called: **False**

## Quality Spread

- zero_std_group_rate: **0.15**
- retrieval_quality_spread_group_rate: **0.85**

## Loss Dry-Run

- policy_loss_finite: **True**
- clipfrac: **0.0000**
- mean_valid_ratio: **1.0202**
- mean_valid_kl: **0.020000**

## Mini-batch

- num_records: **80**
- num_ppo_minibatches: **4**
- num_microbatches_per_minibatch: **5**

Phase 1.20 no-update VERL trainer dry-run only. No trainer.fit(), update_actor(), or optimizer.step().

## Next Steps

Proceed to Phase 1.18g/1.18h cleanup and then Phase 2 smoke training only after residual collapse cases are handled.
