# Phase 2.1 Tiny GRPO Smoke Training Report

## Mode

- phase: `2.1`
- mode: `tiny_grpo_smoke_training`
- reward_candidate: `reward_largek_mix_1000`
- checkpoint_label: **SMOKE_ONLY_DO_NOT_PROMOTE**

## Clean Set

- num_groups: **20**
- excluded_group_ids: `['esci_val_3']`
- num_replacements_added: **2**

## Preflight Gate

- v2_gate_passed: **True**
- v2_retrieval_quality_spread_group_rate: **0.95**
- v2_zero_std_group_rate: **0.05**
- v2_strategy_collapse_count: **0**

## Training

- max_update_steps: **1**
- actual_update_steps: **1**
- optimizer_step_called: **True**
- training_smoke_passed: **True**
- nan_detected: **False**
- oom_detected: **False**
- kl_exploded: **False**
- policy_loss: **0.0956357792019844**
- mean_kl: **-1.4615039825439453**
- clipfrac: **0.3544136881828308**
- grad_norm: **0.4140625**
- checkpoint_saved: **True**
- checkpoint_promoted: **False**

## Post-Train Eval

- parse_success_rate: **1.0**
- finish_rate: **1.0**
- invalid_action_rate: **0.0**
- mean_reward_largek_mix_1000: **0.39059002180091124**

Phase 2.1 tiny GRPO smoke training only. Checkpoints are SMOKE_ONLY_DO_NOT_PROMOTE and must not be promoted.

## Next Steps

Inspect tiny smoke logs; if stable, run 3-step smoke before any larger training.
