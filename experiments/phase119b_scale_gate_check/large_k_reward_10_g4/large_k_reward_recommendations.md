# Phase 1.18f Large-K Reward Recommendations

## Gate Result

- gate_passed: **True**
- safe_for_phase_119: **True**
- recommended_candidate: **`reward_largek_mix_1000`**

Deployable global candidate `reward_largek_mix_1000` passes gate: retrieval_spread=0.80, penalty_only=0.00, zero_std=0.20.

## Candidate Summary

### `reward_current`

- diagnostic_only: **False**
- zero_std_group_rate: **0.20**
- retrieval_quality_spread_group_rate: **0.20**
- penalty_only_spread_group_rate: **0.60**
- mean_abs_sequence_advantage: **0.7146**
- safe_candidate: **False**
- groups_with_spread: ['esci_val_0', 'esci_val_1', 'esci_val_2', 'esci_val_3', 'esci_val_4', 'esci_val_6', 'esci_val_7', 'esci_val_8']
- groups_still_collapsed: ['esci_val_5', 'esci_val_9']

### `reward_ndcg10`

- diagnostic_only: **False**
- zero_std_group_rate: **0.80**
- retrieval_quality_spread_group_rate: **0.20**
- penalty_only_spread_group_rate: **0.00**
- mean_abs_sequence_advantage: **0.1732**
- safe_candidate: **False**
- groups_with_spread: ['esci_val_1', 'esci_val_6']
- groups_still_collapsed: ['esci_val_0', 'esci_val_2', 'esci_val_3', 'esci_val_4', 'esci_val_5', 'esci_val_7', 'esci_val_8', 'esci_val_9']

### `reward_ndcg100`

- diagnostic_only: **False**
- zero_std_group_rate: **0.60**
- retrieval_quality_spread_group_rate: **0.40**
- penalty_only_spread_group_rate: **0.00**
- mean_abs_sequence_advantage: **0.3384**
- safe_candidate: **False**
- groups_with_spread: ['esci_val_1', 'esci_val_4', 'esci_val_5', 'esci_val_6']
- groups_still_collapsed: ['esci_val_0', 'esci_val_2', 'esci_val_3', 'esci_val_7', 'esci_val_8', 'esci_val_9']

### `reward_ndcg1000`

- diagnostic_only: **False**
- zero_std_group_rate: **0.20**
- retrieval_quality_spread_group_rate: **0.80**
- penalty_only_spread_group_rate: **0.00**
- mean_abs_sequence_advantage: **0.6918**
- safe_candidate: **True**
- groups_with_spread: ['esci_val_0', 'esci_val_1', 'esci_val_4', 'esci_val_5', 'esci_val_6', 'esci_val_7', 'esci_val_8', 'esci_val_9']
- groups_still_collapsed: ['esci_val_2', 'esci_val_3']

### `reward_largek_mix_100`

- diagnostic_only: **False**
- zero_std_group_rate: **0.60**
- retrieval_quality_spread_group_rate: **0.40**
- penalty_only_spread_group_rate: **0.00**
- mean_abs_sequence_advantage: **0.3384**
- safe_candidate: **False**
- groups_with_spread: ['esci_val_1', 'esci_val_4', 'esci_val_5', 'esci_val_6']
- groups_still_collapsed: ['esci_val_0', 'esci_val_2', 'esci_val_3', 'esci_val_7', 'esci_val_8', 'esci_val_9']

### `reward_largek_mix_1000`

- diagnostic_only: **False**
- zero_std_group_rate: **0.20**
- retrieval_quality_spread_group_rate: **0.80**
- penalty_only_spread_group_rate: **0.00**
- mean_abs_sequence_advantage: **0.6919**
- safe_candidate: **True**
- groups_with_spread: ['esci_val_0', 'esci_val_1', 'esci_val_4', 'esci_val_5', 'esci_val_6', 'esci_val_7', 'esci_val_8', 'esci_val_9']
- groups_still_collapsed: ['esci_val_2', 'esci_val_3']

### `reward_best_global_k`

- diagnostic_only: **True**
- zero_std_group_rate: **0.20**
- retrieval_quality_spread_group_rate: **0.80**
- penalty_only_spread_group_rate: **0.00**
- mean_abs_sequence_advantage: **0.6919**
- safe_candidate: **False**
- groups_with_spread: ['esci_val_0', 'esci_val_1', 'esci_val_4', 'esci_val_5', 'esci_val_6', 'esci_val_7', 'esci_val_8', 'esci_val_9']
- groups_still_collapsed: ['esci_val_2', 'esci_val_3']

### `reward_per_group_best_k`

- diagnostic_only: **True**
- zero_std_group_rate: **0.20**
- retrieval_quality_spread_group_rate: **0.80**
- penalty_only_spread_group_rate: **0.00**
- mean_abs_sequence_advantage: **0.6919**
- safe_candidate: **False**
- groups_with_spread: ['esci_val_0', 'esci_val_1', 'esci_val_4', 'esci_val_5', 'esci_val_6', 'esci_val_7', 'esci_val_8', 'esci_val_9']
- groups_still_collapsed: ['esci_val_2', 'esci_val_3']

## Decision

Proceed to **Phase 1.19: Real GRPO Loss Dry-Run** with `reward_largek_mix_1000` as quality-only advantage.

- Penalties must NOT enter GRPO advantage.
- Penalties remain diagnostics or minimal auxiliary terms only.
