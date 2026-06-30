# Phase 1.18c Reward Shaping Recommendations

## Main Finding

Pure retrieval-quality candidates still show high zero-std rates on this smoke set. best_step_delta candidate reduces zero_std from 0.80 to 0.60, but still not training-ready. Overlap diagnostic improves variance without retrieval-quality spread — qrels/metrics may be blind; do not use overlap as training reward. Do not train yet. Current smoke set has insufficient retrieval-quality reward variance.

## Current Reward Problem

Phase 1.17/1.18b showed that total_reward spread comes mainly from search penalties, not NDCG@10. GRPO would learn to avoid penalties rather than improve retrieval.

## Candidate Results

### `reward_current`

- zero_std_group_rate: **0.80**
- mean_group_reward_std: **0.0050**
- retrieval_quality_spread_rate: **0.00**
- penalty_only_spread_rate: **0.20**
- diagnostic_only: **False**

### `reward_quality_only`

- zero_std_group_rate: **1.00**
- mean_group_reward_std: **0.0000**
- retrieval_quality_spread_rate: **0.00**
- penalty_only_spread_rate: **0.00**
- diagnostic_only: **False**

### `reward_quality_best_step`

- zero_std_group_rate: **0.60**
- mean_group_reward_std: **0.0073**
- retrieval_quality_spread_rate: **0.00**
- penalty_only_spread_rate: **0.00**
- diagnostic_only: **False**

### `reward_penalty_decoupled`

- zero_std_group_rate: **1.00**
- mean_group_reward_std: **0.0000**
- retrieval_quality_spread_rate: **0.00**
- penalty_only_spread_rate: **0.00**
- diagnostic_only: **False**

### `reward_hit_depth`

- zero_std_group_rate: **1.00**
- mean_group_reward_std: **0.0000**
- retrieval_quality_spread_rate: **0.00**
- penalty_only_spread_rate: **0.00**
- diagnostic_only: **False**

### `reward_overlap_diagnostic`

- zero_std_group_rate: **0.40**
- mean_group_reward_std: **0.0101**
- retrieval_quality_spread_rate: **0.00**
- penalty_only_spread_rate: **0.20**
- diagnostic_only: **True**

## Recommendation

- **Safe for training:** `none`
- **Best variance candidate:** `reward_quality_best_step`
- **Next phase:** Phase 1.18e: Qrels / Metric Blindness Analysis

**Preferred future formula (dry-run only):**

`retrieval_quality + 0.5 * best_step_delta_ndcg (penalties tracked separately)`

- Adding Recall@50/MRR@50 alone does not create group spread on this smoke set.
- best_step_delta improves spread slightly; consider retrieval_quality + best_step_delta in future dry-runs.
- Overlap diagnostic reduces zero_std but quality metrics do not — investigate qrels/metric blindness.
- All non-diagnostic candidates keep zero_std >= 0.6; improve rollout diversity before reward changes.
