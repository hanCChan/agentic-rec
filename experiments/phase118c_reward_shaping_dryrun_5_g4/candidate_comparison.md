# Phase 1.18c Candidate Reward Comparison

**Main conclusion:** Pure retrieval-quality candidates still show high zero-std rates on this smoke set. best_step_delta candidate reduces zero_std from 0.80 to 0.60, but still not training-ready. Overlap diagnostic improves variance without retrieval-quality spread — qrels/metrics may be blind; do not use overlap as training reward. Do not train yet. Current smoke set has insufficient retrieval-quality reward variance.

| Candidate | zero_std_rate | mean_std | mean_abs_adv | retrieval_spread_rate | penalty_only_rate | diagnostic_only |
|-----------|---------------|----------|--------------|----------------------|-------------------|-----------------|
| `reward_current` | 0.80 | 0.0050 | 0.2000 | 0.00 | 0.20 | False |
| `reward_quality_only` | 1.00 | 0.0000 | 0.0000 | 0.00 | 0.00 | False |
| `reward_quality_best_step` | 0.60 | 0.0073 | 0.3464 | 0.00 | 0.00 | False |
| `reward_penalty_decoupled` | 1.00 | 0.0000 | 0.0000 | 0.00 | 0.00 | False |
| `reward_hit_depth` | 1.00 | 0.0000 | 0.0000 | 0.00 | 0.00 | False |
| `reward_overlap_diagnostic` | 0.40 | 0.0101 | 0.5910 | 0.00 | 0.20 | True |

**Best by variance:** `reward_quality_best_step` (zero_std=0.6)

**Safe for training:** `none`

