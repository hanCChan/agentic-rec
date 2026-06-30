# Phase 1.18f Large-K Reward Candidate Comparison

**Gate passed:** True
**Safe for Phase 1.19:** True
**Recommended candidate:** `reward_largek_mix_1000`

Deployable global candidate `reward_largek_mix_1000` passes gate: retrieval_spread=0.85, penalty_only=0.00, zero_std=0.15.

| Candidate | diagnostic | zero_std | retrieval_spread | penalty_only | safe |
|-----------|------------|----------|------------------|--------------|------|
| `reward_current` | False | 0.35 | 0.15 | 0.50 | False |
| `reward_ndcg10` | False | 0.85 | 0.15 | 0.00 | False |
| `reward_ndcg100` | False | 0.45 | 0.55 | 0.00 | False |
| `reward_ndcg1000` | False | 0.15 | 0.85 | 0.00 | True |
| `reward_largek_mix_100` | False | 0.45 | 0.55 | 0.00 | False |
| `reward_largek_mix_1000` | False | 0.15 | 0.85 | 0.00 | True |
| `reward_best_global_k` | True | 0.15 | 0.85 | 0.00 | False |
| `reward_per_group_best_k` | True | 0.15 | 0.85 | 0.00 | False |

**Best non-diagnostic:** `reward_largek_mix_1000` (zero_std=0.15)

