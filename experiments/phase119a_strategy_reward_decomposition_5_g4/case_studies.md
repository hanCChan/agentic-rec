# Phase 1.19a Case Studies

## Q1: Is zero_std=0 from retrieval quality?

- Groups with `retrieval_quality_spread`: **1/5**
- `zero_std_group_rate_quality_only`: **0.80**
- Gate passed: **False** — Do not proceed to GRPO loss dry-run. Continue reward/prompt fixes or Phase 1.18e.

## Q2: Why is broad_recall better?

- broad_recall mean NDCG@10: **0.2002**
- broad_recall mean quality reward: **0.3004**
- broad_recall mean penalty: **-0.1100**
- broad_recall mean search calls: **1.80**
- exact_match mean NDCG@10: **0.0741**
- exact_match mean penalty: **-0.1300**

### esci_val_1 (broad_recall NDCG spike)

- broad_recall: reward=0.8464, ndcg=0.6309, searches=2
- exact_match: reward=-0.1500, ndcg=0.0000, searches=1
- Conclusion: broad_recall advantage is driven by **higher NDCG**, not fewer searches.

## Q3: Why are other strategies low?

- exact_match: mean_ndcg=0.0741, mean_penalty=-0.1300, mean_searches=1.40
- attribute_expansion: mean_ndcg=0.0741, mean_penalty=-0.1200, mean_searches=1.60
- constraint_preserving: mean_ndcg=0.0741, mean_penalty=-0.1200, mean_searches=1.60

## Q4: Why did esci_val_3 collapse?

- spread_source: `penalty_only_spread`
- unique_final_query_count: **1**
- total_reward_spread: **0.0500**
- ndcg_spread: **0.0000**

Per-strategy final queries:
- exact_match: `# 10 self-seal envelopes without window` (ndcg=0.1357)
- attribute_expansion: `# 10 self-seal envelopes without window` (ndcg=0.1357)
- broad_recall: `# 10 self-seal envelopes without window` (ndcg=0.1357)
- constraint_preserving: `# 10 self-seal envelopes without window` (ndcg=0.1357)

All four strategies converged to the same final query despite different prompts. Likely causes: query is too specific / qrels sparse / model defaults to original wording.

## Q5: Quality-only zero_std rate

- zero_std_group_rate_total_reward: **0.00**
- zero_std_group_rate_quality_only: **0.80**

Removing penalties collapses most groups — penalties were masking lack of quality signal.
