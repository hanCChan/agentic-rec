# Phase 1.18b Reward Sensitivity Diagnostics

## Main Finding

Current NDCG@10 reward is too sparse for GRPO grouping. Reward variance mostly comes from penalties rather than retrieval quality. Recall@50/MRR@50 or best-step retrieval signals should be considered before training.

**Recommended next phase:** Phase 1.18c: Reward Shaping Proposal + Dry-Run (separate retrieval quality from penalties)

## Current Reward Problem

- Zero-std group rate: **0.80**
- Penalty-only spread rate: **0.20**
- Retrieval-sensitive rate: **0.00**

Current total reward mixes final NDCG@10, step ΔNDCG, and penalties (search/repeat/invalid). Most GRPO groups collapse because retrieval-quality components do not spread across samples.

## Group-Level Evidence

### `esci_val_0` — `retrieval_results_change_but_metric_blind`

- Original query: bathroom fan without light
- Rewards: [-0.1, -0.1, -0.1, -0.1]
- Main spread source: **no_spread**
- NDCG@10 spread: 0.0000, Recall@50 spread: 0.0000
- Recommendation: BM25 topK changes but qrels/metrics do not reflect differences. Investigate label sparsity or add topK overlap / new-candidate signals.

### `esci_val_1` — `retrieval_results_change_but_metric_blind`

- Original query: !awnmower tires without rims
- Rewards: [-0.1, -0.1, -0.1, -0.1]
- Main spread source: **no_spread**
- NDCG@10 spread: 0.0000, Recall@50 spread: 0.0000
- Recommendation: BM25 topK changes but qrels/metrics do not reflect differences. Investigate label sparsity or add topK overlap / new-candidate signals.

### `esci_val_2` — `penalty_only_spread`

- Original query: expandable outdoor gate
- Rewards: [-0.1, -0.15000000000000002, -0.1, -0.15000000000000002]
- Main spread source: **penalty_only**
- NDCG@10 spread: 0.0000, Recall@50 spread: 0.0000
- Recommendation: Reward spread comes from search/repeat/invalid penalties, not retrieval quality. Separate retrieval-quality reward from cost penalties before GRPO.

### `esci_val_3` — `query_too_similar`

- Original query: # 10 self-seal envelopes without window
- Rewards: [0.10347796014867167, 0.10347796014867167, 0.10347796014867167, 0.10347796014867167]
- Main spread source: **no_spread**
- NDCG@10 spread: 0.0000, Recall@50 spread: 0.0000
- Recommendation: Query rewrites are semantically too close and retrieve similar topK. Improve rollout prompt diversity before changing reward.

### `esci_val_4` — `query_too_similar`

- Original query: #10 window envelopes without plastic
- Rewards: [0.2519590445170673, 0.2519590445170673, 0.2519590445170673, 0.2519590445170673]
- Main spread source: **no_spread**
- NDCG@10 spread: 0.0000, Recall@50 spread: 0.0000
- Recommendation: Query rewrites are semantically too close and retrieve similar topK. Improve rollout prompt diversity before changing reward.

## Metric Sensitivity

- Mean NDCG@10 spread: **0.0000**
- Mean NDCG@50 spread: **0.0000**
- Mean Recall@50 spread: **0.0000**
- Mean MRR@50 spread: **0.0000**

When NDCG@10 is uniformly zero but Recall@50/MRR@50 spread > 0, NDCG@10 is too coarse for GRPO grouping.

## TopK Overlap Analysis

- Mean pairwise top-10 overlap: **0.601**
- Mean pairwise top-50 overlap: **0.633**
- Mean pairwise top-100 overlap: **0.650**

High overlap + zero reward spread suggests query rewrites retrieve similar documents. Low overlap + zero metric spread suggests labels/metrics are blind to retrieval changes.

## Reward Spread Source

Groups where spread comes from penalties (not NDCG):
- `esci_val_2`: rewards=[-0.1, -0.15000000000000002, -0.1, -0.15000000000000002], search_cost_spread=0.050, ndcg_spread=0.000

## Recommendation for Phase 1.18c / 1.19

**Phase 1.18c: Reward Shaping Proposal + Dry-Run (separate retrieval quality from penalties)**

Proposed shaping dry-run (do not change formal reward yet):

```text
R = NDCG@10
  + alpha * Recall@50
  + beta * MRR@50
  + gamma * best_step_delta
  - penalties (tracked separately for GRPO advantage)
```

GRPO advantage should prioritize retrieval-quality terms; cost penalties should not be the primary source of group spread.
