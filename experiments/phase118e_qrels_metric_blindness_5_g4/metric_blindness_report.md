# Phase 1.18e Metric Blindness Report

## Executive Summary

- Groups analyzed: **5**
- metric_has_quality_signal: **0.20**
- small_k_blind_large_k_signal: **0.40**
- qrels_sparse_all_k_blind: **0.00**
- bm25_retrieval_failure: **0.20**
- strategy_query_too_similar: **0.20**
- Recommended next phase: **Phase 1.18f: Large-K Reward Candidate Dry-Run**

Some groups show retrieval-quality spread at larger K. Proceed to Phase 1.18f reward candidate dry-run, then re-run Phase 1.19a gate.

## Q1-Q8 Answers

### Q1: How many relevant docs per ESCI query?

- `esci_val_0`: **9** relevant docs (highly relevant: 9)
- `esci_val_1`: **1** relevant docs (highly relevant: 1)
- `esci_val_2`: **1** relevant docs (highly relevant: 1)
- `esci_val_3`: **3** relevant docs (highly relevant: 3)
- `esci_val_4`: **3** relevant docs (highly relevant: 3)

### Q2: Are smoke qrels too sparse?

- qrels_sparse_query_rate: **0.40**

### Q3: Are relevant docs in BM25 top100 / top1000?

- `esci_val_0`: top10=0, top50=0, top100=0, top1000=1, best_rank=891
- `esci_val_1`: top10=0, top50=0, top100=1, top1000=1, best_rank=78
- `esci_val_2`: top10=0, top50=0, top100=0, top1000=0, best_rank=None
- `esci_val_3`: top10=1, top50=2, top100=2, top1000=3, best_rank=10
- `esci_val_4`: top10=1, top50=3, top100=3, top1000=3, best_rank=3

### Q4: When NDCG@10=0, does larger K have signal?

- `esci_val_0`: ndcg@10=[0.0, 0.0, 0.0, 0.0], ndcg@100=[0.0, 0.0, 0.0, 0.0], ndcg@1000=[0.023982037549061863, 0.0, 0.0, 0.0], type=`small_k_blind_large_k_signal`
- `esci_val_1`: ndcg@10=[0.0, 0.0, 0.6309297535714575, 0.0], ndcg@100=[0.15863495891559604, 0.15863495891559604, 0.6309297535714575, 0.15863495891559604], ndcg@1000=[0.15863495891559604, 0.15863495891559604, 0.6309297535714575, 0.15863495891559604], type=`metric_has_quality_signal`
- `esci_val_2`: ndcg@10=[0.0, 0.0, 0.0, 0.0], ndcg@100=[0.0, 0.0, 0.0, 0.0], ndcg@1000=[0.0, 0.0, 0.0, 0.0], type=`bm25_retrieval_failure`
- `esci_val_3`: ndcg@10=[0.13565197343244778, 0.13565197343244778, 0.13565197343244778, 0.13565197343244778], ndcg@100=[0.2332687988242164, 0.2332687988242164, 0.2332687988242164, 0.2332687988242164], ndcg@1000=[0.2989045084764575, 0.2989045084764575, 0.2989045084764575, 0.2989045084764575], type=`strategy_query_too_similar`
- `esci_val_4`: ndcg@10=[0.23463936301137822, 0.23463936301137822, 0.23463936301137822, 0.23463936301137822], ndcg@100=[0.42330579351536485, 0.42330579351536485, 0.42330579351536485, 0.42596238450433555], ndcg@1000=[0.42330579351536485, 0.42330579351536485, 0.42330579351536485, 0.42596238450433555], type=`small_k_blind_large_k_signal`

### Q5-Q6: Recall/MRR group spread by K

- `esci_val_0`: recall@100_spread=0.0000, recall@1000_spread=0.1111, mrr@100_spread=0.0000, candidate_k=['recall@1000', 'ndcg@1000', 'mrr@1000']
- `esci_val_1`: recall@100_spread=0.0000, recall@1000_spread=0.0000, mrr@100_spread=0.4872, candidate_k=['recall@50', 'recall@10', 'ndcg@50', 'ndcg@10', 'mrr@50']
- `esci_val_2`: recall@100_spread=0.0000, recall@1000_spread=0.0000, mrr@100_spread=0.0000, candidate_k=[]
- `esci_val_3`: recall@100_spread=0.0000, recall@1000_spread=0.0000, mrr@100_spread=0.0000, candidate_k=[]
- `esci_val_4`: recall@100_spread=0.0000, recall@1000_spread=0.0000, mrr@100_spread=0.0000, candidate_k=['ndcg@50', 'ndcg@1000', 'ndcg@100']

### Q7: Why metric unchanged when topK changes?

- `esci_val_0`: final_query_jaccard=0.640, top100_overlap=0.356, type=`small_k_blind_large_k_signal` — NDCG@10 spread is zero but larger-K NDCG/Recall/MRR shows group spread. Consider Rec-R1-style larger-K reward or quality-only advantage at @100.
- `esci_val_1`: final_query_jaccard=0.750, top100_overlap=0.621, type=`metric_has_quality_signal` — NDCG/Recall/MRR at some K produces group spread. Candidate metric K exists for reward dry-run after Phase 1.18f.
- `esci_val_2`: final_query_jaccard=0.526, top100_overlap=0.209, type=`bm25_retrieval_failure` — Query has relevant docs in qrels but BM25 top1000 does not retrieve them. BM25 tool cannot provide learnable feedback for this query.
- `esci_val_3`: final_query_jaccard=1.000, top100_overlap=1.000, type=`strategy_query_too_similar` — Strategy final queries and BM25 topK overlap are too similar; metric spread is zero despite query rewrites. Improve strategy prompt differentiation.
- `esci_val_4`: final_query_jaccard=0.857, top100_overlap=0.699, type=`small_k_blind_large_k_signal` — NDCG@10 spread is zero but larger-K NDCG/Recall/MRR shows group spread. Consider Rec-R1-style larger-K reward or quality-only advantage at @100.

### Q8: Metric K too small, qrels wrong, or bad smoke sample?

- **Recommendation:** Phase 1.18f: Large-K Reward Candidate Dry-Run

## Per-Group Classification

### esci_val_0

- Type: `small_k_blind_large_k_signal`
- Original query: `bathroom fan without light`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 4
- ndcg@10_spread: 0.0000
- ndcg@100_spread: 0.0000
- recall@100_spread: 0.0000

### esci_val_1

- Type: `metric_has_quality_signal`
- Original query: `!awnmower tires without rims`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 2
- ndcg@10_spread: 0.6309
- ndcg@100_spread: 0.4723
- recall@100_spread: 0.0000

### esci_val_2

- Type: `bm25_retrieval_failure`
- Original query: `expandable outdoor gate`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 4
- ndcg@10_spread: 0.0000
- ndcg@100_spread: 0.0000
- recall@100_spread: 0.0000

### esci_val_3

- Type: `strategy_query_too_similar`
- Original query: `# 10 self-seal envelopes without window`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 1
- ndcg@10_spread: 0.0000
- ndcg@100_spread: 0.0000
- recall@100_spread: 0.0000

### esci_val_4

- Type: `small_k_blind_large_k_signal`
- Original query: `#10 window envelopes without plastic`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 2
- ndcg@10_spread: 0.0000
- ndcg@100_spread: 0.0027
- recall@100_spread: 0.0000
