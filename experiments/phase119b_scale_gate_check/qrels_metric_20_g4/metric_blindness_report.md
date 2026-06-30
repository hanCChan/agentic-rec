# Phase 1.18e Metric Blindness Report

## Executive Summary

- Groups analyzed: **20**
- metric_has_quality_signal: **0.15**
- small_k_blind_large_k_signal: **0.70**
- qrels_sparse_all_k_blind: **0.05**
- bm25_retrieval_failure: **0.05**
- strategy_query_too_similar: **0.05**
- Recommended next phase: **Phase 1.18f: Large-K Reward Candidate Dry-Run**

Some groups show retrieval-quality spread at larger K. Proceed to Phase 1.18f reward candidate dry-run, then re-run Phase 1.19a gate.

## Q1-Q8 Answers

### Q1: How many relevant docs per ESCI query?

- `esci_val_0`: **9** relevant docs (highly relevant: 9)
- `esci_val_1`: **1** relevant docs (highly relevant: 1)
- `esci_val_10`: **2** relevant docs (highly relevant: 2)
- `esci_val_11`: **2** relevant docs (highly relevant: 2)
- `esci_val_12`: **1** relevant docs (highly relevant: 1)
- `esci_val_13`: **4** relevant docs (highly relevant: 4)
- `esci_val_14`: **4** relevant docs (highly relevant: 4)
- `esci_val_15`: **6** relevant docs (highly relevant: 6)
- `esci_val_16`: **2** relevant docs (highly relevant: 2)
- `esci_val_17`: **1** relevant docs (highly relevant: 1)
- `esci_val_18`: **1** relevant docs (highly relevant: 1)
- `esci_val_19`: **2** relevant docs (highly relevant: 2)
- `esci_val_2`: **1** relevant docs (highly relevant: 1)
- `esci_val_3`: **3** relevant docs (highly relevant: 3)
- `esci_val_4`: **3** relevant docs (highly relevant: 3)
- `esci_val_5`: **3** relevant docs (highly relevant: 3)
- `esci_val_6`: **1** relevant docs (highly relevant: 1)
- `esci_val_7`: **1** relevant docs (highly relevant: 1)
- `esci_val_8`: **4** relevant docs (highly relevant: 4)
- `esci_val_9`: **1** relevant docs (highly relevant: 1)

### Q2: Are smoke qrels too sparse?

- qrels_sparse_query_rate: **0.45**

### Q3: Are relevant docs in BM25 top100 / top1000?

- `esci_val_0`: top10=0, top50=0, top100=0, top1000=1, best_rank=891
- `esci_val_1`: top10=0, top50=0, top100=1, top1000=1, best_rank=78
- `esci_val_10`: top10=0, top50=0, top100=0, top1000=1, best_rank=644
- `esci_val_11`: top10=1, top50=2, top100=2, top1000=2, best_rank=5
- `esci_val_12`: top10=0, top50=0, top100=0, top1000=1, best_rank=192
- `esci_val_13`: top10=0, top50=1, top100=1, top1000=3, best_rank=29
- `esci_val_14`: top10=0, top50=0, top100=2, top1000=4, best_rank=66
- `esci_val_15`: top10=0, top50=2, top100=2, top1000=2, best_rank=15
- `esci_val_16`: top10=2, top50=2, top100=2, top1000=2, best_rank=1
- `esci_val_17`: top10=0, top50=0, top100=0, top1000=1, best_rank=148
- `esci_val_18`: top10=0, top50=0, top100=0, top1000=1, best_rank=133
- `esci_val_19`: top10=1, top50=2, top100=2, top1000=2, best_rank=2
- `esci_val_2`: top10=0, top50=0, top100=0, top1000=0, best_rank=None
- `esci_val_3`: top10=1, top50=2, top100=2, top1000=3, best_rank=10
- `esci_val_4`: top10=1, top50=3, top100=3, top1000=3, best_rank=3
- `esci_val_5`: top10=1, top50=1, top100=2, top1000=3, best_rank=3
- `esci_val_6`: top10=1, top50=1, top100=1, top1000=1, best_rank=8
- `esci_val_7`: top10=0, top50=0, top100=0, top1000=1, best_rank=604
- `esci_val_8`: top10=0, top50=0, top100=0, top1000=0, best_rank=None
- `esci_val_9`: top10=0, top50=0, top100=0, top1000=1, best_rank=664

### Q4: When NDCG@10=0, does larger K have signal?

- `esci_val_0`: ndcg@10=[0.0, 0.0, 0.0, 0.0], ndcg@100=[0.0, 0.0, 0.0, 0.0], ndcg@1000=[0.023982037549061863, 0.0, 0.0, 0.0], type=`small_k_blind_large_k_signal`
- `esci_val_1`: ndcg@10=[0.0, 0.0, 0.6309297535714575, 0.0], ndcg@100=[0.15863495891559604, 0.15863495891559604, 0.6309297535714575, 0.15863495891559604], ndcg@1000=[0.15863495891559604, 0.15863495891559604, 0.6309297535714575, 0.15863495891559604], type=`metric_has_quality_signal`
- `esci_val_10`: ndcg@10=[0.0, 0.0, 0.0, 0.0], ndcg@100=[0.0, 0.0, 0.0, 0.0], ndcg@1000=[0.06569559487135852, 0.0, 0.06569559487135852, 0.0], type=`small_k_blind_large_k_signal`
- `esci_val_11`: ndcg@10=[0.23719771276929622, 0.3065735963827292, 0.23719771276929622, 0.3065735963827292], ndcg@100=[0.35673628974777083, 0.4261121733612038, 0.35673628974777083, 0.42028114449095844], ndcg@1000=[0.35673628974777083, 0.4261121733612038, 0.35673628974777083, 0.42028114449095844], type=`metric_has_quality_signal`
- `esci_val_12`: ndcg@10=[0.0, 0.0, 0.0, 0.0], ndcg@100=[0.0, 0.0, 0.0, 0.0], ndcg@1000=[0.13170966856861138, 0.0, 0.0, 0.12676290360187092], type=`small_k_blind_large_k_signal`
- `esci_val_13`: ndcg@10=[0.0, 0.0, 0.0, 0.0], ndcg@100=[0.06815388814994837, 0.07807600999842033, 0.06090751116319715, 0.08035849674662018], ndcg@1000=[0.1123342667473422, 0.12049883080215115, 0.20063111328503114, 0.08035849674662018], type=`small_k_blind_large_k_signal`
- `esci_val_14`: ndcg@10=[0.0, 0.0, 0.0, 0.0], ndcg@100=[0.06327144169513317, 0.12383864596820819, 0.12391276750075686, 0.12391276750075686], ndcg@1000=[0.17017518497650763, 0.22082017817244434, 0.23092503304938702, 0.23092503304938702], type=`small_k_blind_large_k_signal`
- `esci_val_15`: ndcg@10=[0.0, 0.0, 0.0, 0.0], ndcg@100=[0.1238171174792244, 0.12123461252839605, 0.13418189994368512, 0.12123461252839605], ndcg@1000=[0.1238171174792244, 0.12123461252839605, 0.13418189994368512, 0.12123461252839605], type=`small_k_blind_large_k_signal`
- `esci_val_16`: ndcg@10=[1.0, 1.0, 1.0, 1.0], ndcg@100=[1.0, 1.0, 1.0, 1.0], ndcg@1000=[1.0, 1.0, 1.0, 1.0], type=`qrels_sparse_all_k_blind`
- `esci_val_17`: ndcg@10=[0.0, 0.0, 0.0, 0.0], ndcg@100=[0.0, 0.15117821092177644, 0.0, 0.0], ndcg@1000=[0.13852010756717748, 0.15117821092177644, 0.0, 0.13852010756717748], type=`small_k_blind_large_k_signal`
- `esci_val_18`: ndcg@10=[0.0, 0.0, 0.0, 0.0], ndcg@100=[0.0, 0.0, 0.15117821092177644, 0.0], ndcg@1000=[0.12022969719514143, 0.1477501131786861, 0.15117821092177644, 0.12022969719514143], type=`small_k_blind_large_k_signal`
- `esci_val_19`: ndcg@10=[0.38685280723454163, 0.38685280723454163, 0.38685280723454163, 0.38685280723454163], ndcg@100=[0.5094822457876332, 0.5172974527427161, 0.5172974527427161, 0.5172974527427161], ndcg@1000=[0.5094822457876332, 0.5172974527427161, 0.5172974527427161, 0.5172974527427161], type=`small_k_blind_large_k_signal`
- `esci_val_2`: ndcg@10=[0.0, 0.0, 0.0, 0.0], ndcg@100=[0.0, 0.0, 0.0, 0.0], ndcg@1000=[0.0, 0.0, 0.0, 0.0], type=`bm25_retrieval_failure`
- `esci_val_3`: ndcg@10=[0.13565197343244778, 0.13565197343244778, 0.13565197343244778, 0.13565197343244778], ndcg@100=[0.2332687988242164, 0.2332687988242164, 0.2332687988242164, 0.2332687988242164], ndcg@1000=[0.2989045084764575, 0.2989045084764575, 0.2989045084764575, 0.2989045084764575], type=`strategy_query_too_similar`
- `esci_val_4`: ndcg@10=[0.23463936301137822, 0.23463936301137822, 0.23463936301137822, 0.23463936301137822], ndcg@100=[0.42330579351536485, 0.42330579351536485, 0.42330579351536485, 0.42596238450433555], ndcg@1000=[0.42330579351536485, 0.42330579351536485, 0.42330579351536485, 0.42596238450433555], type=`small_k_blind_large_k_signal`
- `esci_val_5`: ndcg@10=[0.23463936301137822, 0.23463936301137822, 0.23463936301137822, 0.23463936301137822], ndcg@100=[0.31408534536487603, 0.31172874133871575, 0.31408534536487603, 0.31544701680920845], ndcg@1000=[0.3799094576784662, 0.38079703731201603, 0.3799094576784662, 0.37402464930678553], type=`small_k_blind_large_k_signal`
- `esci_val_6`: ndcg@10=[0.3562071871080222, 0.31546487678572877, 0.31546487678572877, 0.31546487678572877], ndcg@100=[0.3562071871080222, 0.31546487678572877, 0.31546487678572877, 0.31546487678572877], ndcg@1000=[0.3562071871080222, 0.31546487678572877, 0.31546487678572877, 0.31546487678572877], type=`metric_has_quality_signal`
- `esci_val_7`: ndcg@10=[0.0, 0.0, 0.0, 0.0], ndcg@100=[0.0, 0.0, 0.0, 0.0], ndcg@1000=[0.0, 0.0, 0.0, 0.10204829686736673], type=`small_k_blind_large_k_signal`
- `esci_val_8`: ndcg@10=[0.0, 0.0, 0.0, 0.0], ndcg@100=[0.0, 0.0, 0.0, 0.0], ndcg@1000=[0.0, 0.04358329827891656, 0.0, 0.0], type=`small_k_blind_large_k_signal`
- `esci_val_9`: ndcg@10=[0.0, 0.0, 0.0, 0.0], ndcg@100=[0.0, 0.0, 0.0, 0.0], ndcg@1000=[0.10664152167207065, 0.1109039882067372, 0.0, 0.10664152167207065], type=`small_k_blind_large_k_signal`

### Q5-Q6: Recall/MRR group spread by K

- `esci_val_0`: recall@100_spread=0.0000, recall@1000_spread=0.1111, mrr@100_spread=0.0000, candidate_k=['recall@1000', 'ndcg@1000', 'mrr@1000']
- `esci_val_1`: recall@100_spread=0.0000, recall@1000_spread=0.0000, mrr@100_spread=0.4872, candidate_k=['recall@50', 'recall@10', 'ndcg@50', 'ndcg@10', 'mrr@50']
- `esci_val_10`: recall@100_spread=0.0000, recall@1000_spread=0.5000, mrr@100_spread=0.0000, candidate_k=['recall@1000', 'ndcg@1000', 'mrr@1000']
- `esci_val_11`: recall@100_spread=0.0000, recall@1000_spread=0.0000, mrr@100_spread=0.1333, candidate_k=['mrr@50', 'mrr@1000', 'mrr@100', 'mrr@10', 'ndcg@50']
- `esci_val_12`: recall@100_spread=0.0000, recall@1000_spread=1.0000, mrr@100_spread=0.0000, candidate_k=['recall@1000', 'ndcg@1000', 'mrr@1000']
- `esci_val_13`: recall@100_spread=0.0000, recall@1000_spread=0.7500, mrr@100_spread=0.0238, candidate_k=['recall@1000', 'recall@50', 'ndcg@1000', 'ndcg@50', 'mrr@50']
- `esci_val_14`: recall@100_spread=0.2500, recall@1000_spread=0.2500, mrr@100_spread=0.0015, candidate_k=['recall@1000', 'recall@100', 'ndcg@1000', 'ndcg@100', 'mrr@1000']
- `esci_val_15`: recall@100_spread=0.0000, recall@1000_spread=0.0000, mrr@100_spread=0.0167, candidate_k=['recall@50', 'ndcg@50', 'mrr@50', 'mrr@1000', 'mrr@100']
- `esci_val_16`: recall@100_spread=0.0000, recall@1000_spread=0.0000, mrr@100_spread=0.0000, candidate_k=[]
- `esci_val_17`: recall@100_spread=1.0000, recall@1000_spread=1.0000, mrr@100_spread=0.0103, candidate_k=['recall@1000', 'recall@100', 'ndcg@1000', 'ndcg@100', 'mrr@1000']
- `esci_val_18`: recall@100_spread=1.0000, recall@1000_spread=0.0000, mrr@100_spread=0.0103, candidate_k=['recall@100', 'ndcg@100', 'ndcg@1000', 'mrr@100', 'mrr@1000']
- `esci_val_19`: recall@100_spread=0.0000, recall@1000_spread=0.0000, mrr@100_spread=0.0000, candidate_k=['ndcg@50', 'ndcg@1000', 'ndcg@100']
- `esci_val_2`: recall@100_spread=0.0000, recall@1000_spread=0.0000, mrr@100_spread=0.0000, candidate_k=[]
- `esci_val_3`: recall@100_spread=0.0000, recall@1000_spread=0.0000, mrr@100_spread=0.0000, candidate_k=[]
- `esci_val_4`: recall@100_spread=0.0000, recall@1000_spread=0.0000, mrr@100_spread=0.0000, candidate_k=['ndcg@50', 'ndcg@1000', 'ndcg@100']
- `esci_val_5`: recall@100_spread=0.0000, recall@1000_spread=0.0000, mrr@100_spread=0.0000, candidate_k=['ndcg@1000', 'ndcg@100']
- `esci_val_6`: recall@100_spread=0.0000, recall@1000_spread=0.0000, mrr@100_spread=0.0417, candidate_k=['mrr@50', 'mrr@1000', 'mrr@100', 'mrr@10', 'ndcg@50']
- `esci_val_7`: recall@100_spread=0.0000, recall@1000_spread=1.0000, mrr@100_spread=0.0000, candidate_k=['recall@1000', 'ndcg@1000', 'mrr@1000']
- `esci_val_8`: recall@100_spread=0.0000, recall@1000_spread=0.3333, mrr@100_spread=0.0000, candidate_k=['recall@1000', 'ndcg@1000', 'mrr@1000']
- `esci_val_9`: recall@100_spread=0.0000, recall@1000_spread=1.0000, mrr@100_spread=0.0000, candidate_k=['recall@1000', 'ndcg@1000', 'mrr@1000']

### Q7: Why metric unchanged when topK changes?

- `esci_val_0`: final_query_jaccard=0.640, top100_overlap=0.356, type=`small_k_blind_large_k_signal` — NDCG@10 spread is zero but larger-K NDCG/Recall/MRR shows group spread. Consider Rec-R1-style larger-K reward or quality-only advantage at @100.
- `esci_val_1`: final_query_jaccard=0.750, top100_overlap=0.621, type=`metric_has_quality_signal` — NDCG/Recall/MRR at some K produces group spread. Candidate metric K exists for reward dry-run after Phase 1.18f.
- `esci_val_10`: final_query_jaccard=0.811, top100_overlap=0.433, type=`small_k_blind_large_k_signal` — NDCG@10 spread is zero but larger-K NDCG/Recall/MRR shows group spread. Consider Rec-R1-style larger-K reward or quality-only advantage at @100.
- `esci_val_11`: final_query_jaccard=0.568, top100_overlap=0.865, type=`metric_has_quality_signal` — NDCG/Recall/MRR at some K produces group spread. Candidate metric K exists for reward dry-run after Phase 1.18f.
- `esci_val_12`: final_query_jaccard=0.473, top100_overlap=0.850, type=`small_k_blind_large_k_signal` — NDCG@10 spread is zero but larger-K NDCG/Recall/MRR shows group spread. Consider Rec-R1-style larger-K reward or quality-only advantage at @100.
- `esci_val_13`: final_query_jaccard=0.644, top100_overlap=0.501, type=`small_k_blind_large_k_signal` — NDCG@10 spread is zero but larger-K NDCG/Recall/MRR shows group spread. Consider Rec-R1-style larger-K reward or quality-only advantage at @100.
- `esci_val_14`: final_query_jaccard=0.787, top100_overlap=0.699, type=`small_k_blind_large_k_signal` — NDCG@10 spread is zero but larger-K NDCG/Recall/MRR shows group spread. Consider Rec-R1-style larger-K reward or quality-only advantage at @100.
- `esci_val_15`: final_query_jaccard=0.867, top100_overlap=0.833, type=`small_k_blind_large_k_signal` — NDCG@10 spread is zero but larger-K NDCG/Recall/MRR shows group spread. Consider Rec-R1-style larger-K reward or quality-only advantage at @100.
- `esci_val_16`: final_query_jaccard=0.833, top100_overlap=0.725, type=`qrels_sparse_all_k_blind` — No retrieval-quality spread at any K; qrels may be too sparse or BM25 cannot surface relevant docs for this query. Replace or expand smoke samples.
- `esci_val_17`: final_query_jaccard=0.583, top100_overlap=0.670, type=`small_k_blind_large_k_signal` — NDCG@10 spread is zero but larger-K NDCG/Recall/MRR shows group spread. Consider Rec-R1-style larger-K reward or quality-only advantage at @100.
- `esci_val_18`: final_query_jaccard=0.780, top100_overlap=0.778, type=`small_k_blind_large_k_signal` — NDCG@10 spread is zero but larger-K NDCG/Recall/MRR shows group spread. Consider Rec-R1-style larger-K reward or quality-only advantage at @100.
- `esci_val_19`: final_query_jaccard=0.622, top100_overlap=0.826, type=`small_k_blind_large_k_signal` — NDCG@10 spread is zero but larger-K NDCG/Recall/MRR shows group spread. Consider Rec-R1-style larger-K reward or quality-only advantage at @100.
- `esci_val_2`: final_query_jaccard=0.526, top100_overlap=0.209, type=`bm25_retrieval_failure` — Query has relevant docs in qrels but BM25 top1000 does not retrieve them. BM25 tool cannot provide learnable feedback for this query.
- `esci_val_3`: final_query_jaccard=1.000, top100_overlap=1.000, type=`strategy_query_too_similar` — Strategy final queries and BM25 topK overlap are too similar; metric spread is zero despite query rewrites. Improve strategy prompt differentiation.
- `esci_val_4`: final_query_jaccard=0.857, top100_overlap=0.699, type=`small_k_blind_large_k_signal` — NDCG@10 spread is zero but larger-K NDCG/Recall/MRR shows group spread. Consider Rec-R1-style larger-K reward or quality-only advantage at @100.
- `esci_val_5`: final_query_jaccard=0.795, top100_overlap=0.758, type=`small_k_blind_large_k_signal` — NDCG@10 spread is zero but larger-K NDCG/Recall/MRR shows group spread. Consider Rec-R1-style larger-K reward or quality-only advantage at @100.
- `esci_val_6`: final_query_jaccard=0.800, top100_overlap=0.885, type=`metric_has_quality_signal` — NDCG/Recall/MRR at some K produces group spread. Candidate metric K exists for reward dry-run after Phase 1.18f.
- `esci_val_7`: final_query_jaccard=0.412, top100_overlap=0.080, type=`small_k_blind_large_k_signal` — NDCG@10 spread is zero but larger-K NDCG/Recall/MRR shows group spread. Consider Rec-R1-style larger-K reward or quality-only advantage at @100.
- `esci_val_8`: final_query_jaccard=0.684, top100_overlap=0.442, type=`small_k_blind_large_k_signal` — NDCG@10 spread is zero but larger-K NDCG/Recall/MRR shows group spread. Consider Rec-R1-style larger-K reward or quality-only advantage at @100.
- `esci_val_9`: final_query_jaccard=0.632, top100_overlap=0.221, type=`small_k_blind_large_k_signal` — NDCG@10 spread is zero but larger-K NDCG/Recall/MRR shows group spread. Consider Rec-R1-style larger-K reward or quality-only advantage at @100.

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

### esci_val_10

- Type: `small_k_blind_large_k_signal`
- Original query: `resveratrol complex 60 caps`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 3
- ndcg@10_spread: 0.0000
- ndcg@100_spread: 0.0000
- recall@100_spread: 0.0000

### esci_val_11

- Type: `metric_has_quality_signal`
- Original query: `#10 window envelopes not self seal`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 4
- ndcg@10_spread: 0.0694
- ndcg@100_spread: 0.0694
- recall@100_spread: 0.0000

### esci_val_12

- Type: `small_k_blind_large_k_signal`
- Original query: `#4 braiding hair not stretched`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 4
- ndcg@10_spread: 0.0000
- ndcg@100_spread: 0.0000
- recall@100_spread: 0.0000

### esci_val_13

- Type: `small_k_blind_large_k_signal`
- Original query: `overnight pads for women extra heavy without wings`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 4
- ndcg@10_spread: 0.0000
- ndcg@100_spread: 0.0195
- recall@100_spread: 0.0000

### esci_val_14

- Type: `small_k_blind_large_k_signal`
- Original query: `10 hour pads without wings`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 3
- ndcg@10_spread: 0.0000
- ndcg@100_spread: 0.0606
- recall@100_spread: 0.2500

### esci_val_15

- Type: `small_k_blind_large_k_signal`
- Original query: `maxi pads without wings`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 3
- ndcg@10_spread: 0.0000
- ndcg@100_spread: 0.0129
- recall@100_spread: 0.0000

### esci_val_16

- Type: `qrels_sparse_all_k_blind`
- Original query: `always infinity without wings`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 2
- ndcg@10_spread: 0.0000
- ndcg@100_spread: 0.0000
- recall@100_spread: 0.0000

### esci_val_17

- Type: `small_k_blind_large_k_signal`
- Original query: `always maxi overnight pads without wings`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 3
- ndcg@10_spread: 0.0000
- ndcg@100_spread: 0.1512
- recall@100_spread: 1.0000

### esci_val_18

- Type: `small_k_blind_large_k_signal`
- Original query: `always maxi pads long super without wings`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 3
- ndcg@10_spread: 0.0000
- ndcg@100_spread: 0.1512
- recall@100_spread: 1.0000

### esci_val_19

- Type: `small_k_blind_large_k_signal`
- Original query: `#8 tags without string`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 3
- ndcg@10_spread: 0.0000
- ndcg@100_spread: 0.0078
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

### esci_val_5

- Type: `small_k_blind_large_k_signal`
- Original query: `10 open window envelopes without plastic window`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 3
- ndcg@10_spread: 0.0000
- ndcg@100_spread: 0.0037
- recall@100_spread: 0.0000

### esci_val_6

- Type: `metric_has_quality_signal`
- Original query: `#2 pencils with erasers sharpened not soft`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 2
- ndcg@10_spread: 0.0407
- ndcg@100_spread: 0.0407
- recall@100_spread: 0.0000

### esci_val_7

- Type: `small_k_blind_large_k_signal`
- Original query: `08 do not disturb`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 4
- ndcg@10_spread: 0.0000
- ndcg@100_spread: 0.0000
- recall@100_spread: 0.0000

### esci_val_8

- Type: `small_k_blind_large_k_signal`
- Original query: `flat tummy cream for women`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 3
- ndcg@10_spread: 0.0000
- ndcg@100_spread: 0.0000
- recall@100_spread: 0.0000

### esci_val_9

- Type: `small_k_blind_large_k_signal`
- Original query: `#1 best and not expensive bath back brush cream color`
- Strategies: exact_match, attribute_expansion, broad_recall, constraint_preserving
- unique_final_query_count: 4
- ndcg@10_spread: 0.0000
- ndcg@100_spread: 0.0000
- recall@100_spread: 0.0000
