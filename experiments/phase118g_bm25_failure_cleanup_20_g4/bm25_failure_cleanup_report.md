# Phase 1.18g BM25 Failure / Unlearnable Sample Cleanup Report

## Summary

- num_input_groups: **20**
- num_keep_for_phase2: **16**
- num_replace_recommended: **3**
- bm25_failure_count: **2**
- qrels_sparse_all_k_blind_count: **1**
- strategy_collapse_count: **1**
- phase2_candidate_set_size: **19**
- phase2_candidate_ready: **False**

## Learnability Counts

- `learnable_large_k`: **11**
- `learnable_small_k`: **5**
- `bm25_retrieval_failure`: **2**
- `qrels_sparse_all_k_blind`: **1**
- `strategy_collapse`: **1**
- `ambiguous_keep_for_analysis`: **0**

## Per-Group Cleanup

- `esci_val_0`: **learnable_large_k** (keep=True, action=`keep_for_phase2`)
- `esci_val_1`: **learnable_small_k** (keep=True, action=`keep_for_phase2`)
- `esci_val_10`: **learnable_large_k** (keep=True, action=`keep_for_phase2`)
- `esci_val_11`: **learnable_small_k** (keep=True, action=`keep_for_phase2`)
- `esci_val_12`: **learnable_large_k** (keep=True, action=`keep_for_phase2`)
- `esci_val_13`: **learnable_small_k** (keep=True, action=`keep_for_phase2`)
- `esci_val_14`: **learnable_large_k** (keep=True, action=`keep_for_phase2`)
- `esci_val_15`: **learnable_small_k** (keep=True, action=`keep_for_phase2`)
- `esci_val_16`: **qrels_sparse_all_k_blind** (keep=False, action=`replace_sample`)
  - Qrels are sparse and NDCG/Recall/MRR show no group spread at any K.
- `esci_val_17`: **learnable_large_k** (keep=True, action=`keep_for_phase2`)
- `esci_val_18`: **learnable_large_k** (keep=True, action=`keep_for_phase2`)
- `esci_val_19`: **learnable_large_k** (keep=True, action=`keep_for_phase2`)
- `esci_val_2`: **bm25_retrieval_failure** (keep=False, action=`replace_sample`)
  - Relevant documents exist in qrels, but BM25 top1000 did not retrieve any relevant document.
- `esci_val_3`: **strategy_collapse** (keep=False, action=`fix_strategy_prompt_phase118h`)
  - Strategy final queries collapsed to identical or highly similar queries, producing zero metric spread.
- `esci_val_4`: **learnable_large_k** (keep=True, action=`keep_for_phase2`)
- `esci_val_5`: **learnable_large_k** (keep=True, action=`keep_for_phase2`)
- `esci_val_6`: **learnable_small_k** (keep=True, action=`keep_for_phase2`)
- `esci_val_7`: **learnable_large_k** (keep=True, action=`keep_for_phase2`)
- `esci_val_8`: **bm25_retrieval_failure** (keep=False, action=`replace_sample`)
  - Relevant documents exist in qrels, but BM25 top1000 did not retrieve any relevant document.
- `esci_val_9`: **learnable_large_k** (keep=True, action=`keep_for_phase2`)

## Phase 2 Readiness

- blocking_reason: strategy_collapse remains; run Phase 1.18h before Phase 2 training.

## Next Steps

1. Phase 1.18h — Strategy Prompt V2 for collapse cases
2. Apply replacement candidates and re-run 20_g4 smoke set construction
3. Phase 2.1 — Tiny GRPO smoke training only after cleanup + prompt fix
