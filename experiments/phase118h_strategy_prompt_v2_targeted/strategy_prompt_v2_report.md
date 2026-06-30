# Phase 1.18h Strategy Prompt V2 Report

- mode: `strategy_prompt_v2`
- target_collapse_only: **True**
- num_groups: **1**
- v1_strategy_collapse_count: **1**
- v2_strategy_collapse_count: **1**
- collapse_fixed_count: **0**
- collapse_fix_rate: **0.00**
- v2_gate_passed: **False**
- phase2_candidate_ready: **False**

## Per-Group Comparison

- `esci_val_3`: v1_unique=1 v2_unique=1 v1_jaccard=1.000 v2_jaccard=1.000 collapse_fixed=False
  - v2 queries: ['# 10 self-seal envelopes without window', '# 10 self-seal envelopes without window', '# 10 self-seal envelopes without window', '# 10 self-seal envelopes without window']

Fix remaining collapse or rerun full candidate set after targeted fix.

Strategy prompt V2 rerollout for collapse diagnostics only; no GRPO training was performed.

Blocking reason: strategy collapse remains; replace collapse group or revise prompt again.
