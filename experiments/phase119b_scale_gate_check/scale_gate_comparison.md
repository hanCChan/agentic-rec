# Phase 1.19b Scale Gate Comparison

**Candidate:** `reward_largek_mix_1000`
**Stable gate passed:** True
**Safe for Phase 1.20:** True

Large-K reward_largek_mix_1000 passed gate at 10_g4 and 20_g4 with loss dry-run checks. Proceed to no-update trainer dry-run only.

## Baseline 5_g4

- zero_std=0.40, retrieval_spread=0.60, gate_passed=True

## Scale Results

| Scale | zero_std | retrieval_spread | penalty_only | gate | loss_check | completed |
|-------|----------|------------------|--------------|------|------------|-----------|
| 10_g4 | 0.20 | 0.80 | 0.00 | True | True | True |
| 20_g4 | 0.15 | 0.85 | 0.00 | True | True | True |
