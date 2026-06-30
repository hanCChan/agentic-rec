# Phase 1.19 Real GRPO Loss Dry-Run Report

## Configuration

- reward_candidate: `reward_largek_mix_1000`
- quality_only_advantage: **True**
- penalties_in_advantage: **False**

## Quality Spread (Large-K Reward)

- zero_std_group_rate: **0.20**
- retrieval_quality_spread_group_rate: **0.80**
- penalty_only_spread_group_rate: **0.00**

## Checks

- advantage_check_passed: **True**
- loss_check_passed: **True**
- used_real_dataproto: **True**

## Loss Dry-Run

- policy_loss_value: **-0.021687**
- clipfrac: **0.0000**
- mean_valid_ratio: **1.0202**
- mean_valid_kl: **0.020000**
- padding_loss_zero: **True**

Real strategy-group GRPO loss dry-run only. No optimizer.step or trainer was invoked.

## Per-Group Quality Rewards

- `esci_val_0`: quality=[0.04631649321685087, 0.0, 0.0, 0.0], spread=0.0463, source=`retrieval_quality_spread`
- `esci_val_1`: quality=[0.3599170101976473, 0.3599170101976473, 0.8809297535714575, 0.3599170101976473], spread=0.5210, source=`retrieval_quality_spread`
- `esci_val_2`: quality=[0.0, 0.0, 0.0, 0.0], spread=0.0000, source=`no_spread`
- `esci_val_3`: quality=[0.5089045084764575, 0.5089045084764575, 0.5089045084764575, 0.5089045084764575], spread=0.0000, source=`no_spread`
- `esci_val_4`: quality=[0.6566391268486982, 0.6566391268486982, 0.6566391268486982, 0.6592957178376689], spread=0.0027, source=`retrieval_quality_spread`
- `esci_val_5`: quality=[0.6132427910117996, 0.6141303706453494, 0.6132427910117996, 0.6073579826401189], spread=0.0068, source=`retrieval_quality_spread`
- `esci_val_6`: quality=[0.5728738537746889, 0.5279648767857288, 0.5279648767857288, 0.5279648767857288], spread=0.0449, source=`retrieval_quality_spread`
- `esci_val_7`: quality=[0.0, 0.0, 0.0, 0.30216065641792855], spread=0.3022, source=`retrieval_quality_spread`
- `esci_val_8`: quality=[0.0, 0.11045157784880903, 0.0, 0.0], spread=0.1105, source=`retrieval_quality_spread`
- `esci_val_9`: quality=[0.3067921240817092, 0.31109741180441614, 0.0, 0.3067921240817092], spread=0.3111, source=`retrieval_quality_spread`

## Next Steps

Phase 1.19 validates the loss path only. Before training:
1. Phase 1.19b — scale gate check on 10_g4 / 20_g4
2. Phase 1.18g — replace BM25 failure samples
3. Phase 1.18h — fix strategy query collapse
4. Phase 1.20 — no-update trainer dry-run
