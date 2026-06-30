# Phase 2.3 Ten-Step Controlled GRPO Smoke Report

- learning_rate: **5e-07**
- actual_update_steps: **10**
- ten_step_smoke_passed: **True**
- stability_class: **mild_drift**
- max_approx_kl_nonnegative: **0.004605456721037626**
- max_grad_norm: **0.421875**

## Trends

- policy_loss_trend: `flat`
- approx_kl_trend: `increasing`
- grad_norm_trend: `flat`
- clipfrac_trend: `flat`
- reward_trend: `flat`

## Post-Train Eval (10-step)

- parse_success_rate: **1.0**
- invalid_action_rate: **0.0**
- json_format_ok: **True**
- mean_reward_largek_mix_1000: **0.3893170713839431**

- 1-step mean_reward: **0.39059002180091124** parse: **1.0**
- 3-step mean_reward: **0.3960897209640319** parse: **1.0**

Try conservative LR=5e-7 or higher kl_coef before scaling steps.
