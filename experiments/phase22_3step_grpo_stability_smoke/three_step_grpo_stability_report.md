# Phase 2.2 Three-Step GRPO Stability Smoke Report

## Mode

- phase: `2.2`
- mode: `3step_grpo_stability_smoke`
- reward_candidate: `reward_largek_mix_1000`
- checkpoint_label: **SMOKE_ONLY_DO_NOT_PROMOTE**

## Training Summary

- max_update_steps: **3**
- actual_update_steps: **3**
- optimizer_steps_called: **3**
- three_step_smoke_passed: **True**
- nan_detected: **False**
- oom_detected: **False**
- kl_exploded: **False**
- max_approx_kl_nonnegative: **0.00470927357673645**
- max_grad_norm: **0.41796875**
- max_abs_signed_logprob_gap: **0.011784723959863186**

## Per-Step Metrics

| step | policy_loss | approx_kl | signed_gap | clipfrac | grad_norm |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0956357792019844 | 0.0029994274955242872 | 0.007248674985021353 | 0.3544136881828308 | 0.4140625 |
| 2 | 0.09540420025587082 | 0.0038333002012223005 | 0.00979997031390667 | 0.3546772003173828 | 0.416015625 |
| 3 | 0.09525670856237411 | 0.00470927357673645 | 0.011784723959863186 | 0.3533596694469452 | 0.41796875 |

## Post-Train Eval Comparison

- 1-step parse_success_rate: **1.0**
- 3-step parse_success_rate: **1.0**
- 1-step mean_reward: **0.39059002180091124**
- 3-step mean_reward: **0.3960897209640319**
- 3-step json_format_ok: **True**

Phase 2.1 tiny GRPO smoke training only. Checkpoints are SMOKE_ONLY_DO_NOT_PROMOTE and must not be promoted.

## Next Steps

Run 10-step controlled smoke only after reviewing stability and post-train JSON metrics.
