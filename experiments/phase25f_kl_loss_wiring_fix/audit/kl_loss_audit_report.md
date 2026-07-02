# Phase 2.5e KL / Loss Wiring Audit

- audit_passed: **True**
- wiring_issue_detected: **False**
- can_run_config_b: **True**

## Expected vs Observed

- expected: `total_loss = policy_loss + kl_coef * kl_loss_used_in_loss`
- observed: `total_loss = policy_loss + kl_coef * kl_loss (backward)`

## Checks

- [PASS] **effective_kl_coef_matches_cli**: Trainer self.kl_coef equals CLI value in audit records
- [PASS] **kl_loss_used_in_backward_nonzero**: Differentiable kl_loss tensor is nonzero in backward path
- [PASS] **total_loss_changes_with_kl_coef**: step-1 total_loss spread=2.44e-04 (invariant=False); policy_loss spread=0.00e+00
- [PASS] **grad_norm_changes_with_kl_coef**: step-1 grad_norm spread=3.32e-02 (invariant=False)
- [PASS] **approx_kl_changes_with_kl_coef**: step-10 approx_kl values={'kl_0': 0.0035349251702427864, 'kl_0.01': 0.0037226902786642313, 'kl_0.1': 0.001973793376237154} (invariant=False)
- [PASS] **kl_coef_times_kl_loss_nonzero**: kl_coef * kl_loss is nonzero when coef>0

## Diagnosis

- policy_loss invariant but total_loss varies — expected when KL enters loss.

## Recommended Next

Proceed with config B: lr=2e-7, kl_coef=0.01
