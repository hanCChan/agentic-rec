# Phase 2.5e KL / Loss Wiring Audit

- audit_passed: **False**
- wiring_issue_detected: **True**
- can_run_config_b: **False**

## Expected vs Observed

- expected: `total_loss = policy_loss + kl_coef * kl_loss_used_in_loss`
- observed: `policy_loss.backward() only; KL not added to total_loss`

## Checks

- [PASS] **effective_kl_coef_matches_cli**: Trainer self.kl_coef equals CLI value in audit records
- [PASS] **kl_loss_used_in_loss_nonzero**: Actor/ref logprob gap exists so KL term could be nonzero
- [FAIL] **total_loss_changes_with_kl_coef**: step-1 policy_loss spread=0.00e+00 (invariant=True)
- [FAIL] **grad_norm_changes_with_kl_coef**: step-1 grad_norm spread=0.00e+00 (invariant=True)
- [PASS] **approx_kl_changes_with_kl_coef**: step-10 approx_kl values={'kl_0': 0.003645054530352354, 'kl_0.01': 0.004079723730683327, 'kl_0.1': 0.0022036104928702116} (invariant=False)
- [PASS] **kl_coef_times_kl_loss_nonzero**: Signed-gap KL penalty magnitude is nonzero when coef>0

## Diagnosis

- kl_coef is stored on trainer but backward uses policy_loss only; logged kl_loss = approx_kl * kl_coef is post-hoc, not in autograd graph.
- Changing kl_coef does not change step-1 policy_loss (backward target).
- Changing kl_coef does not change step-1 grad_norm.

## Recommended Next

Fix TinyGrpoSmokeTrainer: add kl_coef * kl_loss to total_loss before backward; align hard-stop metric with loss KL term.
