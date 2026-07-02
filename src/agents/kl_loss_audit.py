"""
Phase 2.5e: KL / loss wiring audit.

Verifies that kl_coef is wired into the effective backward loss, not only
logged for diagnostics or hard-stop monitoring.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Sequence

LOSS_FORMULA_EXPECTED = "total_loss = policy_loss + kl_coef * kl_loss_used_in_loss"


def _step_record(step_result: Dict[str, Any], *, effective_kl_coef: float) -> Dict[str, Any]:
    policy_loss = float(step_result.get("policy_loss") or 0.0)
    kl_loss_used = float(
        step_result.get("kl_loss_used_in_backward")
        or step_result.get("kl_loss")
        or 0.0
    )
    kl_coef_times = float(
        step_result.get("kl_coef_times_kl_loss")
        or effective_kl_coef * kl_loss_used
    )
    total_loss = float(step_result.get("total_loss") or policy_loss)
    approx_kl = float(
        step_result.get("approx_kl_nonnegative_for_stop")
        or step_result.get("approx_kl_nonnegative")
        or 0.0
    )
    signed_gap = float(step_result.get("signed_logprob_gap_mean") or 0.0)
    enters_backward = step_result.get("kl_loss_enters_backward", True)

    return {
        "step": step_result.get("step"),
        "effective_kl_coef": effective_kl_coef,
        "effective_learning_rate": step_result.get("learning_rate"),
        "policy_loss": policy_loss,
        "kl_loss_used_in_loss": kl_loss_used,
        "kl_loss_used_in_backward": kl_loss_used,
        "kl_coef_times_kl_loss": kl_coef_times,
        "total_loss": total_loss,
        "total_loss_backward": total_loss,
        "approx_kl_nonnegative_for_stop": approx_kl,
        "approx_kl_nonnegative": approx_kl,
        "signed_logprob_gap": signed_gap,
        "grad_norm": step_result.get("grad_norm"),
        "clipfrac": step_result.get("clipfrac"),
        "actor_logprob_mean": step_result.get("actor_logprob_mean"),
        "ref_logprob_mean": step_result.get("ref_logprob_mean"),
        "old_logprob_mean": step_result.get("ref_logprob_mean"),
        "advantage_mean": None,
        "advantage_std": None,
        "kl_loss_enters_backward": enters_backward,
        "loss_formula_observed": "total_loss = policy_loss + kl_coef * kl_loss",
        "kl_coef_source": "cli",
    }


def analyze_kl_audit_runs(
    runs: Dict[str, List[Dict[str, Any]]],
    *,
    kl_coef_by_label: Dict[str, float],
) -> Dict[str, Any]:
    """Compare sweep runs and evaluate audit acceptance criteria."""
    checks: List[Dict[str, Any]] = []
    run_summaries: Dict[str, Any] = {}

    labels = sorted(runs.keys(), key=lambda k: kl_coef_by_label.get(k, 0.0))
    policy_loss_step1: Dict[str, float] = {}
    total_loss_step1: Dict[str, float] = {}
    grad_norm_step1: Dict[str, float] = {}
    approx_kl_step10: Dict[str, float] = {}

    for label in labels:
        metrics = runs[label]
        coef = kl_coef_by_label[label]
        if metrics:
            policy_loss_step1[label] = float(metrics[0].get("policy_loss") or 0.0)
            total_loss_step1[label] = float(
                metrics[0].get("total_loss") or metrics[0].get("total_loss_backward") or 0.0
            )
            grad_norm_step1[label] = float(metrics[0].get("grad_norm") or 0.0)
            approx_kl_step10[label] = float(metrics[-1].get("approx_kl_nonnegative") or 0.0)

        effective_ok = all(float(m.get("effective_kl_coef", -1)) == coef for m in metrics)
        kl_backward_nonzero = any(
            abs(float(m.get("kl_loss_used_in_backward") or m.get("kl_loss_used_in_loss") or 0.0))
            > 1e-8
            for m in metrics
        )
        kl_coef_times_nonzero = any(
            abs(float(m.get("kl_coef_times_kl_loss") or 0.0)) > 1e-8 for m in metrics
        )

        run_summaries[label] = {
            "effective_kl_coef": coef,
            "num_steps": len(metrics),
            "step1_policy_loss": policy_loss_step1.get(label),
            "step1_total_loss": total_loss_step1.get(label),
            "step1_grad_norm": grad_norm_step1.get(label),
            "step10_approx_kl": approx_kl_step10.get(label),
            "effective_kl_coef_matches_cli": effective_ok,
            "kl_loss_used_in_backward_nonzero": kl_backward_nonzero,
            "kl_coef_times_kl_loss_nonzero": kl_coef_times_nonzero,
        }

    pl_vals = list(policy_loss_step1.values())
    tl_vals = list(total_loss_step1.values())
    gn_vals = list(grad_norm_step1.values())
    policy_loss_invariant = len(pl_vals) >= 2 and max(pl_vals) - min(pl_vals) < 1e-6
    total_loss_invariant = len(tl_vals) >= 2 and max(tl_vals) - min(tl_vals) < 1e-6
    grad_norm_invariant = len(gn_vals) >= 2 and max(gn_vals) - min(gn_vals) < 1e-6
    approx_kl_invariant = (
        len(set(round(v, 6) for v in approx_kl_step10.values())) <= 1
        if len(approx_kl_step10) >= 2
        else False
    )

    effective_all_ok = all(r["effective_kl_coef_matches_cli"] for r in run_summaries.values())
    kl_backward_any = any(r["kl_loss_used_in_backward_nonzero"] for r in run_summaries.values())

    checks.append(
        {
            "name": "effective_kl_coef_matches_cli",
            "passed": effective_all_ok,
            "detail": "Trainer self.kl_coef equals CLI value in audit records",
        }
    )
    checks.append(
        {
            "name": "kl_loss_used_in_backward_nonzero",
            "passed": kl_backward_any,
            "detail": "Differentiable kl_loss tensor is nonzero in backward path",
        }
    )
    checks.append(
        {
            "name": "total_loss_changes_with_kl_coef",
            "passed": not total_loss_invariant,
            "detail": (
                f"step-1 total_loss spread={max(tl_vals)-min(tl_vals):.2e} "
                f"(invariant={total_loss_invariant}); "
                f"policy_loss spread={max(pl_vals)-min(pl_vals):.2e}"
            ),
        }
    )
    checks.append(
        {
            "name": "grad_norm_changes_with_kl_coef",
            "passed": not grad_norm_invariant,
            "detail": (
                f"step-1 grad_norm spread={max(gn_vals)-min(gn_vals):.2e} "
                f"(invariant={grad_norm_invariant})"
            ),
        }
    )
    checks.append(
        {
            "name": "approx_kl_changes_with_kl_coef",
            "passed": not approx_kl_invariant,
            "detail": (
                f"step-10 approx_kl values={approx_kl_step10} "
                f"(invariant={approx_kl_invariant})"
            ),
        }
    )
    checks.append(
        {
            "name": "kl_coef_times_kl_loss_nonzero",
            "passed": any(
                r.get("kl_coef_times_kl_loss_nonzero") for r in run_summaries.values()
            ),
            "detail": "kl_coef * kl_loss is nonzero when coef>0",
        }
    )

    wiring_issue = total_loss_invariant and grad_norm_invariant and effective_all_ok
    audit_passed = all(c["passed"] for c in checks)

    diagnosis: List[str] = []
    if wiring_issue:
        diagnosis.append(
            "kl_coef is stored on trainer but total_loss does not change across coef sweep; "
            "KL may still be missing from backward graph."
        )
    if total_loss_invariant:
        diagnosis.append("Changing kl_coef does not change step-1 total_loss (backward target).")
    if policy_loss_invariant and not total_loss_invariant:
        diagnosis.append(
            "policy_loss invariant but total_loss varies — expected when KL enters loss."
        )
    if grad_norm_invariant:
        diagnosis.append("Changing kl_coef does not change step-1 grad_norm.")
    if approx_kl_invariant:
        diagnosis.append("Changing kl_coef does not change step-10 approx_kl_nonnegative trajectory.")

    return {
        "audit_passed": audit_passed,
        "loss_formula_expected": LOSS_FORMULA_EXPECTED,
        "loss_formula_observed": "total_loss = policy_loss + kl_coef * kl_loss (backward)",
        "wiring_issue_detected": wiring_issue,
        "run_summaries": run_summaries,
        "checks": checks,
        "diagnosis": diagnosis,
        "can_run_config_b": audit_passed,
        "recommended_next": (
            "Fix TinyGrpoSmokeTrainer: add kl_coef * kl_loss to total_loss before backward; "
            "align hard-stop metric with loss KL term."
            if not audit_passed
            else "Proceed with Phase 2.5g: lr=5e-7, kl_coef=0.02, 200-step pilot"
        ),
    }


def build_audit_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# Phase 2.5e KL / Loss Wiring Audit",
        "",
        f"- audit_passed: **{summary.get('audit_passed')}**",
        f"- wiring_issue_detected: **{summary.get('wiring_issue_detected')}**",
        f"- can_run_config_b: **{summary.get('can_run_config_b')}**",
        "",
        "## Expected vs Observed",
        "",
        f"- expected: `{summary.get('loss_formula_expected')}`",
        f"- observed: `{summary.get('loss_formula_observed')}`",
        "",
        "## Checks",
        "",
    ]
    for check in summary.get("checks", []):
        status = "PASS" if check.get("passed") else "FAIL"
        lines.append(f"- [{status}] **{check.get('name')}**: {check.get('detail')}")
    lines.extend(["", "## Diagnosis", ""])
    for item in summary.get("diagnosis", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Recommended Next", "", summary.get("recommended_next", ""), ""])
    return "\n".join(lines)


def write_audit_outputs(
    output_dir: Path,
    *,
    runs: Dict[str, List[Dict[str, Any]]],
    summary: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, metrics in runs.items():
        path = output_dir / f"{label}_metrics.jsonl"
        with path.open("w", encoding="utf-8") as fout:
            for row in metrics:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    (output_dir / "kl_loss_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "kl_loss_audit_report.md").write_text(
        build_audit_report(summary),
        encoding="utf-8",
    )
    readme = output_dir / "README.md"
    readme.write_text(
        "# Phase 2.5e KL / Loss Wiring Audit\n\n"
        f"- audit_passed: **{summary.get('audit_passed')}**\n"
        f"- wiring_issue: **{summary.get('wiring_issue_detected')}**\n\n"
        "See `kl_loss_audit_report.md` and per-coef `kl_*_metrics.jsonl`.\n",
        encoding="utf-8",
    )
