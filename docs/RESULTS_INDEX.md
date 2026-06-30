# Results Index — Mainline Evidence Chain

> Smoke/pilot checkpoints are **local only** (`SMOKE_ONLY_DO_NOT_PROMOTE`), never promoted or committed.

## Phase 1.x — Engineering Dry-Run (archived locally, metrics summarized here)

> Phase 1.15–1.20 / 1.19b experiment directories were removed from Git in cleanup commit `3161297`.
> Core metrics are summarized below; detailed JSONL artifacts are not in the repo.

| Phase | Commit | Dir | Key Result | Trained? | Next |
|-------|--------|-----|------------|----------|------|
| 1.15–1.20 | `98024c1` etc. | (removed) | DataProto, logprob, advantage, loss, no-update trainer dry-run | No | Phase 2.1 |
| 1.18g | `b1a64b1` | `experiments/phase118g_bm25_failure_cleanup_20_g4/` | BM25 failure cleanup, replacement candidates | No | Clean set |
| 1.18h | `52f8802` | `experiments/phase118h_strategy_prompt_v2_20_g4/` | Strategy prompt V2, 19/20 gate pass | No | Phase 2.1 |
| 1.19b | `3cde4a4` | (removed) | Scale gate stable @ 20_g4 | No | Phase 2.1 |

## Phase 2 — GRPO Smoke → Pilot (mainline)

| Phase | Commit | Dir | Key Metrics | Trained? | Checkpoint |
|-------|--------|-----|-------------|----------|------------|
| **2.1** | `7bb5ade` | `experiments/phase21_tiny_grpo_smoke/` | 1-step, parse=1.0, optimizer_step=true | 1 step | SMOKE_ONLY |
| **2.2** | `27b7bf0` | `experiments/phase22_3step_grpo_stability_smoke/` | 3/3 steps, max_kl=0.0047, max_grad=0.418 | 3 steps | SMOKE_ONLY |
| **2.3** | `5475563` | `experiments/phase23_10step_grpo_controlled_smoke/` | 10/10 steps, lr=5e-7 max_kl=0.0046, both_stable | 10 steps | SMOKE_ONLY |
| **2.4** | `286b12c` | `experiments/phase24_50step_grpo_pilot/` | 50/50 steps, pilot_passed, max_kl=0.049, fresh eval reward 0.373→0.395 | 50 steps | SMOKE_ONLY |

### Phase 2.4 Fresh Eval Curve

> **Note:** In-batch training `mean_reward` stayed flat (~0.373) on fixed preflight batch.
> The +5.9% trend below is from **fresh rollout eval**, not the training reward curve.
> See [EXTERNAL_CLAIMS_GUIDE.md](./EXTERNAL_CLAIMS_GUIDE.md).

| Step | mean_reward_largek_mix_1000 | parse_success_rate |
|------|----------------------------|-------------------|
| 0 | 0.373 | 1.0 |
| 10 | 0.385 | 1.0 |
| 25 | 0.390 | 1.0 |
| 50 | 0.395 | 1.0 |

### Phase 2.4 Training Stability

```text
pilot_passed = true
actual_update_steps = 50/50
max_approx_kl_nonnegative = 0.049 (< 0.2)
max_grad_norm = 0.422 (< 10)
NaN / OOM / early_stop = false
checkpoint_promoted = false
```

## Current Mainline Entry Points

```text
1. build_phase2_clean_smoke_set.py     → clean 20_g4
2. smoke_strategy_prompt_v2.py        → V2 strategy rollout
3. reward_largek_mix_1000             → outcome reward
4. run_tiny_grpo_smoke_training.py    → 1-step smoke
5. run_3step_grpo_stability_smoke.py  → 3-step stability
6. run_10step_grpo_controlled_smoke.py → 10-step controlled
7. run_50step_grpo_pilot.py           → 50-step pilot
```

## Next Step

**Phase 2.5**: Expand clean set (50 train + 20 heldout from ESCI val rescan), fix eval bugs, write 200-step pilot plan. See [PHASE2_5_ALIGNMENT_QUESTIONS.md](./PHASE2_5_ALIGNMENT_QUESTIONS.md).

## Claim Boundary

```text
Phase 2.4 establishes pilot-level engineering stability and early fresh-eval signal
on a curated clean 20-group set. It does NOT establish full benchmark performance
or outperformance over Rec-R1.
```

See [EXTERNAL_CLAIMS_GUIDE.md](./EXTERNAL_CLAIMS_GUIDE.md).
