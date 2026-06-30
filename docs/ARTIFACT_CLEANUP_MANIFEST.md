# Artifact Cleanup Manifest

> Cleanup commit: remove local checkpoint weights and early Phase 1 exploration dirs.
> Mainline Phase 2 evidence chain preserved. See `docs/RESULTS_INDEX.md`.

## Retained (mainline)

### Code
- `src/agents/phase2_smoke_dataset.py`, `tiny_grpo_smoke_trainer.py`, `grpo_stability_monitor.py`
- `src/agents/controlled_grpo_smoke_trainer.py`, `grpo_curve_analyzer.py`, `grpo_pilot_monitor.py`
- `src/agents/search_strategy_prompts_v2.py`, `strategy_episode_runner.py`, `large_k_reward_dryrun.py`
- `src/agents/no_update_trainer_dryrun.py`, `real_grpo_loss_dryrun.py`, `grpo_loss_dryrun.py`

### Scripts
- `scripts/build_phase2_clean_smoke_set.py`
- `scripts/run_tiny_grpo_smoke_training.py`, `run_3step_grpo_stability_smoke.py`
- `scripts/run_10step_grpo_controlled_smoke.py`, `compare_grpo_lr_sweep.py`, `run_50step_grpo_pilot.py`
- `scripts/smoke_strategy_prompt_v2.py`

### Experiment dirs (lightweight JSON/JSONL only)
- `experiments/phase21_tiny_grpo_smoke/`
- `experiments/phase22_3step_grpo_stability_smoke/`
- `experiments/phase23_10step_grpo_controlled_smoke/`
- `experiments/phase24_50step_grpo_pilot/`
- `experiments/phase118g_bm25_failure_cleanup_20_g4/`
- `experiments/phase118h_strategy_prompt_v2_20_g4/`

## Deleted (local only)

### Checkpoint weights (all experiments)
- `**/checkpoints/`, `**/smoke_step_*/`, `**/pilot_step_*/`
- `*.safetensors`, `*.bin`, `*.pt`, `*.pth` under experiments/

**Reason**: SMOKE_ONLY_DO_NOT_PROMOTE; never in Git; reclaim ~70GB disk.

### Early Phase 1 exploration dirs
Removed after metrics archived in `docs/RESULTS_INDEX.md`:

```text
experiments/phase15_*
experiments/phase16_*
experiments/phase17_*
experiments/phase18_*
experiments/phase19_*
experiments/phase110_*
experiments/phase111_*
experiments/phase112_*
experiments/phase113_*
experiments/phase114_*
experiments/phase115_*
experiments/phase116_*
experiments/phase117_*
experiments/phase118a_* through phase118f_*
experiments/phase118h_strategy_prompt_v2/ (non-20_g4)
experiments/phase118h_strategy_prompt_v2_targeted/
experiments/phase119a_*
experiments/phase119_real_grpo_loss_dryrun*
experiments/phase119b_scale_gate_check/
experiments/phase120_no_update_trainer_dryrun_20_g4/
experiments/phase1_env_smoke/
```

**Reason**: Superseded by Phase 2 mainline; core metrics in RESULTS_INDEX.

## Uncertain (kept for now)

- `experiments/phase118h_strategy_prompt_v2_20_g4/` — kept (clean set source)
- `experiments/phase118g_bm25_failure_cleanup_20_g4/` — kept (replacement logic evidence)

## Not deleted

- `Rec-R1/`, `verl/` upstream code
- `src/reward/`, `src/tools/` reward and BM25 paths
- Phase 2.1–2.4 lightweight experiment artifacts in Git
