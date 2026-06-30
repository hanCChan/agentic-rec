# Phase 1.16：GRPO Loss Dry-Run

PPO/GRPO-style clipped policy loss + KL penalty 独立 dry-run。**不训练、不接 GRPO trainer。**

## 运行

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

python scripts/smoke_grpo_loss_dryrun.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --output-dir experiments/phase116_grpo_loss_dryrun_10_g4 \
  --num-base-records 10 --group-size 4
```

## 模式

- `10_g4` — 正常 jitter + delta=0.02
- `10_g4_collapse` — `--no-synthetic-jitter`
- `10_g4_clipstress` — `--synthetic-logprob-delta 0.5`

## 说明

Phase 1.17 再做真实 multi-sample rollout smoke。
