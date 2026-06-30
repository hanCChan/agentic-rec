# Phase 1.10：DataProto / Reward Function Dry-Run

将 Phase 1.9 training fields 映射为 `DataProtoMock`，验证 payload shape，运行 `CommerceRewardFn` dry-run 与 actor input field check。**不训练 GRPO，不调用 actor.forward，不 optimizer.step。**

## 运行

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

python scripts/smoke_dataproto_reward_dryrun.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --tokenizer-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
  --output-dir experiments/phase110_dataproto_reward_dryrun_10
```

## 产物

- `summary.json`
- `dataproto_shapes.json`
- `reward_fn_output.json`

## 说明

本阶段使用 `DataProtoMock`，不是真实 `verl.protocol.DataProto`。Phase 1.11 再做真实 DataProto import 兼容性检查。
