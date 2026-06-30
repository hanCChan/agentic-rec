# Phase 1.11：Real DataProto Compatibility Check

尝试将 `DataProtoMock` 转换为真实 `verl.protocol.DataProto`；若 verl/tensordict 不可用或版本不兼容，则 graceful fallback。**不训练 GRPO，不调用 actor.forward。**

## 运行

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

python scripts/smoke_real_dataproto_compat.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --tokenizer-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
  --output-dir experiments/phase111_real_dataproto_compat_10
```

## 产物

- `summary.json`
- `compatibility_report.json`

## 说明

真实 `DataProto` 要求 `non_tensor_batch` 为 `np.ndarray(dtype=object)`，适配器会自动转换。Phase 1.12 再做 actor logprob dry-run 设计。
