# Phase 1.8：VERL Batch Mock / Shape Check

读取 Phase 1.7 的 `rollout_records.jsonl`，用 Qwen tokenizer 构造 VERL-like mock batch，验证 `input_ids` / `attention_mask` / `response_mask` / `rewards` 的 shape。**不训练 GRPO，不调用 vLLM/env/BM25。**

## 运行

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

python scripts/smoke_verl_batch_mock.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --tokenizer-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
  --output-dir experiments/phase18_verl_batch_mock_10 \
  --max-prompt-length 1024 \
  --max-response-length 2048 \
  --max-total-length 3072
```

## 产物

- `summary.json` — batch 统计 + `shape_check_passed`
- `batch_shapes.json` — 各 tensor shape/dtype
- `batch_meta.json` — 每条 record 的 token 长度与截断信息

## 说明

本阶段是 **mock batch**，尚未对齐 VERL trainer 的 `old_log_probs` / `advantages` / `returns`。Phase 1.9 再对齐 Rec-R1/VERL 训练字段。
