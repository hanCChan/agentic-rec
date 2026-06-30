# Phase 1.12：Actor LogProb Interface Mock

侦察 verl `compute_log_prob` 所需字段，构造 actor-logprob-ready request，生成 mock `old_log_probs` / `entropys` 并做 shape check。**不调用 actor.forward，不计算真实 logprob，不训练。**

## 运行

```bash
source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec

python scripts/smoke_actor_logprob_mock.py \
  --rollout-path experiments/phase17_verl_adapter_smoke_10/rollout_records.jsonl \
  --tokenizer-path /data1/hcc/.hf_home/Qwen2.5-3B-Instruct \
  --output-dir experiments/phase112_actor_logprob_mock_10
```

## 产物

- `summary.json`
- `actor_logprob_request_shapes.json`
- `mock_logprob_output_shapes.json`

## 说明

verl `dp_actor.compute_log_prob` 使用 `responses`, `input_ids`, `attention_mask`, `position_ids`。本阶段 mock 输出 `is_mock=True`，**不可用于训练**。Phase 1.13 才考虑真实 actor logprob dry-run。
