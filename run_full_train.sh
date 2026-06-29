#!/bin/bash
# 后台启动 ESCI 全量训练
# 用法: bash /data1/hcc/agentic-rec/run_full_train.sh
set -euo pipefail

source /data1/hcc/agentic-rec/env.sh
cd /data1/hcc/agentic-rec/Rec-R1

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2,3}
export N_GPUS=2
export ROLLOUT_TP_SIZE=1
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

LOG=/data1/hcc/agentic-rec/full_train.log
PIDFILE=/data1/hcc/agentic-rec/full_train.pid

nohup bash scripts/train/train-esci_3b_full.sh > "${LOG}" 2>&1 &
echo $! > "${PIDFILE}"

echo "已后台启动, pid=$(cat ${PIDFILE})"
echo "日志: tail -f ${LOG}"
echo "断点: ls -lt checkpoints/recr1-esci/esci-qwen3b-grpo-full/actor/"
