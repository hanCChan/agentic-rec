#!/bin/bash
# ESCI 全量 GRPO 训练 (3B + BM25)
# 断点: 每 save_freq 步保存到 checkpoints/.../actor/global_step_*
# 注意: 本仓库 verl 无 PPO 自动 resume; 中断后可用最新 checkpoint 作 BASE_MODEL 继续训(优化器状态会丢)
set -euo pipefail

DATE=$(date '+%Y-%m-%d-%H-%M-%S')

export BASE_MODEL=${BASE_MODEL:-/data1/hcc/.hf_home/Qwen2.5-3B-Instruct}
export DATA_DIR=${DATA_DIR:-data/esci/inst/sparse/subset}
export N_GPUS=${N_GPUS:-2}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
export PROJECT_NAME=${PROJECT_NAME:-recr1-esci}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-esci-qwen3b-grpo-full}
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=${WANDB_MODE:-offline}

CKPT_DIR="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
mkdir -p exp_log "${CKPT_DIR}"

echo "[train] BASE_MODEL=${BASE_MODEL}"
echo "[train] DATA_DIR=${DATA_DIR}"
echo "[train] CKPT_DIR=${CKPT_DIR}"
echo "[train] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/val.parquet" \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path="${BASE_MODEL}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=12 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=['console'] \
    +trainer.val_before_train=True \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.total_epochs=20 \
    2>&1 | tee "exp_log/${EXPERIMENT_NAME}-${DATE}.log"
