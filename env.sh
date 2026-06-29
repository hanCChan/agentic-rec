# Rec-R1 / agentic-rec 环境激活脚本
# 用法: source /data1/hcc/agentic-rec/env.sh
source /home/hanchengcheng/miniconda3/etc/profile.d/conda.sh
conda activate recr1

# Java (pyserini / Lucene BM25 需要)
export JAVA_HOME=/home/hanchengcheng/miniconda3/envs/recr1/lib/jvm
export PATH=$JAVA_HOME/bin:$PATH
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.so

# pip / conda 缓存指向可写的大盘
export PIP_CACHE_DIR=/data1/hcc/.pip_cache
export CONDA_PKGS_DIRS=/data1/hcc/.conda_pkgs

# vllm / 训练相关
export VLLM_ATTENTION_BACKEND=XFORMERS
# HF 模型缓存放到大盘，避免塞满 home
export HF_HOME=/data1/hcc/.hf_home

echo "[env] recr1 activated | python=$(python --version 2>&1) | java=$(java -version 2>&1 | head -1)"
