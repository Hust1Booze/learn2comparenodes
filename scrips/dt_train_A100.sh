#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=a100
#SBATCH -J dt_bnb
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --qos=a100


nvidia-smi

# ✅ 加载 CUDA（如果没有自动加载）
export CUDA_HOME=/lab/cuda/cuda-11.8

# ✅ 激活 Conda 环境（必须在加载 gcc 之后）
source activate bnb

root_dir=$(pwd)
echo "root_dir:"${root_dir}

# 创建 checkpoints 目录
mkdir -p checkpoints

# ✅ 启动训练任务
deepspeed --num_gpus=4 learning/dt_train.py --deepspeed_config ds_config.json
