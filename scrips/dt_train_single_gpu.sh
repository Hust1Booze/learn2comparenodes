#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=a100
#SBATCH -J dt_bnb_single
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --qos=a100

source activate bnb

root_dir=$(pwd)
echo "root_dir:"${root_dir}

# 创建checkpoints目录
mkdir -p checkpoints

# 单GPU测试，避免NCCL通信问题
export CUDA_VISIBLE_DEVICES=0

# 直接运行Python脚本（不使用deepspeed）
python learning/dt_train_single.py 