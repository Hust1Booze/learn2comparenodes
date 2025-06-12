#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=a100
#SBATCH -J dt_bnb
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --qos=a100

# module load cuda/11.8
source activate bnb

root_dir=$(pwd)
echo "root_dir:"${root_dir}

# 创建checkpoints目录
mkdir -p checkpoints


# 设置环境变量以解决GCC版本兼容性问题
export TORCH_CUDA_ARCH_LIST="8.0"
export CUDA_HOME=/lab/cuda/cuda-11.8
export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"

# 使用DeepSpeed启动多GPU训练，明确指定GPU设备
deepspeed --num_gpus=4 --master_port=29500 learning/dt_train.py --deepspeed_config ds_config.json
