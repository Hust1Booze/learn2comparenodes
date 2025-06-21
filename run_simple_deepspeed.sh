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
export TORCH_CUDA_ARCH_LIST="8.0"
export CUDA_HOME=/lab/cuda/cuda-11.8
export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"


# 设置环境变量解决NCCL问题
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker,lo

# 设置CUDA相关环境变量
export CUDA_LAUNCH_BLOCKING=1

# 创建检查点目录
mkdir -p checkpoints

# 单机多卡训练 (假设有2个GPU)
deepspeed --num_gpus=4 simple_deepspeed_example.py \
    --deepspeed_config simple_ds_config.json \
    --epochs 10 \
    --batch_size 16

# 多机多卡训练示例 (注释掉，按需使用)
# deepspeed --num_nodes=2 --num_gpus=2 --master_addr=192.168.1.100 --master_port=29500 \
#     simple_deepspeed_example.py \
#     --deepspeed_config simple_ds_config.json \
#     --epochs 10 \
#     --batch_size 32 