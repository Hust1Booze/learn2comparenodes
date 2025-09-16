#!/bin/bash
#SBATCH -o learning/job.%j.out
#SBATCH --partition=a100
#SBATCH -J ds_bnb
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --qos=a100

nvidia-smi
# module load cuda/11.8
source activate bnb

root_dir=$(pwd)
echo "root_dir:"${root_dir}

# 创建checkpoints目录
mkdir -p checkpoints


# 设置环境变量以解决GCC版本兼容性问题
export TORCH_CUDA_ARCH_LIST="8.0"
export CUDA_HOME=/lab/cuda/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"

# python node_selection/behaviour_gen.py -n_cpu 20

# 使用DeepSpeed启动多GPU训练，明确指定GPU设备
deepspeed learning/ds_train.py  