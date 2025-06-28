#!/bin/bash
#SBATCH -o job.%j.out                 # 输出文件名（%j表示job id）
#SBATCH --partition=titan            # 分区名
#SBATCH --qos=titan                  # QoS（服务等级）
#SBATCH -J dt_bnb              # 作业名
#SBATCH --nodes=1                    # 申请1个节点
#SBATCH --ntasks-per-node=4         # 启动1个任务（一般只要1）
#SBATCH --gres=gpu:4                 # 

nvidia-smi
# module load cuda/11.8
source activate bnb

root_dir=$(pwd)
echo "root_dir:"${root_dir}

# 设置环境变量以解决GCC版本兼容性问题
export TORCH_CUDA_ARCH_LIST="8.0"
export CUDA_HOME=/lab/cuda/cuda-11.8
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"

# python node_selection/behaviour_gen.py -n_cpu 20

# 使用DeepSpeed启动多GPU训练，明确指定GPU设备
deepspeed learning/dt_train_ds.py  
