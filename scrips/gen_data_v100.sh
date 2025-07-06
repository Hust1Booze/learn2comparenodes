#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=v100
#SBATCH --qos=dcgpu
#SBATCH -J data_gen_cpu              # 作业名
#SBATCH --nodes=1                    # 申请1个节点
#SBATCH --ntasks=1                   # 启动1个任务（一般只要1）
#SBATCH --cpus-per-task=100           # 每个任务使用的CPU核心数（可以根据你需要调整）
#SBATCH --gres=gpu:0                 # 不申请GPU

# this shell use CPU and just validate one checkpoint
source activate bnb

root_dir=$(pwd)
echo "root_dir:"${root_dir}

lscpu

# python problem_generation/gisp.py -data_partition train -n_instance 1000 -n_cpu 20
# python problem_generation/gisp.py -data_partition valid -n_instance 100 -n_cpu 20

# python problem_generation/solve.py -data_partition train -n_cpu 32
# python problem_generation/solve.py -data_partition valid -n_cpu 20
python node_selection/behaviour_gen.py -n_cpu 100 -problem SETCOVER -n_instance -1 |tee logs/gen_setcover.txt

# python node_selection/behaviour_gen.py |tee logs/gen_new_data_2.txt

# python node_selection/behaviour_gen.py |tee logs/gen_new_data_3.txt

# python node_selection/behaviour_gen.py |tee logs/gen_new_data_4.txt

# python learning/dt_train.py |tee logs/train_log.txt