#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=l40s
#SBATCH -J dt_bnb
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --qos=dcgpu


# this shell use CPU and just validate one checkpoint
source activate bnb

root_dir=$(pwd)
echo "root_dir:"${root_dir}


# python problem_generation/gisp.py -data_partition train -n_instance 1000 -n_cpu 20
# python problem_generation/gisp.py -data_partition valid -n_instance 100 -n_cpu 20

python node_selection/behaviour_gen.py -n_cpu 20

# python node_selection/behaviour_gen.py |tee logs/gen_new_data_2.txt

# python node_selection/behaviour_gen.py |tee logs/gen_new_data_3.txt

# python node_selection/behaviour_gen.py |tee logs/gen_new_data_4.txt

# python learning/dt_train.py |tee logs/train_log.txt