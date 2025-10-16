#!/bin/bash
#SBATCH -o node_selection/job.%j.out
#SBATCH -J gen_data
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20


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