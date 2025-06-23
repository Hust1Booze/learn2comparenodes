#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=cpu
#SBATCH -J gen_data
#SBATCH -n 40                 # 总核数 40
#SBATCH --ntasks-per-node=40   # 每节点核数
#SBATCH --qos=cpu     


# this shell use CPU and just validate one checkpoint
source activate bnb

root_dir=$(pwd)
echo "root_dir:"${root_dir}


# python problem_generation/gisp.py |tee logs/gen_problem.txt

python node_selection/behaviour_gen.py |tee logs/gen_new_data.txt

# python node_selection/behaviour_gen.py |tee logs/gen_new_data_2.txt

# python node_selection/behaviour_gen.py |tee logs/gen_new_data_3.txt

# python node_selection/behaviour_gen.py |tee logs/gen_new_data_4.txt

# python learning/dt_train.py |tee logs/train_log.txt