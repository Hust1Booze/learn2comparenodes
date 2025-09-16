#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=cpu
#SBATCH -J solve
#SBATCH -n 1             
#SBATCH --cpus-per-task=50   # 每节点核数
#SBATCH --qos=cpu     


# this shell use CPU and just validate one checkpoint
source activate bnb

root_dir=$(pwd)
echo "root_dir:"${root_dir}


python problem_generation/solve.py -data_partition train -n_cpu 40
python problem_generation/solve.py -data_partition valid -n_cpu 40
python node_selection/behaviour_gen.py 
