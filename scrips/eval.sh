#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=cpu
#SBATCH -J eval
#SBATCH -n 1                 # 总核数 40
#SBATCH --ntasks-per-node=20   # 每节点核数
#SBATCH --qos=cpu     

python learning/eval.py -debug_model 2  -n_cpu 10|tee logs/dt_solve_only_selector.txt
python learning/eval.py -debug_model 4  -n_cpu 10|tee logs/dt_solve_none.txt