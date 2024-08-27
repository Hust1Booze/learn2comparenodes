#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=a100
#SBATCH -J learn2compare
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --qos=a100

source activate l2sn

root_dir=$(pwd)
echo "root_dir:"${root_dir}


python learning/train.py


