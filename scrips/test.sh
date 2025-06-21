#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=titan
#SBATCH -J dt_bnb
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --qos=titan

nvidia-smi

# find / -name nvcc 2>/dev/null




