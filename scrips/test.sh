#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH --partition=a100
#SBATCH -J dt_bnb
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --qos=a100

nvidia-smi

# find / -name nvcc 2>/dev/null

which nvcc
nvcc -V

module spider GCC


module spider gcc



