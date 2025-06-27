#!/bin/bash


# python problem_generation/solve.py -data_partition train |tee logs/solve_train.txt
# python problem_generation/solve.py -data_partition valid |tee logs/solve_valid.txt


python node_selection/behaviour_gen.py |tee logs/gen_setcover.txt

python learning/dt_train.py |tee logs/dt_train_setcover.txt
