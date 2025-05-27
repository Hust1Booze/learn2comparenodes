#!/bin/bash

python node_selection/behaviour_gen.py |tee logs/scip_solve_one.txt

python learning/dt_eval.py -debug_model 1 |tee logs/dt_solve_one.txt
python learning/dt_eval.py -debug_model 2 |tee logs/dt_solve_one_only_selector.txt
python learning/dt_eval.py -debug_model 3 |tee logs/dt_solve_one_only_brancher.txt
python learning/dt_eval.py -debug_model 4 |tee logs/dt_solve_one_none.txt