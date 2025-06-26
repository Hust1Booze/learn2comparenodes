#!/bin/bash

python learning/dt_eval.py -debug_model 3 |tee logs/dt_solve_only_brancher.txt
python learning/dt_eval.py -debug_model 4 |tee logs/dt_solve_none.txt