#!/bin/bash

python node_selection/behaviour_gen.py |tee logs/gen_setcover.txt

python learning/dt_train.py |tee logs/dt_train_setcover.txt
