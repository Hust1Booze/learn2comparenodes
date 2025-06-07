#!/bin/bash

python node_selection/behaviour_gen.py |tee logs/gen_new_data_1.txt

python node_selection/behaviour_gen.py |tee logs/gen_new_data_2.txt

python node_selection/behaviour_gen.py |tee logs/gen_new_data_3.txt

python node_selection/behaviour_gen.py |tee logs/gen_new_data_4.txt

python learning/dt_train.py |tee logs/train_log.txt