import sys
import os
import re
import numpy as np
import torch
from torch.multiprocessing import Process, set_start_method
from functools import partial
from utils import record_stats, display_stats, distribute
from pathlib import Path 

    
n_cpu = 16
n_instance = -1
nodesels =  ['expert_dummy', 'gnn_dummy_nprimal=2', 'ranknet_dummy_nprimal=2', 'svm_dummy_nprimal=2', 'estimate_dummy']
nodesels =  ['expert_dummy', 'estimate_dummy']
problem = 'WPMS'
data_partition = 'transfer'

normalize = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
verbose = True
on_log = False
default = True

instances =  list(Path(os.path.join(os.path.abspath(''), 
                           f"./problem_generation/data/{problem}/{data_partition}")).glob("*.lp"))

print(len(instances))


processes = [  Process(name=f"worker {p}", 
               target=partial(record_stats,
                              nodesels=nodesels,
                              instances=instances[p1:p2], 
                              problem=problem,
                              device=torch.device(device),
                              normalize=normalize,
                              verbose=verbose,
                              default=default))
        for p,(p1,p2) in enumerate(distribute(n_instance, n_cpu)) ]  


try:
    set_start_method('spawn')
except RuntimeError:
    ''

a = list(map(lambda p: p.start(), processes)) #run processes
b = list(map(lambda p: p.join(), processes)) #join processes