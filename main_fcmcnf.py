import sys
import os
import re
import numpy as np
import torch
from torch.multiprocessing import Process, set_start_method
from functools import partial
from utils import record_stats, display_stats, distribute
from pathlib import Path 

if __name__ == "__main__":

    n_cpu = 8
    n_instance = -1
    nodesels =  ['expert_dummy', 'gnn_dummy_nprimal=2', 'ranknet_dummy_nprimal=2', 'svm_dummy_nprimal=2', 'estimate_dummy']
    nodesels =  ['expert_dummy', 'estimate_dummy']
    nodesels =  ['gnn_dummy_nprimal=2']
    problem = 'FCMCNF'
    data_partition = 'test' #'transfer'

    normalize = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    verbose = False
    on_log = False
    default = False

    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-n_cpu':
            n_cpu = int(sys.argv[i + 1])
        if sys.argv[i] == '-nodesels':
            nodesels = str(sys.argv[i + 1]).split(',')
        if sys.argv[i] == '-normalize':
            normalize = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-n_instance':
            n_instance = int(sys.argv[i + 1])
        if sys.argv[i] == '-data_partition':
            data_partition = str(sys.argv[i + 1])
        if sys.argv[i] == '-problem':
            problem = str(sys.argv[i + 1])
        if sys.argv[i] == '-device':
            device = str(sys.argv[i + 1])
        if sys.argv[i] == '-verbose':
            verbose = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-on_log':
            on_log = bool(int(sys.argv[i + 1]))    
        if sys.argv[i] == '-default':
            default = bool(int(sys.argv[i + 1]))  
        if sys.argv[i] == '-delete':
            delete = bool(int(sys.argv[i + 1]))  

    instances =  list(Path(os.path.join(os.path.abspath(''), 
                            f"./problem_generation/data/{problem}/{data_partition}")).glob("*.lp"))

    print("Evaluation")
    print(f"  Problem:                    {problem}")
    print(f"  n_instance/problem:         {len(instances)}")
    print(f"  Nodeselectors evaluated:    {','.join( ['default' if default else '' ] + nodesels)}")
    print(f"  Device for GNN inference:   {device}")
    print(f"  Normalize features:         {normalize}")
    print("----------------")


    processes = [  Process(name=f"worker {p}", 
                target=partial(record_stats,
                                nodesels=nodesels,
                                instances=instances[p1:p2], 
                                problem=problem,
                                device=torch.device(device),
                                normalize=normalize,
                                verbose=verbose,
                                default=default,
                                with_root_info = True))
            for p,(p1,p2) in enumerate(distribute(n_instance, n_cpu)) ]  


    try:
        set_start_method('spawn')
    except RuntimeError:
        ''

    a = list(map(lambda p: p.start(), processes)) #run processes
    b = list(map(lambda p: p.join(), processes)) #join processes


    nnode_means = display_stats(problem, nodesels, instances, 0, 0, default=True)
        