#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:26:18 2021

@author: abdel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:54:57 2021

@author: abdel

"""


import os
import sys
import random
import numpy as np
import pyscipopt.scip as sp
from pyscipopt import SCIP_EVENTTYPE,Eventhdlr
from pathlib import Path 
from functools import partial
from torch.multiprocessing import Process, set_start_method
import torch
import time
from pyscipopt.scip import Nodesel
from dt_model import DTModel
import re
from recorders_debug import LPFeatureRecorder,CompFeaturizer


from bnb_utiles import*



def run_episode(oracle_type, instance,  save_dir, save_dir_svm, device, debug_model = 0):
    
    model = sp.Model()
    model.hideOutput()
    
    
    #Setting up oracle selector
    instance = str(instance)
    model.readProblem(instance)
    model.setParam('constraints/linear/upgrade/logicor',0)
    model.setParam('constraints/linear/upgrade/indicator',0)
    model.setParam('constraints/linear/upgrade/knapsack', 0)
    model.setParam('constraints/linear/upgrade/setppc', 0)
    model.setParam('constraints/linear/upgrade/xor', 0)
    model.setParam('constraints/linear/upgrade/varbound', 0)
    
    
    optsol = model.readSolFile(instance.replace(".lp", ".sol"))

    save_dir = save_dir + str(instance).split("/")[-1]

    comb_model = DTModel()
    comb_model.load_state_dict(torch.load("models/best_model_2025-06-26_12-55-29.pth"))
    comb_model.eval()

    comp_behaviour_saver = CompFeaturizer(f"{save_dir}", instance_name=str(instance).split("/")[-1])

    bnbstates = BNB_States(comb_model, device)
    selector = BNB_Node_Selector(comb_model, bnbstates, device, comp_behaviour_saver)    
    selector.set_LP_feature_recorder(LPFeatureRecorder(model, device))
    #selector.setOptsol(optsol)
    state_trigger = BNB_State_Trigger(model,bnbstates,save_dir,device)
    brancher = BNB_Brancher(comb_model, bnbstates, device)


    if debug_model != 0:
        state_trigger.debug = True
        brancher.debug = True
        selector.debug = True

    if debug_model == 1: # selector + brancher
        pass
    elif debug_model == 2: #only selector
        brancher.default_brancher = True 
    elif debug_model ==3: # only brancher
        selector.default_selector = True
    elif debug_model ==4: #do nothing
        brancher.default_brancher = True 
        selector.default_selector = True

    # model.includeBranchrule(
    #     branchrule=brancher,
    #     name="BNB_Brancher",
    #     desc="custom BNB_Brancher",
    #     priority=666666, maxdepth=-1, maxbounddist=1)
    model.includeNodesel(selector, "BNB_Node_Selector", "custrom node selector",
                        536870911,  536870911)
    model.includeEventhdlr(state_trigger, "state_trigger", "Event handler when nodes are pouned")  
    # Run the optimizer
    model.optimize()

    # if brancher.debug == True:
    #     branch_correct_rate = brancher.branch_correct/ brancher.step
    #     print(f"Brancher correct rate : {branch_correct_rate} for " + str(instance).split("/")[-1])
    print(f"Got behaviour for instance with debug_model: {debug_model}  "+ str(instance).split("/")[-1])
    
    with open("nnodes.csv", "a+") as f:
        f.write(f"{model.getNNodes()},")
        f.close()
    with open("times.csv", "a+") as f:
        f.write(f"{model.getSolvingTime()},")
        f.close()
        
    return 1


def run_episodes(oracle_type, instances, save_dir, save_dir_svm, device, debug_model):
    
    for instance in instances:
        run_episode(oracle_type, instance, save_dir, save_dir_svm, device, debug_model)
        
    print("finished running episodes for process")
        
    return 1
    
def distribute(n_instance, n_cpu):
    if n_cpu == 1:
        return [(0, n_instance)]
    
    k = n_instance //( n_cpu -1 )
    r = n_instance % (n_cpu - 1 )
    res = []
    for i in range(n_cpu -1):
        res.append( ((k*i), (k*(i+1))) )
    
    res.append(((n_cpu - 1) *k ,(n_cpu - 1) *k + r ))
    return res


if __name__ == "__main__":
    
    oracle = 'optimal_plunger'
    problem = 'GISP'
    data_partitions = ['valid'] #dont change
    n_cpu = 1
    n_instance = -1
    device = 'cpu'
    debug_model = 3 # 0 : no_debug; 1: dt; 2:selector_only; 3:brancher_only; 4:no decisions

    with open("nnodes.csv", "w") as f:
        f.write("")
        f.close()
    with open("times.csv", "w") as f:
        f.write("")
        f.close()
        
    
    #Initializing the model 
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-oracle':
            oracle = str(sys.argv[i + 1])
        if sys.argv[i] == '-problem':
            problem = str(sys.argv[i + 1])
        if sys.argv[i] == '-n_cpu':
            n_cpu = int(sys.argv[i + 1])
        if sys.argv[i] == '-n_instance':
            n_instance = int(sys.argv[i + 1])
        if sys.argv[i] == '-device':
            device = str(sys.argv[i + 1])
        if sys.argv[i] == '-debug_model':
            debug_model = int(sys.argv[i + 1])
   
    if debug_model ==0:
        print('normal eval')
    elif debug_model ==1:
        print('debug model')
    elif debug_model ==2:
        print('debug model and selector only')
    elif debug_model ==3:
        print('debug model and brancher only')
    elif debug_model ==4:
        print('debug model and do nothing')

    for data_partition in data_partitions:
        

        save_dir = os.path.join(os.path.dirname(__file__), f'./data/{problem}/{data_partition}')
        save_dir_svm = os.path.join(os.path.dirname(__file__), f'./data_svm/{problem}/{data_partition}')
        
        # try:
        #     os.makedirs(save_dir)
        # except FileExistsError:
        #     ""
            
        # try:
        #     os.makedirs(save_dir_svm)
        # except FileExistsError:
        #     ""
        
        #n_keep  = n_instance if data_partition == 'train' or n_instance == -1 else int(0.2*n_instance)
        n_keep = -1

        instances = list(Path(os.path.join(os.path.dirname(__file__), 
                                           f"../problem_generation/data/{problem}/{data_partition}")).glob("*.lp"))
        random.shuffle(instances)
        instances = instances[:n_instance]
        
        print(f"Eval {data_partition} samples from {len(instances)} instances using oracle {oracle}")
        
      
        processes = [  Process(name=f"worker {p}", 
                                        target=partial(run_episodes,
                                                        oracle_type=oracle,
                                                        instances=instances[ p1 : p2], 
                                                        save_dir=save_dir,
                                                        save_dir_svm=save_dir_svm,
                                                        device=device,
                                                        debug_model=debug_model))
                        for p,(p1,p2) in enumerate(distribute(len(instances), n_cpu))]
        
        
        try:
            set_start_method('spawn')
        except RuntimeError:
            ''
            
        a = list(map(lambda p: p.start(), processes)) #run processes
        b = list(map(lambda p: p.join(), processes)) #join processes
        
            
    nnodes = np.genfromtxt("nnodes.csv", delimiter=",")[:-1]
    times = np.genfromtxt("times.csv", delimiter=",")[:-1]
        
    print(f"Mean number of node created  {np.mean(nnodes)}")
    print(f"Mean solving time  {np.mean(times)}")
    print(f"Median number of node created  {np.median(nnodes)}")
    print(f"Median solving time  {np.median(times)}")
    
    
                         
            
        

        
