#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pyscipopt.scip as sp
from pathlib import Path 
from recorders import LPFeatureRecorder, CompFeaturizer, CompFeaturizerSVM
import time
from brancher import StrongBranchingRule
from selector import OracleNodeSelRecorder, ScipEvent
from saver import SequenceSaver
from torch.multiprocessing import Process, set_start_method
from functools import partial

def run_episode(oracle_type, instance, save_dir, save_dir_svm, device, debug_model):
    
    model = sp.Model()
    model.hideOutput()      
    #model.setIntParam("display/verblevel", 5)  # 设置详细程度为 5（最高）
    
    #Setting up oracle selector
    instance = str(instance)
    model.readProblem(instance)
    model.setParam('constraints/linear/upgrade/logicor',0)
    model.setParam('constraints/linear/upgrade/indicator',0)
    model.setParam('constraints/linear/upgrade/knapsack', 0)
    model.setParam('constraints/linear/upgrade/setppc', 0)
    model.setParam('constraints/linear/upgrade/xor', 0)
    model.setParam('constraints/linear/upgrade/varbound', 0)
    #model.setParam("presolving/maxrounds", 1)
    #model.setParam('limits/restarts',0)
    #model.setParam('limits/autorestartnodes',0)
    #model.setParam("restarts/enabled", False)  # 禁用重启

    model.setParam("presolving/maxrestarts", 0)

    # for p in model.getParams():
    #     if "restart" in p:
    #         print(p)
    
    optsol = model.readSolFile(instance.replace(".lp", ".sol"))

    save_dir = save_dir + '/' + str(instance).split("/")[-1] + f"_{int(time.time())}"
    sequence_saver = SequenceSaver(save_dir)

    comp_behaviour_saver = CompFeaturizer(f"{save_dir}", instance_name=str(instance).split("/")[-1])
    comp_behaviour_saver_svm = CompFeaturizerSVM(model, f"{save_dir_svm}", instance_name=str(instance).split("/")[-1])
    
    oracle_ns = OracleNodeSelRecorder(oracle_type, comp_behaviour_saver, comp_behaviour_saver_svm, sequence_saver, save_dir)
    oracle_ns.setOptsol(optsol)
    oracle_ns.set_LP_feature_recorder(LPFeatureRecorder(model, device))
        
    
    model.includeNodesel(oracle_ns, "oracle_recorder", "testing",
                         536870911,  536870911)
    
    scipEvent = ScipEvent(model,sequence_saver,device)
    model.includeEventhdlr(scipEvent, "ScipEvent", "Event handler when nodes are pouned")

    brancher = StrongBranchingRule(model,sequence_saver,save_dir, use_gasse_representation=True, random_branching_prob=0)
    model.includeBranchrule(
    branchrule=brancher,
    name="BNB_Brancher",
    desc="custom BNB_Brancher",
    priority=666666, maxdepth=-1, maxbounddist=1)

    # Run the optimizer
    model.optimize()
    objval = model.getObjVal()
    sequence_saver.save()

    print(f"Got behaviour for instance  "+ str(instance).split("/")[-1] + f' with {oracle_ns.counter} comparisons, {model.getNNodes()} nodes, {model.getSolvingTime()} time, objval:{objval}' )
    
    with open("nnodes.csv", "a+") as f:
        f.write(f"{model.getNNodes()},")
        f.close()
    with open("times.csv", "a+") as f:
        f.write(f"{model.getSolvingTime()},")
        f.close()
        
    return 1


def run_episodes(oracle_type, instances, save_dir, save_dir_svm, device,debug_model):
    
    for instance in instances:
        print(f'dealing {instance}', flush= True)      
        run_episode(oracle_type, instance, save_dir, save_dir_svm, device,debug_model)
        print(f'done {instance}\n', flush= True)  
        
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
    problem = 'SETCOVER' #'GISP'
    data_partitions = ['train','valid'] #dont change
    n_cpu = 10
    n_instance = 100
    device = 'cpu'
    debug_model = 0
    
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
   
   
  
    for data_partition in data_partitions:
        with open("nnodes.csv", "w") as f:
            f.write("")
            f.close()
        with open("times.csv", "w") as f:
            f.write("")
            f.close()

        save_dir = os.path.join(os.path.dirname(__file__), f'./data/{problem}/{data_partition}')
        save_dir_svm = os.path.join(os.path.dirname(__file__), f'./data_svm/{problem}/{data_partition}')
        
        
        n_keep  = n_instance if data_partition == 'train' or n_instance == -1 else int(0.5*n_instance)
        #n_keep = n_instance
        instances = list(Path(os.path.join(os.path.dirname(__file__), 
                                           f"../problem_generation/data/{problem}/{data_partition}")).glob("*.lp"))
        # random.shuffle(instances)
        instances = instances[:n_keep]
        
        print(f"Generating {data_partition} samples from {len(instances)} instances using oracle {oracle}", flush= True)
        
        # run_episodes(oracle_type=oracle,
        #             instances=instances, 
        #             save_dir=save_dir,
        #             save_dir_svm=save_dir_svm,
        #             device=device,
        #             debug_model=debug_model)
      
        processes = [ Process(name=f"worker {p}", 
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
    
    
                         
            
        

        
