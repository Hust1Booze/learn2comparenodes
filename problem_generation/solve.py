import os
import sys
import networkx as nx
import random
import pyscipopt as sp
import numpy as np
import multiprocessing as md
from functools import partial
from pathlib import Path 
from torch.multiprocessing import Process, set_start_method

seed = 0
def generate_instances(instances, save_dir) :
    random.seed(seed)
    for instance in instances:
        model = sp.Model()
        model.hideOutput()
        model.readProblem(instance)
        model.optimize()
        
        # 将 .lp 扩展名替换为 .sol
        instance_name = instance.stem  # 获取不带扩展名的文件名
        sol_file = os.path.join(save_dir, f"{instance_name}.sol")
        model.writeBestSol(sol_file)

        print(f"Solve instance  "+ str(instance).split("/")[-1] + f' with {model.getNNodes()} nodes, {model.getSolvingTime()} time' )
        
        with open("nnodes.csv", "a+") as f:
            f.write(f"{model.getNNodes()},")
            f.close()
        with open("times.csv", "a+") as f:
            f.write(f"{model.getSolvingTime()},")
            f.close()
        

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
    instance = None
    n_cpu = 20
    data_partition = 'train'
    problem = 'facilities'
    timelimit = 7200.0
    solveInstance = True
    n_instance = 10000
    seed = 0
    

    # seed = 0
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-instance':
            instance = sys.argv[i + 1]
        if sys.argv[i] == '-data_partition':
            data_partition = sys.argv[i + 1]
        if sys.argv[i] == '-min_n':
            min_n = int(sys.argv[i + 1])
        if sys.argv[i] == '-max_n':
            max_n = int(sys.argv[i + 1])
        if sys.argv[i] == '-er_prob':
            er_prob = float(sys.argv[i + 1])
        if sys.argv[i] == '-whichSet':
            whichSet = sys.argv[i + 1]
        if sys.argv[i] == '-setParam':
            setParam = float(sys.argv[i + 1])
        if sys.argv[i] == '-alphaE2':
            alphaE2 = float(sys.argv[i + 1])
        if sys.argv[i] == '-timelimit':
            timelimit = float(sys.argv[i + 1])
        if sys.argv[i] == '-solve':
            solveInstance = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-seed_start':
            seed = int(sys.argv[i + 1])
        if sys.argv[i] == '-n_instance':
            n_instance = int(sys.argv[i + 1])
        if sys.argv[i] == '-n_cpu':
            n_cpu = int(sys.argv[i + 1])
    
    print("Summary for generation")
    print(f"n_instance    :     {n_instance}")
    print(f"n_cpu         :     {n_cpu} ")
    print(f"solve         :     {solveInstance}")
    
    with open("nnodes.csv", "w") as f:
        f.write("")
        f.close()
    with open("times.csv", "w") as f:
        f.write("")
        f.close()
            
    cpu_count = md.cpu_count()//2 if n_cpu == None else n_cpu
    
    n_keep = n_instance
    instances = list(Path(os.path.join(os.path.dirname(__file__), 
                                        f"./data/{problem}/{data_partition}")).glob("*.lp"))
    # random.shuffle(instances)
    instances = instances[:n_keep]

    save_dir = os.path.join(os.path.dirname(__file__), f'./data/{problem}/{data_partition}')
    
    processes = [ Process(name=f"worker {p}", 
                                target=partial(generate_instances,
                                                instances=instances[ p1 : p2], 
                                                save_dir=save_dir))
                for p,(p1,p2) in enumerate(distribute(len(instances), n_cpu))]
    
 
    try:
        set_start_method('spawn')
    except RuntimeError:
        ''
        
    a = list(map(lambda p: p.start(), processes)) #run processes
    b = list(map(lambda p: p.join(), processes)) #join processes
    print('Solved')
 
    nnodes = np.genfromtxt("nnodes.csv", delimiter=",")[:-1]
    times = np.genfromtxt("times.csv", delimiter=",")[:-1]
        
    print(f"Mean number of node created  {np.mean(nnodes)}")
    print(f"Mean solving time  {np.mean(times)}")
    print(f"Median number of node created  {np.median(nnodes)}")
    print(f"Median solving time  {np.median(times)}")

    
    
            
        

