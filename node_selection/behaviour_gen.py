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
from pyscipopt import SCIP_EVENTTYPE,Eventhdlr,SCIP_RESULT
from pathlib import Path 
from functools import partial
from node_selectors import OracleNodeSelectorAbdel
from recorders import LPFeatureRecorder, CompFeaturizer, CompFeaturizerSVM
from torch.multiprocessing import Process, set_start_method
import torch
import time

class Brancher(sp.Branchrule):

    def __init__(self):
        super().__init__()


    def branchexeclp(self, allowaddcons):

        cands, scores, npriocands, bestcand = self.model.getVanillafullstrongData()
        self.model.branchVar(cands[bestcand])
        result = SCIP_RESULT.BRANCHED

        return {"result": result}

class StrongBranchingRule(sp.Branchrule):

    def __init__(self, scip, save_dir):
        self.scip = scip
        self.save_dir = save_dir

        varrs = self.scip.getVars() # equal to variables nums in bipartite graph representation
        original_conss = self.scip.getConss()
        self.varrs = varrs
        self.original_conss = original_conss
        self.var2idx = dict([ (str_var, idx) for idx, var in enumerate(self.varrs) for str_var in [str(var)]  ])

    def branchexeclp(self, allowaddcons):

        branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = self.scip.getLPBranchCands()

        # Initialise scores for each variable
        scores = [-self.scip.infinity() for _ in range(npriocands)]
        down_bounds = [None for _ in range(npriocands)]
        up_bounds = [None for _ in range(npriocands)]

        # Initialise placeholder values
        num_nodes = self.scip.getNNodes()
        lpobjval = self.scip.getLPObjVal()
        lperror = False
        best_cand_idx = 0

        # Start strong branching and iterate over the branching candidates
        self.scip.startStrongbranch()
        for i in range(npriocands):

            # Check the case that the variable has already been strong branched on at this node.
            # This case occurs when events happen in the node that should be handled immediately.
            # When processing the node again (because the event did not remove it), there's no need to duplicate work.
            if self.scip.getVarStrongbranchNode(branch_cands[i]) == num_nodes:
                down, up, downvalid, upvalid, _, lastlpobjval = self.scip.getVarStrongbranchLast(branch_cands[i])
                if downvalid:
                    down_bounds[i] = down
                if upvalid:
                    up_bounds[i] = up
                downgain = max([down - lastlpobjval, 0])
                upgain = max([up - lastlpobjval, 0])
                scores[i] = self.scip.getBranchScoreMultiple(branch_cands[i], [downgain, upgain])
                continue

            # Strong branch!
            down, up, downvalid, upvalid, downinf, upinf, downconflict, upconflict, lperror = self.scip.getVarStrongbranch(
                branch_cands[i], 200, idempotent=False)

            # In the case of an LP error handle appropriately (for this example we just break the loop)
            if lperror:
                break

            # In the case of both infeasible sub-problems cutoff the node
            if downinf and upinf:
                return {"result": SCIP_RESULT.CUTOFF}

            # Calculate the gains for each up and down node that strong branching explored
            if not downinf and downvalid:
                down_bounds[i] = down
                downgain = max([down - lpobjval, 0])
            else:
                downgain = 0
            if not upinf and upvalid:
                up_bounds[i] = up
                upgain = max([up - lpobjval, 0])
            else:
                upgain = 0

            # Update the pseudo-costs
            lpsol = branch_cands[i].getLPSol()
            if not downinf and downvalid:
                self.scip.updateVarPseudocost(branch_cands[i], -self.scip.frac(lpsol), downgain, 1)
            if not upinf and upvalid:
                self.scip.updateVarPseudocost(branch_cands[i], 1 - self.scip.frac(lpsol), upgain, 1)

            scores[i] = self.scip.getBranchScoreMultiple(branch_cands[i], [downgain, upgain])
            if scores[i] > scores[best_cand_idx]:
                best_cand_idx = i

        # End strong branching
        self.scip.endStrongbranch()

        # In the case of an LP error
        if lperror:
            return {"result": SCIP_RESULT.DIDNOTRUN}

        # 10%的概率随机选择分支变量，90%的概率选择得分最高的变量
        if random.random() < 0.1:
            # 随机选择
            best_cand_idx = random.randint(0, npriocands - 1)
        else:
            # 选择得分最高的变量（保持原逻辑）
            pass  # best_cand_idx 已经在上面的循环中确定

        # Branch on the variable with the largest score
        down_child, eq_child, up_child = self.model.branchVarVal(
            branch_cands[best_cand_idx], branch_cands[best_cand_idx].getLPSol())

        # Update the bounds of the down node and up node. Some cols might not exist due to pricing
        if self.scip.allColsInLP():
            if down_child is not None and down_bounds[best_cand_idx] is not None:
                self.scip.updateNodeLowerbound(down_child, down_bounds[best_cand_idx])
            if up_child is not None and up_bounds[best_cand_idx] is not None:
                self.scip.updateNodeLowerbound(up_child, up_bounds[best_cand_idx])

        node_number = self.model.getCurrentNode().getNumber()
        cands_indexs = []
        current_time = time.time()
        file_path = self.save_dir + f"/{current_time:.4f}_branchinfo_{node_number}.pt"
        for i in range(npriocands):
            var = str(branch_cands[i])
            if var in self.var2idx:
                _var_idx = self.var2idx[var]
            elif var.startswith("t_") and var[2:] in self.var2idx:
                _var_idx = self.var2idx[var[2:]]
            else:
                print("error in save branch_cands info")
            cands_indexs.append(_var_idx) 
        info = {
            "candidate_indices": cands_indexs,
            "scores": scores,
            "selected_var_index": best_cand_idx
        }
        # print(f'branch on the node {node_number} and  var {branch_cands[best_cand_idx]}')
        torch.save(info, file_path)
        return {"result": SCIP_RESULT.BRANCHED}

class OracleNodeSelRecorder(OracleNodeSelectorAbdel):
    
    def __init__(self, oracle_type, comp_behaviour_saver, comp_behaviour_saver_svm,save_dir):
        super().__init__(oracle_type)
        self.counter = 0
        self.comp_behaviour_saver = comp_behaviour_saver
        self.comp_behaviour_saver_svm = comp_behaviour_saver_svm
        self.save_dir = save_dir

    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.comp_behaviour_saver.set_LP_feature_recorder(LP_feature_recorder)

    def nodeselect(self):
        leaves, children, siblings = self.model.getOpenNodes()
        open_nodes = set(leaves + children + siblings)
        if len(open_nodes)==0:
            #print("no open nodes", len(open_nodes))
            return {"selnode":self.model.getBestNode()}

        # selnode = self.model.getPrioChild()
        # if selnode is None:
        #     selnode = self.model.getPrioSibling()
        #     if selnode is None:
        #         selnode = self.model.getBestLeaf()

        # return {"selnode": selnode}
        select_node = super().nodeselect()
        select_node_number = select_node['selnode'].getNumber()
        if select_node_number == 1:
            gpu_gpu, g = self.comp_behaviour_saver.get_graph_for_inf(self.model, select_node['selnode'])
            current_time = time.time()
            file_path = self.save_dir + f"/{current_time:.4f}_origin_milp.pt"
            torch.save(g, file_path)
        leaves, children, siblings = self.model.getOpenNodes()
        open_nodes = leaves + children + siblings

        open_nodes_number = []
        for open_node in open_nodes:
            open_nodes_number.append(open_node.getNumber())
        if not os.path.exists(self.save_dir) :
            os.makedirs(self.save_dir , exist_ok=True)
        current_time = time.time()

        file_path = self.save_dir + f"/{current_time:.4f}_node_{select_node_number}_selected.pt"
        # print(f"Node: {select_node_number} , !!!!select node.")
        torch.save(open_nodes_number, file_path)
 
        return select_node        
        
    def nodecomp(self, node1, node2):
        comp_res, comp_type = super().nodecomp(node1, node2, return_type=True)
        self.counter += 1
        return comp_res
        if comp_type in [-1,1]:
            self.comp_behaviour_saver.save_comp(self.model, 
                                                node1, 
                                                node2,
                                                comp_res,
                                                self.counter) 
            
            self.comp_behaviour_saver_svm.save_comp(self.model, 
                                                node1, 
                                                node2,
                                                comp_res,
                                                self.counter) 
        
            #print("saved comp # " + str(self.counter))
            self.counter += 1
        
        #make it bad to generate more data !
        if comp_type in [-1,1]:
            comp_res = -1 if comp_res == 1 else 1
        else:
            comp_res = 0
            
        
class InfeasibleCounter(Eventhdlr):
    def __init__(self, model, save_dir,device):
        Eventhdlr.__init__(model)
        self.count = 0
        self.model = model
        self.save_dir = save_dir
        self.device = device
        varrs = model.getVars() # equal to variables nums in bipartite graph representation
        original_conss = model.getConss()
        self.varrs = varrs
        self.original_conss = original_conss
        self.var2idx = dict([ (str_var, idx) for idx, var in enumerate(self.varrs) for str_var in [str(var)]  ])
        # print(f'the total vars {self.var2idx}')
        if not os.path.exists(self.save_dir) :
            os.makedirs(self.save_dir , exist_ok=True)

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
        self.model.catchEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)

    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.NODESOLVED, self)
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
        self.model.catchEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)

    def eventexec(self, event):
        self.count += 1
        node = self.model.getCurrentNode()
        node_number = node.getNumber()
        # print(f"Node: {node_number} , {event.getName()}")

        if(event.getName() == 'BESTSOLFOUND'):
            current_time = time.time()

            file_path = self.save_dir + f"/{current_time:.4f}_bestsolfound.pt"
            data = {}
            torch.save(data, file_path)
        if(event.getName() == 'NODEINFEASIBLE'):
            current_time = time.time()

            file_path = self.save_dir + f"/{current_time:.4f}_nodeinfeasible.pt"
            data = {}
            torch.save(data, file_path)
        if(event.getName() == 'NODEFOCUSED'):
            pass
        if(event.getName() == 'NODEBRANCHED'):
            leaves, children, siblings = self.model.getOpenNodes()
            open_nodes = leaves + children + siblings
            open_nodes_number = []
            for open_node in open_nodes:
                open_nodes_number.append(open_node.getNumber())
                if open_node.getParent().getNumber() == node_number:
                    child_number = open_node.getNumber()
                    # print(f'chile node {child_number}')
                    lb, ub = node.getLowerbound(), node.getEstimate()
                    depth = node.getDepth()
                    bvars, bbounds, btypes = open_node.getParentBranchings()
                    for bvar, bbound, btype in zip(bvars, bbounds, btypes): 
                        if str(bvar) in self.var2idx:
                            var_idx = self.var2idx[str(bvar)]
                        elif 't_'+str(bvar) in self.var2idx:
                            var_idx = self.var2idx['t_' + str(bvar)]
                        else:
                            var_idx = self.var2idx[ '_'.join(str(bvar).split('_')[1:]) ] 
                    child_node = torch.tensor([[lb, -1*ub,depth,node_number,child_number,var_idx,bbound,btype]], device=self.device).float()
                    current_time = time.time()
                    file_path = self.save_dir + f"/{current_time:.4f}_branch_on_{node_number}_to_{child_number}.pt"
                    torch.save(child_node, file_path)
        



def run_episode(oracle_type, instance,  save_dir, save_dir_svm, device,debug_model):
    
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
    
    
    optsol = model.readSolFile(instance.replace(".lp", ".sol"))

    save_dir = save_dir + str(instance).split("/")[-1] + f"_{int(time.time())}"

    comp_behaviour_saver = CompFeaturizer(f"{save_dir}", instance_name=str(instance).split("/")[-1])
    comp_behaviour_saver_svm = CompFeaturizerSVM(model, f"{save_dir_svm}", instance_name=str(instance).split("/")[-1])
    
    oracle_ns = OracleNodeSelRecorder(oracle_type, comp_behaviour_saver, comp_behaviour_saver_svm,save_dir)
    oracle_ns.setOptsol(optsol)
    oracle_ns.set_LP_feature_recorder(LPFeatureRecorder(model, device))
        
    
    model.includeNodesel(oracle_ns, "oracle_recorder", "testing",
                         536870911,  536870911)
    
    infeasible_Counter = InfeasibleCounter(model,save_dir,device)
    model.includeEventhdlr(infeasible_Counter, "infeasible_Counter", "Event handler when nodes are pouned")

    brancher = StrongBranchingRule(model,save_dir)
    model.includeBranchrule(
    branchrule=brancher,
    name="BNB_Brancher",
    desc="custom BNB_Brancher",
    priority=666666, maxdepth=-1, maxbounddist=1)

    # Run the optimizer
    model.optimize()
    print(f"Got behaviour for instance  "+ str(instance).split("/")[-1] + f' with {oracle_ns.counter} comparisons' )
    
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
        try:
            
            run_episode(oracle_type, instance, save_dir, save_dir_svm, device,debug_model)
            print(f'dealing  done', flush= True)
        except:
            print(f'error in dealing', flush= True)
    
        
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
    data_partitions = ['train'] #dont change
    n_cpu = 8
    n_instance = -1
    device = 'cpu'
    debug_model = 0

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
        
        # n_keep  = n_instance if data_partition == 'train' or n_instance == -1 else int(0.2*n_instance)
        n_keep = n_instance
        instances = list(Path(os.path.join(os.path.dirname(__file__), 
                                           f"../problem_generation/data/{problem}/{data_partition}")).glob("*.lp"))
        # random.shuffle(instances)
        # instances = instances[:n_keep]
        
        print(f"Generating {data_partition} samples from {len(instances)} instances using oracle {oracle}", flush= True)
        
        run_episodes(oracle_type=oracle,
                    instances=instances, 
                    save_dir=save_dir,
                    save_dir_svm=save_dir_svm,
                    device=device,
                    debug_model=debug_model)
      
        # processes = [  Process(name=f"worker {p}", 
        #                                 target=partial(run_episodes,
        #                                                 oracle_type=oracle,
        #                                                 instances=instances[ p1 : p2], 
        #                                                 save_dir=save_dir,
        #                                                 save_dir_svm=save_dir_svm,
        #                                                 device=device,
        #                                                 debug_model=debug_model))
        #                 for p,(p1,p2) in enumerate(distribute(len(instances), n_cpu))]
        
        
        # try:
        #     set_start_method('spawn')
        # except RuntimeError:
        #     ''
            
        # a = list(map(lambda p: p.start(), processes)) #run processes
        # b = list(map(lambda p: p.join(), processes)) #join processes
        
            
    nnodes = np.genfromtxt("nnodes.csv", delimiter=",")[:-1]
    times = np.genfromtxt("times.csv", delimiter=",")[:-1]
        
    print(f"Mean number of node created  {np.mean(nnodes)}")
    print(f"Mean solving time  {np.mean(times)}")
    print(f"Median number of node created  {np.median(nnodes)}")
    print(f"Median solving time  {np.median(times)}")
    
    
                         
            
        

        
