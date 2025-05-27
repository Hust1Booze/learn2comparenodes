

import os
import sys
import random
import numpy as np
import pyscipopt.scip as sp
from pyscipopt import SCIP_EVENTTYPE,Eventhdlr,SCIP_RESULT
from pathlib import Path 
from functools import partial
from torch.multiprocessing import Process, set_start_method
import torch
import time
from pyscipopt.scip import Nodesel

from dt_model import DTModel
import re
from recorders_debug import LPFeatureRecorder,CompFeaturizer


# this class use to save Branch and bound sequence states, and output dt models input
class BNB_States():
    def __init__(self, comb_model, device, init_reward = 1000): #for gisp avg_reward = 700
        self.sequence = [] # [states, selected node, reward, branch, node, node, selected node ...]
        self.type_ids = []
        self.actions = []  
        self.candidates = []

        self.comb_model  = comb_model
        self.device = device
        self.total_reward = init_reward # best

        self.state_embedded = False

        self.var2idx = None
    def receive_var2idx(self,var2idx):
        self.var2idx = var2idx

    def receive_origin_milp(self, graph):
        gnn_input = graph
        gnn_input = [x.to(self.device) for x in gnn_input]
        state_emb = self.comb_model.gnn_encoder(*gnn_input)  # Tensor [D]
        self.sequence.append(state_emb)
        state_length = state_emb.shape[0]
        self.type_ids.extend([torch.tensor([1])] * state_emb.size(0))  # type 1 = state
        self.actions.extend([torch.tensor([-1])] * state_emb.size(0))
        self.candidates.extend([torch.tensor([-1])] * state_emb.size(0))
        reward_embd = self.comb_model.reward_embedding(torch.tensor([self.total_reward], dtype=torch.float32,device=self.device))
        self.sequence.append(reward_embd.unsqueeze(0))
        self.type_ids.append(torch.tensor([3]))  # type 3 = reward      
        self.actions.append(torch.tensor([-1],dtype=torch.long,device=self.device))       
        self.candidates.append(torch.tensor([-1],dtype=torch.long,device=self.device)) 

        self.state_embedded = True

    def get_dt_input(self):
        sequence_tensor = torch.stack(self.sequence) if self.sequence[0].dim() == 1 else torch.cat(self.sequence).view(len(self.type_ids), -1)
        type_ids_tensor = torch.tensor(self.type_ids, dtype=torch.long)
        actions_tensor = torch.tensor(self.actions, dtype=torch.long)

        return sequence_tensor, type_ids_tensor, self.candidates, actions_tensor
    
    def get_sel_mask(self):
        return None
    def get_branch_mask(self):
        return None

    def receive_states(self, state, info = None):
        if self.state_embedded is False:
            return
        if state =='BESTSOLFOUND':
            self.total_reward -= 100
            reward_embd = self.comb_model.reward_embedding(torch.tensor([self.total_reward], dtype=torch.float32,device=self.device))
            self.sequence[-1] = reward_embd.unsqueeze(0)
        elif state == 'NODEINFEASIBLE':
            self.total_reward  -= 10
            reward_embd = self.comb_model.reward_embedding(torch.tensor([self.total_reward ], dtype=torch.float32,device=self.device))
            self.sequence[-1] = reward_embd.unsqueeze(0)
        elif state == 'NODEBRANCHED_INFO':
            branch_var_idx = info["selected_var_index"]
            cand_var_idx = info["candidate_indices"]

            branch_var_embd = self.comb_model.branch_var_embedding(torch.tensor([branch_var_idx], dtype=torch.long,device=self.device))
            self.sequence.append(branch_var_embd)
            self.type_ids.append(torch.tensor([4]))  # type 4 = branch
            self.actions.append(torch.tensor([branch_var_idx],dtype=torch.long,device=self.device)) 
            self.candidates.append(cand_var_idx) 
            self.total_reward += 1
        elif 'branch_on' in state:
            match = re.search(r'branch_on_\d+_to_(\d+)', state)
            node_idx =int(match.group(1))    
            child_node = info
            child_node_embd = self.comb_model.node_embedding(child_node.to(self.device))
            self.sequence.append(child_node_embd)
            self.type_ids.append(torch.tensor([5]))  # type 5 = node
            self.actions.append(torch.tensor([node_idx])) 
            self.candidates.append(torch.tensor([-1],dtype=torch.long,device=self.device)) 



# Depth First Search Node Selector
class BNB_Node_Selector(Nodesel):
    def __init__(self, comb_model, bnbstates, device, comp_behaviour_saver):
        self.comb_model = comb_model
        #self.comb_model.to(device)
        self.sel_counter = 0
        self.comp_counter = 0
        self.device = device
        self.paths = []
        self.nodes = []
        self.open_nodes = []
        self.gaps = []

        self.added_ids = set()
        self.logit_lookup = torch.zeros(1)
        self.bnbstates  = bnbstates
        self.step = 0

        self.comp_behaviour_saver = comp_behaviour_saver

        self.debug = False
        
        self.default_selector = False

    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.comp_behaviour_saver.set_LP_feature_recorder(LP_feature_recorder)

    @torch.inference_mode()
    def nodeselect(self):
        self.step+=1
        #if self.step>=750:

        self.comb_model.eval()
        leaves, children, siblings = self.model.getOpenNodes()
        open_nodes = set(leaves + children + siblings)

        nodes = sorted(list(filter(lambda x: x.getNumber() not in self.added_ids, open_nodes)), key=lambda node: node.getNumber())

        if len(open_nodes)==0:
            if self.debug :
                print("no open nodes", len(open_nodes))
            return {"selnode":self.model.getBestboundNode()}
        if len(open_nodes)==1 and self.step<3:
            if self.debug:
                print("root nodes", len(open_nodes))
            gpu_gpu, g = self.comp_behaviour_saver.get_graph_for_inf(self.model, nodes[0])
            self.bnbstates.receive_origin_milp(g)
            return {"selnode":self.model.getBestboundNode()}
        
        if self.default_selector :
            return {"selnode":self.model.getBestNode()}
        
        if nodes is None:
            print("dumb selection")
            return {"selnode":self.model.getBestboundNode()}
        rank_node_ids = self.comb_model.get_select_node_decision(*self.bnbstates.get_dt_input())
        node = None
        for node_id in rank_node_ids:
            for _node in open_nodes:
                if _node.getNumber() == node_id:
                    node = _node
                    break
            if node is not None:
                break  # 找到就退出外层循环

        return {"selnode": node}
    
    def nodecomp(self, node1, node2):
        #n1 = node1.getNumber()
        #n2 = node2.getNumber()
        #p1 = self.logit_lookup # type: ignore
        #p = p1[n1].exp() / (p1[n1].exp() + p1[n2].exp())
        return -1 if node1.getLowerbound() <= node2.getLowerbound() else 1
        #return -1 if torch.rand(1) < 0.5 else 1


        
class BNB_State_Trigger(Eventhdlr):
    def __init__(self, model, bnbstates, save_dir,device, debug_model = False):
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
        self.bnbstates  = bnbstates

        self.bnbstates.receive_var2idx(self.var2idx)

        self.debug = debug_model

        self.primalbound = 0
        self.dualbound = 0

    def eventinit(self):
        self.model.catchEvent(SCIP_EVENTTYPE.NODESOLVED, self)
        self.model.catchEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
        self.model.catchEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)
        self.model.catchEvent(SCIP_EVENTTYPE.GBDCHANGED, self) 
        self.model.catchEvent(SCIP_EVENTTYPE.BOUNDCHANGED, self)
    def eventexit(self):
        self.model.dropEvent(SCIP_EVENTTYPE.NODESOLVED, self)
        self.model.dropEvent(SCIP_EVENTTYPE.BESTSOLFOUND, self)
        self.model.dropEvent(SCIP_EVENTTYPE.NODEFOCUSED, self)
        self.model.dropEvent(SCIP_EVENTTYPE.GBDCHANGED, self)
        self.model.dropEvent(SCIP_EVENTTYPE.BOUNDCHANGED, self)

    def eventexec(self, event):
        self.count += 1
        node = self.model.getCurrentNode()
        node_number = node.getNumber()
        if self.debug:
            print(f"Node: {node_number} , {event.getName()}")
            primalbound = self.model.getPrimalbound()
            dualbound = self.model.getDualbound()
            if primalbound != self.primalbound or dualbound != self.dualbound:
                self.primalbound = primalbound 
                self.dualbound = dualbound
                print(f'primalbound : {primalbound}, dualbound : {dualbound} ')

        if(event.getName() == 'BESTSOLFOUND'):
            current_time = time.time()

            file_path = self.save_dir + f"/{current_time:.4f}_bestsolfound.pt"
            data = {}
            self.bnbstates.receive_states('BESTSOLFOUND')
        if(event.getName() == 'NODEINFEASIBLE'):
            current_time = time.time()

            file_path = self.save_dir + f"/{current_time:.4f}_nodeinfeasible.pt"
            data = {}
            self.bnbstates.receive_states('NODEINFEASIBLE')
        if(event.getName() == 'NODEFOCUSED'):
            # branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = self.model.getLPBranchCands()
            # last_branch_candidates = [str(cand) for cand in branch_cands]
            pass
        if(event.getName() == 'BOUNDCHANGED'): # this event is no triggered, dont know why
            primalbound = self.model.getPrimalbound()
            dualbound = self.model.getDualbound()
            print(f'BOUNDCHANGED and primalbound : {primalbound}, dualbound : {dualbound} ')
        if(event.getName() == 'NODEBRANCHED'):
            leaves, children, siblings = self.model.getOpenNodes()
            open_nodes = leaves + children + siblings
            branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = self.model.getLPBranchCands()

            save_branch_info = False

            open_nodes_number = []
            for open_node in open_nodes:
                open_nodes_number.append(open_node.getNumber())
                if open_node.getParent().getNumber() == node_number:
                    child_number = open_node.getNumber()
                    if self.debug:
                        print(f'chile node {child_number}')
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
                    
                    if save_branch_info is False:
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
                            "selected_var_index": var_idx
                        }
                        #torch.save(info, file_path)
                        print(f'branch on the node {node_number} and  var {bvar}')
                        self.bnbstates.receive_states('NODEBRANCHED_INFO',info)
                        save_branch_info = True
                    child_node = torch.tensor([[lb, -1*ub,depth,node_number,child_number,var_idx,bbound,btype]], device=self.device).float()
                    current_time = time.time()
                    name = f"/{current_time:.4f}_branch_on_{node_number}_to_{child_number}"
                    file_path = self.save_dir + f"/{current_time:.4f}_branch_on_{node_number}_to_{child_number}.pt"
                    #torch.save(child_node, file_path)
                    self.bnbstates.receive_states(name,child_node)
        


class BNB_Brancher(sp.Branchrule):

    def __init__(self, comb_model, bnbstates, device):
        super().__init__()

        self.comb_model = comb_model
        self.device = device
        self.bnbstates  = bnbstates
        self.step = 0
        self.debug = False
        self.default_brancher = False

    # def branchinitsol(self):
    #     self.ndomchgs = 0
    #     self.ncutoffs = 0
    #     self.state_buffer = {}
    #     self.khalil_root_buffer = {}

    def branchexeclp(self, allowaddcons):

        # candidate_vars, *_ = self.model.getPseudoBranchCands()
        # candidate_mask = [var.getCol().getLPPos() for var in candidate_vars]
        if self.default_brancher:
            result = SCIP_RESULT.DIDNOTRUN
            return {'result': result}
        branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = self.model.getLPBranchCands()
        cands_indexs = []
        for i in range(npriocands):
            var = str(branch_cands[i])
            if var in self.bnbstates.var2idx:
                _var_idx = self.bnbstates.var2idx[var]
            elif var.startswith("t_") and var[2:] in self.bnbstates.var2idx:
                _var_idx = self.bnbstates.var2idx[var[2:]]
            else:
                print("error in save branch_cands info")
            cands_indexs.append(_var_idx) 
        if npriocands == 1:
            best_var = branch_cands[0]

        else:

            var_logits = self.comb_model.get_branch_var_decision(*self.bnbstates.get_dt_input())

            candidate_scores = var_logits[cands_indexs]
            best_var = branch_cands[candidate_scores.argmax()]

            # candidate_scores = var_logits[candidate_mask]
            # best_var = candidate_vars[candidate_scores.argmax()]
            #best_var = candidate_vars[0]


        self.model.branchVar(best_var)
        result = SCIP_RESULT.BRANCHED

        return {'result': result}