

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
    def __init__(self, comb_model, device): #for gisp avg_reward = 70
        self.data = []
        self.type = []
        self.node_id = []

        self.comb_model  = comb_model
        self.device = device
        self.state_embedded = False
        self.var2idx = None
        self.state_emb = None

        self.max_data_length = 8
    def receive_var2idx(self,var2idx):
        self.var2idx = var2idx

    def receive_origin_milp(self, graph):
        gnn_input = graph
        gnn_input = [x.to(self.device) for x in gnn_input]
        self.state_emb = self.comb_model.gnn_encoder(*gnn_input)  # Tensor [D]
        self.state_embedded = True

    def get_dt_input(self):
        sequence_tensor = torch.tensor(self.data, dtype=torch.float32) 
        type = torch.tensor(self.type, dtype=torch.int64)
        node_id = torch.tensor(self.node_id, dtype=torch.int64)

        return self.state_emb, sequence_tensor, type, node_id

    def receive_states(self, item):
        self.node_id.append(-1)
        if item['type'] == 'select':
            self.data.append(item["data"] + [0] * (self.max_data_length - len(item["data"])))
            self.type.append(1)
        elif item['type'] == 'branch':
            self.data.append(item["data"] + [0] * (self.max_data_length - len(item["data"])))
            self.type.append(2)
        elif item['type'] == 'node':
            self.data.append(item["data"] + [0] * (self.max_data_length - len(item["data"])))
            self.type.append(3)
            self.node_id[-1] = item["node_number"]


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

        leaves, children, siblings = self.model.getOpenNodes()
        open_nodes = set(leaves + children + siblings)

        nodes = sorted(list(filter(lambda x: x.getNumber() not in self.added_ids, open_nodes)), key=lambda node: node.getNumber())

        if len(open_nodes)==0:
            if self.debug :
                print("no open nodes", len(open_nodes))
            node = self.model.getBestboundNode()
        if len(open_nodes)==1 and self.step<3:
            if self.debug:
                print("root nodes", len(open_nodes))
            gpu_gpu, g = self.comp_behaviour_saver.get_graph_for_inf(self.model, nodes[0])
            self.bnbstates.receive_origin_milp(g)
            node = self.model.getBestboundNode()
        
        if self.default_selector :
            node = self.model.getBestboundNode()
        else:
            rank_node_ids = self.comb_model.get_select_node_decision(*self.bnbstates.get_dt_input())
            node = None
            for node_id in rank_node_ids:
                for _node in open_nodes:
                    if _node.getNumber() == node_id:
                        node = _node
                        break
                if node is not None:
                    break  # 找到就退出外层循环
        
        if node is None:
            print("dumb selection")
            return {"selnode": node}

        select_node_number = node.getNumber()
        open_nodes_number = []
        for open_node in open_nodes:
            open_nodes_number.append(open_node.getNumber())
        data = {
            "type" : "select",
            "data" : [select_node_number],
            "cand" : open_nodes_number
        }
        self.bnbstates.receive_states(data)
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
            pass
        if(event.getName() == 'NODEINFEASIBLE'):
            pass
        if(event.getName() == 'NODEFOCUSED'):
            pass
        if(event.getName() == 'BOUNDCHANGED'): # this event is no triggered, dont know why
            primalbound = self.model.getPrimalbound()
            dualbound = self.model.getDualbound()
            # print(f'BOUNDCHANGED and primalbound : {primalbound}, dualbound : {dualbound} ')
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

                        data = {
                            "type" : "branch",
                            "data" : [var_idx],
                            "cand" : cands_indexs,
                        }

                        self.bnbstates.receive_states(data)
                        #torch.save(info, file_path)
                        print(f'from states : branch on the node {node_number} and  var {bvar}')
                        #print(f'from states : branch on the node {node_number} and  var {bvar} and candidates {branch_cands}')
                        
                        save_branch_info = True

                    child_node = [lb, -1*ub,depth,node_number,child_number,var_idx,bbound,btype]
                    data = {
                        "type" : "node",
                        "data" : child_node,
                        "cand" : None,
                        "node_number" : child_number
                    }
                    self.bnbstates.receive_states(data)
        


class BNB_Brancher(sp.Branchrule):

    def __init__(self, comb_model, bnbstates, device):
        super().__init__()

        self.comb_model = comb_model
        self.device = device
        self.bnbstates  = bnbstates
        self.step = 0
        self.debug = False
        self.default_brancher = False

        self.branch_correct = 0

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
            var_logits = var_logits.squeeze(0)
            candidate_scores = var_logits[cands_indexs]
            select_branch_var_idx = candidate_scores.argmax()
            best_var = branch_cands[select_branch_var_idx]

        if self.debug:
            # strong branch logic
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

            print(f'branch on {self.model.getCurrentNode().getNumber()} and select {select_branch_var_idx} and strong branch select {best_cand_idx}')
            self.step += 1
            if select_branch_var_idx == best_cand_idx:
                self.branch_correct += 1
            #print(f'branch on {self.model.getCurrentNode().getNumber()} - {best_var} and candidates :{branch_cands}')

        self.model.branchVar(best_var)
        result = SCIP_RESULT.BRANCHED

        return {'result': result}