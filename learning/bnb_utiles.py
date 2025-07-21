

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
    def __init__(self, comb_model, device): 

        self.data = []
        self.type = []
        self.cand = []
        self.node_id = []
        self.branch_label = []
        self.branch_action = []
        self.select_action = []
        self.select_label = []

        self.comb_model  = comb_model
        self.device = device
        self.state_embedded = False
        self.var2idx = None
        self.state_emb = None

        self.max_data_length = 13
        self.max_cand_length = 500
    def receive_var2idx(self,var2idx):
        self.var2idx = var2idx

    def receive_origin_milp(self, graph):
        gnn_input = graph
        gnn_input = [x.to(self.device) for x in gnn_input]
        self.state_emb = self.comb_model.gnn_encoder(*gnn_input)  # Tensor [D]
        self.state_embedded = True

    def get_dt_input(self):

        data_tensor = torch.tensor(self.data, dtype=torch.float32)
        type_tensor = torch.tensor(self.type, dtype=torch.int64)
        cand_tensor = torch.tensor(self.cand, dtype=torch.int64)
        node_id_tensor = torch.tensor(self.node_id, dtype=torch.int64)
        branch_label_tensor = torch.tensor(self.branch_label, dtype=torch.int64)
        branch_action_tensor = torch.tensor(self.branch_action, dtype=torch.int64)
        select_action_tensor = torch.tensor(self.select_action, dtype=torch.int64)
        select_label_tensor = torch.tensor(self.select_label, dtype=torch.int64)

        return self.state_emb, data_tensor, type_tensor, cand_tensor,node_id_tensor,branch_label_tensor,branch_action_tensor,select_action_tensor,select_label_tensor

    def receive_states(self, item):
        self.node_id.append(-1)
        self.branch_label.append(-1)
        self.branch_action.append(-1)
        self.select_action.append(-1)
        self.select_label.append(-1)
        if item['type'] == 'select':
            select_node_number = item["data"][0]
            if select_node_number != 1:
            # find node feature, use node feature when select node, except for node1
                index = self.node_id.index(select_node_number)
                self.data.append(self.data[index])
            else:
                self.data.append(item["data"] + [0] * (self.max_data_length - len(item["data"])))

            self.select_action[-1] = select_node_number
            self.select_label[-1] = select_node_number
            self.type.append(1)
            self.cand.append(item["cand"] + [-1] * (self.max_cand_length - len(item["cand"])))
        elif item['type'] == 'branch':
            if len(self.data)!=0 and self.type[-1] ==2 :
                continue_branch = True
                print(f'why continue branch')
            self.data.append(item["data"] + [0] * (self.max_data_length - len(item["data"])))
            self.type.append(2)
            self.cand.append(item["cand"] + [-1] * (self.max_cand_length - len(item["cand"])))
            self.branch_label[-1] = item["branch_label"]
            self.branch_action[-1] = item["data"][0]
        elif item['type'] == 'node':
            self.data.append(item["data"] + [0] * (self.max_data_length - len(item["data"])))
            self.type.append(3)
            self.cand.append([-1] * self.max_cand_length)
            self.node_id[-1] = item["node_number"]




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

    def nodeselect(self):
        self.step+=1
        #if self.step>=750:

        leaves, children, siblings = self.model.getOpenNodes()
        open_nodes = set(leaves + children + siblings)

        nodes = sorted(list(filter(lambda x: x.getNumber() not in self.added_ids, open_nodes)), key=lambda node: node.getNumber())

        if len(open_nodes)==0:
            node = self.model.getBestboundNode()
        if len(open_nodes)==1 and self.step<3:
            gpu_gpu, g = self.comp_behaviour_saver.get_graph_for_inf(self.model, nodes[0])
            self.bnbstates.receive_origin_milp(g)
            node = self.model.getBestboundNode()
        
        if self.default_selector or len(open_nodes)<=1:
            node = self.model.getBestNode()
        else:
            sorted_node_ids = self.comb_model.get_select_node_decision(*self.bnbstates.get_dt_input())
            node = None
            for node_id in sorted_node_ids:
                for _node in open_nodes:
                    if _node.getNumber() == node_id:
                        node = _node
                        break
                if node is not None:
                    break  # 找到就退出外层循环
        
        if node is None:
            #print("dumb selection")
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
            #print(f"Node: {node_number} , {event.getName()}")
            primalbound = self.model.getPrimalbound()
            dualbound = self.model.getDualbound()
            if primalbound != self.primalbound or dualbound != self.dualbound:
                self.primalbound = primalbound 
                self.dualbound = dualbound
                #print(f'primalbound : {primalbound}, dualbound : {dualbound} ')

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
            open_nodes_number = []
            open_nodes_depth = []
            open_nodes_lb = []

            branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = self.model.getLPBranchCands()
            save_branch_info = False

            for open_node in open_nodes:
                open_nodes_number.append(open_node.getNumber())
                open_nodes_depth.append(open_node.getDepth())
                open_nodes_lb.append(open_node.getLowerbound())

            for open_node in open_nodes:
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

                    if save_branch_info is False:
                        cands_indexs = []
                        for i in range(npriocands):
                            var = str(branch_cands[i])
                            if var in self.var2idx:
                                _var_idx = self.var2idx[var]
                            elif var.startswith("t_") and var[2:] in self.var2idx:
                                _var_idx = self.var2idx[var[2:]]
                            else:
                                print("error in save branch_cands info")
                            cands_indexs.append(_var_idx) 


                        data = {
                            "type" : "branch",
                            "data" : [var_idx], 
                            "branch_label" : var_idx,
                            "cand" : cands_indexs
                        }
                        self.bnbstates.receive_states(data)
                        save_branch_info = True

                    #child_node = torch.tensor([[lb, -1*ub,depth,node_number,child_number,var_idx,bbound,btype]]).float()
                    child_node = [lb, -1*ub,depth,node_number,child_number,var_idx,bbound,btype]

                    lb = open_node.getLowerbound()
                    estimate = open_node.getEstimate()
                    addedConss = open_node.getNAddedConss()
                    domchg = open_node.getNDomchg()
                    parentBranchings = open_node.getNParentBranchings()

                    # print(f"Node {child_number} - lb: {lb}, estimate: {estimate}")
                    # print(f"Node {child_number} - addedConss: {addedConss}, domchg: {domchg}, parentBranchings: {parentBranchings}")

                    gap = self.model.getGap()
                    LPObjVal = self.model.getLPObjVal()
                    local_estimate = self.model.getLocalEstimate()

                    # this all postive
                    primal_bound = self.model.getPrimalbound() *-1
                    dualbound = self.model.getDualbound()*-1
                    dualboundRoot = self.model.getDualboundRoot()*-1

                    # print(f"Model - gap: {gap}, LPObjVal: {LPObjVal}, local_estimate: {local_estimate}")
                    # print(f"Model - primal_bound: {primal_bound}, dualbound: {dualbound}, dualboundRoot: {dualboundRoot}")

                    x1 = relDistance(lb, LPObjVal)
                    x2 = relDistance(lb, local_estimate)

                    x3 = relDistance(estimate, LPObjVal)
                    x4 = relDistance(estimate, local_estimate)

                    x5 = relPosition(lb, primal_bound, dualbound)
                    x6 = relPosition(primal_bound, estimate, lb)

                    rel_depth = (np.max(open_nodes_depth) - depth) / np.max(open_nodes_depth)

                    child_node = [x1, x2, x3, x4, x5, x6, rel_depth, lb/np.min(open_nodes_lb), node_number,child_number,var_idx,bbound,btype]
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

        data = {
            "type" : "branch",
            "data" : [cands_indexs[select_branch_var_idx]], 
            "branch_label" : cands_indexs[select_branch_var_idx],
            "cand" : cands_indexs
        }
        self.bnbstates.receive_states(data)
        
        return {'result': result}
    
# static
def relDistance(x, y):
    """Relative distance between x and y."""
    if x*y<0:
        return 0.
    else:
        return np.abs(x-y) / np.max([np.abs(x), np.abs(y), 1e-10])
    
def relPosition(node_bound, ub, lb):
    """Relative position of node_bound with respect to global upper and lower bounds (or other commensurable quantities).
    
    :param node_bound: float, LP bound at node
    :param ub: float, global upper bound
    :param lb: float, global lower bound
    """
    if ub == lb:
        return 0 
    else:
        return np.abs(ub - node_bound) / np.abs(ub -lb)