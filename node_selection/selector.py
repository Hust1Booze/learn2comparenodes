#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pyscipopt import SCIP_EVENTTYPE,Eventhdlr,SCIP_RESULT
from node_selectors import OracleNodeSelectorAbdel
import torch
import time


class OracleNodeSelRecorder(OracleNodeSelectorAbdel):
    
    def __init__(self, oracle_type, comp_behaviour_saver, comp_behaviour_saver_svm,sequence_saver,save_dir):
        super().__init__(oracle_type)
        self.counter = 0
        self.comp_behaviour_saver = comp_behaviour_saver
        self.comp_behaviour_saver_svm = comp_behaviour_saver_svm
        self.save_dir = save_dir
        self.saver = sequence_saver

    def set_LP_feature_recorder(self, LP_feature_recorder):
        self.comp_behaviour_saver.set_LP_feature_recorder(LP_feature_recorder)

    def nodeselect(self):
        leaves, children, siblings = self.model.getOpenNodes()
        open_nodes = set(leaves + children + siblings)
        if len(open_nodes)==0:
            #print("no open nodes", len(open_nodes))
            return {"selnode":self.model.getBestNode()}

        select_node = super().nodeselect()
        select_node_number = select_node['selnode'].getNumber()
        if select_node_number == 1:
            gpu_gpu, g = self.comp_behaviour_saver.get_graph_for_inf(self.model, select_node['selnode'])
            self.saver.milp_state = g

        leaves, children, siblings = self.model.getOpenNodes()
        open_nodes = leaves + children + siblings

        open_nodes_number = []
        for open_node in open_nodes:
            open_nodes_number.append(open_node.getNumber())

        gap = self.model.getGap()
        LPObjVal = self.model.getLPObjVal()
        local_estimate = self.model.getLocalEstimate()
        obj_val = self.model.getObjVal()
        primal_bound = self.model.getPrimalbound()
        dualbound = self.model.getDualbound()
        dualboundRoot = self.model.getDualboundRoot()

        obj_val_gap = obj_val - local_estimate
        obj_val_gap_ratio = obj_val_gap / obj_val
        obj_val_gap_ratio = obj_val_gap_ratio if obj_val_gap_ratio > 0 else 0
        obj_val_gap_ratio = obj_val_gap_ratio if obj_val_gap_ratio < 1 else 1
        obj_val_gap_ratio = obj_val_gap_ratio if obj_val_gap_ratio > 0 else 0
        
        data = {
            "type" : "select",
            "data" : [select_node_number],
            "cand" : open_nodes_number
        }

        self.saver.squence.append(data)
 
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
            
        
class ScipEvent(Eventhdlr):
    def __init__(self, model,sequence_saver,device):
        Eventhdlr.__init__(model)
        self.count = 0
        self.model = model
        self.device = device
        varrs = model.getVars() # equal to variables nums in bipartite graph representation
        original_conss = model.getConss()
        self.varrs = varrs
        self.original_conss = original_conss
        self.var2idx = dict([ (str_var, idx) for idx, var in enumerate(self.varrs) for str_var in [str(var)]  ])
        # print(f'the total vars {self.var2idx}')

        self.saver = sequence_saver

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
            pass
        if(event.getName() == 'NODEINFEASIBLE'):
            pass
        if(event.getName() == 'NODEFOCUSED'):
            print('debug')
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

                    #child_node = torch.tensor([[lb, -1*ub,depth,node_number,child_number,var_idx,bbound,btype]]).float()
                    child_node = [lb, -1*ub,depth,node_number,child_number,var_idx,bbound,btype]
                    lb = open_node.getLowerbound()
                    estimate = open_node.getEstimate()

                    gap = self.model.getGap()
                    LPObjVal = self.model.getLPObjVal()
                    local_estimate = self.model.getLocalEstimate()
                    obj_val = self.model.getObjVal()
                    primal_bound = self.model.getPrimalbound()
                    dualbound = self.model.getDualbound()
                    dualboundRoot = self.model.getDualboundRoot()
                    
                    data = {
                        "type" : "node",
                        "data" : child_node,
                        "cand" : None,
                        "node_number" : child_number
                    }
                    self.saver.squence.append(data)
        

