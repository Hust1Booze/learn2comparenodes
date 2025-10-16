#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pyscipopt import SCIP_EVENTTYPE,Eventhdlr,SCIP_RESULT
from node_selectors import OracleNodeSelectorAbdel
import torch
import time
import numpy as np


class OracleNodeSelRecorder(OracleNodeSelectorAbdel):
    
    def __init__(self, oracle_type, comp_behaviour_saver, comp_behaviour_saver_svm,sequence_saver,save_dir):
        super().__init__(oracle_type, sel_policy = '')
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
            #_col_features, _edge_features, _row_features, _map =  self.model.getBipartiteGraphRepresentation()
            self.saver.milp_state = g

        leaves, children, siblings = self.model.getOpenNodes()
        open_nodes = leaves + children + siblings

        open_nodes_number = []
        for open_node in open_nodes:
            open_nodes_number.append(open_node.getNumber())
        
        # random select
        # if np.random.rand() < 0.1 and len(open_nodes) > 1:
        #     select_node = {"selnode":open_nodes[np.random.randint(0, len(open_nodes))]}

        data = {
            "type" : "select",
            "select_label" : [select_node_number],
            "select_cand" : open_nodes_number
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
            