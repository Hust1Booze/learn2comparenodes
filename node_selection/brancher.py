#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import pyscipopt.scip as sp
from pyscipopt import SCIP_EVENTTYPE,Eventhdlr,SCIP_RESULT
import torch
import time
import utiles
import numpy as np


class Brancher(sp.Branchrule):

    def __init__(self):
        super().__init__()


    def branchexeclp(self, allowaddcons):

        cands, scores, npriocands, bestcand = self.model.getVanillafullstrongData()
        self.model.branchVar(cands[bestcand])
        result = SCIP_RESULT.BRANCHED

        return {"result": result}

class StrongBranchingRule(sp.Branchrule):

    def __init__(self, scip, sequence_saver, save_dir, use_gasse_representation, random_branching_prob = 0):
        self.scip = scip
        self.save_dir = save_dir
        self.saver = sequence_saver
        self.use_gasse_representation = use_gasse_representation
        self.random_branching_prob = random_branching_prob

        varrs = self.scip.getVars() # equal to variables nums in bipartite graph representation
        original_conss = self.scip.getConss()
        self.varrs = varrs
        self.original_conss = original_conss
        self.var2idx = dict([ (str_var, idx) for idx, var in enumerate(self.varrs) for str_var in [str(var)]  ])

        self.khalil_root_buffer = {}

    def branchexeclp(self, allowaddcons):

        col_features, edge_features, row_features, map =  self.model.getBipartiteGraphRepresentation()
        node_number = self.model.getCurrentNode().getNumber()

        branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = self.scip.getLPBranchCands()

        action_set = [c.getCol().getLPPos() for c in branch_cands]

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

        action = None
        #10%的概率随机选择分支变量，90%的概率选择得分最高的变量
        if random.random() < self.random_branching_prob:
            # 随机选择
            random_cand_idx = random.randint(0, npriocands - 1)
            action = random_cand_idx
                    # Branch on the variable with the largest score
            down_child, eq_child, up_child = self.model.branchVarVal(
                branch_cands[random_cand_idx], branch_cands[random_cand_idx].getLPSol())
        else:
            action =  best_cand_idx
            # Branch on the variable with the largest score
            down_child, eq_child, up_child = self.model.branchVarVal(
                branch_cands[best_cand_idx], branch_cands[best_cand_idx].getLPSol())
        # action =  best_cand_idx
        # # Branch on the variable with the largest score
        # down_child, eq_child, up_child = self.model.branchVarVal(
        #     branch_cands[best_cand_idx], branch_cands[best_cand_idx].getLPSol())

        # Update the bounds of the down node and up node. Some cols might not exist due to pricing
        if self.scip.allColsInLP():
            if down_child is not None and down_bounds[best_cand_idx] is not None:
                self.scip.updateNodeLowerbound(down_child, down_bounds[best_cand_idx])
            if up_child is not None and up_bounds[best_cand_idx] is not None:
                self.scip.updateNodeLowerbound(up_child, up_bounds[best_cand_idx])

        
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


        if self.use_gasse_representation:
            cands_indexs = [c.getCol().getLPPos() for c in branch_cands]


        # branch_cands[best_cand_idx].getLbGlobal()
        # branch_cands[best_cand_idx].getLbLocal()
        # branch_cands[best_cand_idx].getLbOriginal()
        #print(f'branch node {node_number} on {cands_indexs[action]}')

        # 假设 col_features, row_features, edge_features 都是 numpy 数组或 list
        col_features = torch.tensor(col_features, dtype=torch.float)
        row_features = torch.tensor(row_features, dtype=torch.float)

        # 从 map 的定义可以看到：
        # edge_features[i] = [col_idx, row_idx, coef]
        edge_features = np.array(edge_features)

        # 边的特征就是 coef
        edge_attr = torch.tensor(edge_features[:, [2]], dtype=torch.float)

        # 图的连接结构，注意：torch_geometric 中要求 edge_index shape = [2, num_edges]
        edge_index = torch.tensor(
            np.stack([edge_features[:, 1], edge_features[:, 0]]),  # row_idx 在前，col_idx 在后
            dtype=torch.long
        )

        data = {
            "type" : "branch",
            "branch_label" : action,
            "branch_cand" : action_set,
            "col_features" : col_features,
            "row_features" : row_features,
            "edge_attr" :edge_attr,
            "edge_index": edge_index,
            'branch_node' : node_number,
        }
        self.saver.squence.append(data)

        return {"result": SCIP_RESULT.BRANCHED}
    
    def converse_to_gasse_representation(self):

        _col_features, _edge_features, _row_features, _map =  self.model.getBipartiteGraphRepresentation()

        g =  self.saver.milp_state
        constraint_features, edge_indices, edge_features, variable_features = g[0],g[1],g[2],g[3]
        for idx in range(len(edge_indices)):
            cur_edge = [edge_indices[0,idx].item(), edge_indices[1,idx].item(), edge_features[idx].item()]
            if cur_edge not in _edge_features:
                cur_edge = [edge_indices[1,idx].item(), edge_indices[0,idx].item(), edge_features[idx].item()]
                if cur_edge not in _edge_features:
                    print('why not in edge_features')
                    
        # 修改表征形式为 现在gnn输入格式
        _col_features = torch.tensor(_col_features, dtype=torch.float32)
        _row_features = torch.tensor(_row_features, dtype=torch.float32)
        _edge_indices = []
        _edge_features_ = []
        for i in range(len(_edge_features)):
            _edge_indices.append([_edge_features[i][0], _edge_features[i][1]])
            _edge_features_.append(_edge_features[i][2])
        _edge_indices = torch.tensor(_edge_indices, dtype=torch.int32).transpose(0,1)
        _edge_features_ = torch.tensor(_edge_features_, dtype=torch.float32)

        # 重新创建元组而不是直接修改
        self.saver.milp_state = (_row_features, _edge_indices, _edge_features_, _col_features, g[4], g[5])

        