#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import pyscipopt.scip as sp
from pyscipopt import SCIP_EVENTTYPE,Eventhdlr,SCIP_RESULT
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

    def __init__(self, scip, sequence_saver, save_dir):
        self.scip = scip
        self.save_dir = save_dir

        self.saver = sequence_saver

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

        # action = None
        # #10%的概率随机选择分支变量，90%的概率选择得分最高的变量
        # if random.random() < 0.1:
        #     # 随机选择
        #     random_cand_idx = random.randint(0, npriocands - 1)
        #     action = random_cand_idx
        #             # Branch on the variable with the largest score
        #     down_child, eq_child, up_child = self.model.branchVarVal(
        #         branch_cands[random_cand_idx], branch_cands[random_cand_idx].getLPSol())
        # else:
        #     action =  best_cand_idx
        #     # Branch on the variable with the largest score
        #     down_child, eq_child, up_child = self.model.branchVarVal(
        #         branch_cands[best_cand_idx], branch_cands[best_cand_idx].getLPSol())
        action =  best_cand_idx
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
        file_path = self.save_dir + f"/{current_time:.4f}_branch_{node_number}.pt"
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
            "type":'branch',
            "node_number" : node_number,
            "candidate_indices": cands_indexs,
            "scores": scores,
            "selected_var_index": best_cand_idx
        }

        data = {
            "type" : "branch",
            "data" : [action],
            "branch_label" : best_cand_idx,
            "cand" : cands_indexs
        }

        self.saver.squence.append(data)
        # print(f'branch on the node {node_number} and  var {branch_cands[best_cand_idx]}')
        # torch.save(info, file_path)
        return {"result": SCIP_RESULT.BRANCHED}