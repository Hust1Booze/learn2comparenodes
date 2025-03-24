import os
import argparse
import multiprocessing as mp
import pickle
import glob
import numpy as np
import shutil
import gzip
import utilities

import pyscipopt as scip
from pyscipopt import Model, Branchrule, SCIP_RESULT
import numpy as np


class SamplingAgent(scip.Branchrule):

    def __init__(self, episode, instance, seed, exploration_policy, query_expert_prob, follow_expert=True):
        self.episode = episode
        self.instance = instance
        self.seed = seed
        self.exploration_policy = exploration_policy
        self.query_expert_prob = query_expert_prob
        self.follow_expert = follow_expert

        self.rng = np.random.RandomState(seed)
        self.new_node = True
        self.sample_counter = 0

    def branchinit(self):
        self.khalil_root_buffer = {}

    def branchexeclp(self, allowaddcons):

        # once in a while, also run the expert policy and record the (state, action) pair
        query_expert = self.rng.rand() < self.query_expert_prob
        if query_expert:
            state = None #utilities.extract_state(self.model)
            cands, *_ = self.model.getPseudoBranchCands()
            state_khalil = None #utilities.extract_khalil_variable_features(self.model, cands, self.khalil_root_buffer)

            result = self.model.executeBranchRule('vanillafullstrong', allowaddcons)
            cands_, scores, npriocands, bestcand = self.model.getVanillafullstrongData()

            assert result == scip.SCIP_RESULT.DIDNOTRUN
            assert all([c1.getCol().getLPPos() == c2.getCol().getLPPos() for c1, c2 in zip(cands, cands_)])

            action_set = [c.getCol().getLPPos() for c in cands]
            expert_action = action_set[bestcand]

            data = [state, state_khalil, expert_action, action_set, scores]

            # Do not record inconsistent scores. May happen if SCIP was early stopped (time limit).
            # if not any([s < 0 for s in scores]):

            #     filename = f'{self.out_dir}/sample_{self.episode}_{self.sample_counter}.pkl'
            #     with gzip.open(filename, 'wb') as f:
            #         pickle.dump({
            #             'episode': self.episode,
            #             'instance': self.instance,
            #             'seed': self.seed,
            #             'node_number': self.model.getCurrentNode().getNumber(),
            #             'node_depth': self.model.getCurrentNode().getDepth(),
            #             'data': data,
            #             }, f)

            #     self.sample_counter += 1

        # if exploration and expert policies are the same, prevent running it twice
        if not query_expert or (not self.follow_expert and self.exploration_policy != 'vanillafullstrong'):
            result = self.model.executeBranchRule(self.exploration_policy, allowaddcons)

        # apply 'vanillafullstrong' branching decision if needed
        if query_expert and self.follow_expert or self.exploration_policy == 'vanillafullstrong':
            assert result == scip.SCIP_RESULT.DIDNOTRUN
            cands, scores, npriocands, bestcand = self.model.getVanillafullstrongData()
            self.model.branchVar(cands[bestcand])
            result = scip.SCIP_RESULT.BRANCHED

        return {"result": result}
    

class StrongBranchingRule(scip.Branchrule):

    def __init__(self, scip):
        self.scip = scip

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

        # Branch on the variable with the largest score
        down_child, eq_child, up_child = self.model.branchVarVal(
            branch_cands[best_cand_idx], branch_cands[best_cand_idx].getLPSol())

        # Update the bounds of the down node and up node. Some cols might not exist due to pricing
        if self.scip.allColsInLP():
            if down_child is not None and down_bounds[best_cand_idx] is not None:
                self.scip.updateNodeLowerbound(down_child, down_bounds[best_cand_idx])
            if up_child is not None and up_bounds[best_cand_idx] is not None:
                self.scip.updateNodeLowerbound(up_child, up_bounds[best_cand_idx])

        return {"result": SCIP_RESULT.BRANCHED}
    



class MostInfBranchRule(Branchrule):

    def __init__(self, scip):
        self.scip = scip

    def branchexeclp(self, allowaddcons):

        # Get the branching candidates. Only consider the number of priority candidates (they are sorted to be first)
        # The implicit integer candidates in general shouldn't be branched on. Unless specified by the user
        # npriocands and ncands are the same (npriocands are variables that have been designated as priorities)
        branch_cands, branch_cand_sols, branch_cand_fracs, ncands, npriocands, nimplcands = self.scip.getLPBranchCands()

        # Find the variable that is most fractional
        best_cand_idx = 0
        best_dist = np.inf
        for i in range(npriocands):
            if abs(branch_cand_fracs[i] - 0.5) <= best_dist:
                best_dist = abs(branch_cand_fracs[i] - 0.5)
                best_cand_idx = i

        # Branch on the variable with the largest score
        down_child, eq_child, up_child = self.model.branchVarVal(
            branch_cands[best_cand_idx], branch_cand_sols[best_cand_idx])

        return {"result": SCIP_RESULT.BRANCHED}