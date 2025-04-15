from pyscipopt import Model, Branchrule, SCIP_RESULT

class StrongBranchingRule(Branchrule):

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
    


scip = Model()

most_inf_branch_rule = StrongBranchingRule(scip)


# x = scip.addVar(vtype='C', lb=0, ub=None, name='x')
# y = scip.addVar(vtype='C', lb=0, ub=None, name='y')
# z = scip.addVar(vtype='C', lb=0, ub=None, name='z')
# cons_1 = scip.addCons(x + y <= 5, name="cons_1")
# cons_1 = scip.addCons(y + z >= 3, name="cons_2")
# cons_1 = scip.addCons(x + y == 5, name="cons_3")
# scip.setObjective(2 * x + 3 * y - 5 * z, sense="minimize")

instance = '/data/ltf/foundation/learn2comparenodes/problem_generation/data/GISP/train/er_n=60_m=1040_p=0.60_SET2_setparam=100.00_alpha=0.50_31.lp'
scip.readProblem(instance)

most_inf_branch_rule = StrongBranchingRule(scip)
scip.includeBranchrule(
    branchrule=most_inf_branch_rule,
    name="Sampling branching rule", desc="",
    priority=666666, maxdepth=-1, maxbounddist=1)

scip.optimize()