import os
from pyscipopt import SCIP_EVENTTYPE,Eventhdlr,SCIP_RESULT
from node_selectors import OracleNodeSelectorAbdel
import torch
import time
import numpy as np
        
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
            pass
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

                        save_branch_info = True


                    lb = open_node.getLowerbound()
                    estimate = open_node.getEstimate()
                    addedConss = open_node.getNAddedConss()
                    domchg = open_node.getNDomchg()
                    parentBranchings = open_node.getNParentBranchings()

                    gap = self.model.getGap()
                    LPObjVal = self.model.getLPObjVal()
                    local_estimate = self.model.getLocalEstimate()

                    # this all postive
                    primal_bound = self.model.getPrimalbound() *-1
                    dualbound = self.model.getDualbound()*-1
                    dualboundRoot = self.model.getDualboundRoot()*-1

                    x1 = relDistance(lb, LPObjVal)
                    x2 = relDistance(lb, local_estimate)

                    x3 = relDistance(estimate, LPObjVal)
                    x4 = relDistance(estimate, local_estimate)

                    x5 = relPosition(lb, primal_bound, dualbound)
                    x6 = relPosition(primal_bound, estimate, lb)

                    rel_depth = (np.max(open_nodes_depth) - depth) / np.max(open_nodes_depth)

                    child_node = torch.tensor([x1, x2, x3, x4, x5, x6, rel_depth, lb/np.min(open_nodes_lb)], dtype=torch.float)
                    data = {
                        "type" : "node",
                        "node_data" : child_node,
                        "node_number" : child_number,
                        "parent_number" : node_number,  
                        "branch_var" : var
                    }
                    self.saver.squence.append(data)
        

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


def extract_khalil_variable_features(model, candidates, root_buffer):
    """
    Extract features following Khalil et al. (2016) Learning to Branch in Mixed Integer Programming.

    Parameters
    ----------
    model : pyscipopt.scip.Model
        The current model.
    candidates : list of pyscipopt.scip.Variable's
        A list of variables for which to compute the variable features.
    root_buffer : dict
        A buffer to avoid re-extracting redundant root node information (None to deactivate buffering).

    Returns
    -------
    variable_features : 2D np.ndarray
        The features associated with the candidate variables.
    """
    # update state from state_buffer if any
    scip_state = model.getKhalilState(root_buffer, candidates)

    variable_feature_names = sorted(scip_state)
    variable_features = np.stack([scip_state[feature_name] for feature_name in variable_feature_names], axis=1)

    return variable_features