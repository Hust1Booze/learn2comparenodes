import torch
from torch_geometric.data import Data
import numpy as np
from pyscipopt import Model
import os
import imp


def load_src(name, fpath):
     return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

load_src("data_type", "../learning/data_type.py" )
from data_type import TripartiteGraphData


def construct_tripartite_graph(model: Model):
    """
    在 SCIP 求解 MILP 过程中，动态构造 TripartiteGraphData
    
    参数：
    - model: SCIP 求解器模型 (pyscipopt.Model)
    
    返回：
    - TripartiteGraphData (用于 GNN)
    """
    
    # 1. 获取变量 & 约束特征
    variables = model.getVars()
    constraints = model.getConss()
    num_variables = len(variables)
    num_constraints = len(constraints)
    
    constraint_features = torch.tensor([model.getRhs(cons) for cons in constraints], dtype=torch.float32).view(-1, 1)
    variable_features = torch.tensor([[var.getLbOriginal(), var.getUbOriginal(),  model.getObjective()[var]] for var in variables], dtype=torch.float32)
    #variable_idx = torch.tensor([[var.getLbOriginal(), var.getUbOriginal(),  model.getObjective()[var]] for var in variables], dtype=torch.float32)

    # 2. 变量 ↔ 约束 连接 (参考 recorders.py)
    edge_index_cv = []
    edge_attr_cv = []
    constraints = model.getConss()
    variables = model.getVars()
    
    #var_to_index = {str(var): i for i, var in enumerate(variables)}  # 变量索引映射 (使用 str 作为 key)
    var_to_index = dict([ (str_var, idx) for idx, var in enumerate(variables) for str_var in [str(var)]  ])
    
    for j, cons in enumerate(constraints):
        vals_linear = model.getValsLinear(cons)  # 获取变量-约束系数关系
        if vals_linear:
            for var, coef in vals_linear.items():
                var_str = str(var)
                if var_str in var_to_index:
                    var_index = var_to_index[var_str]
                elif 't_' + var_str in var_to_index:
                    var_index = var_to_index['t_' + var_str]
                else:
                    var_index = var_to_index.get('_'.join(var_str.split('_')[1:]), -1)
                
                if coef != 0 and var_index >= 0:
                    edge_index_cv.append([var_index, j])  # 变量 -> 约束
                    edge_attr_cv.append(coef)  # 变量的系数
    
    edge_index_cv = torch.tensor(edge_index_cv, dtype=torch.long).T if edge_index_cv else None
    edge_attr_cv = torch.tensor(edge_attr_cv, dtype=torch.float32) if edge_attr_cv else None
    
    # 3. 获取 B&B 叶子节点
    leaves, children, siblings = model.getOpenNodes()
    open_nodes = leaves + children + siblings
    leaf_features = torch.tensor([[node.getLowerbound(),node.getEstimate(), node.getDepth()] for node in open_nodes ], dtype=torch.float32)
    leaf_idx = torch.tensor([node.getNumber() for node in open_nodes], dtype=torch.int)
    
    edge_index_vl = []
    edge_attr_vl = []
    for leaf_index, node in enumerate(open_nodes):

        if(node.getAncestorBranchings() == None):
            continue
        bvars, bbounds, btypes = node.getAncestorBranchings()
        
        for bvar, bound, btype in zip(bvars, bbounds, btypes): 
            
            if str(bvar) in var_to_index:
                var_index = var_to_index[str(bvar)]
            elif 't_'+str(bvar) in var_to_index:
                var_index = var_to_index['t_' + str(bvar)]
            else:
                var_index = var_to_index[ '_'.join(str(bvar).split('_')[1:]) ] 

            edge_index_vl.append([var_index, leaf_index])
            edge_attr_vl.append([bound,btype])
    
    edge_index_vl = torch.tensor(edge_index_vl, dtype=torch.long).T
    edge_attr_vl = torch.tensor(edge_attr_vl, dtype=torch.float32)
    
    
    # 4. 组装 TripartiteGraphData
    return TripartiteGraphData(
        constraint_features=constraint_features,
        variable_features=variable_features,
        leaf_features=leaf_features,
        leaf_idx = leaf_idx,
        edge_index_cv=edge_index_cv,
        edge_attr_cv=edge_attr_cv,
        edge_index_vl=edge_index_vl,
        edge_attr_vl=edge_attr_vl
    )

