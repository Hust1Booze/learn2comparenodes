#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:08:53 2022

@author: aglabassi
"""

import torch
import torch_geometric

class TripartiteGraphData(torch_geometric.data.Data):
    """
    这个类用于存储 Branch and Bound (B&B) 过程中生成的三分图数据，包括：
    - 约束节点 (constraint nodes)
    - 变量节点 (variable nodes)
    - 叶子节点 (leaf nodes)
    """
    def __init__(self, 
                 constraint_features=None, 
                 variable_features=None, 
                 leaf_features=None,  # 叶子节点特征
                 leaf_idx = None,
                 
                 edge_index_cv=None, edge_attr_cv=None,  # 约束 <-> 变量
                 edge_index_vl=None, edge_attr_vl=None,  # 变量 <-> 叶子
                 
                 bounds=None, depth=None, y=None): 
        super().__init__()
        
        self.constraint_features = constraint_features  # h^c
        self.variable_features = variable_features      # h^v
        self.leaf_features = leaf_features              # h^l (新增叶子节点特征)
        self.leaf_idx = leaf_idx

        self.edge_index_cv = edge_index_cv  # 约束 <-> 变量
        self.edge_attr_cv = edge_attr_cv
        
        self.edge_index_vl = edge_index_vl  # 变量 <-> 叶子 (新增边)
        self.edge_attr_vl = edge_attr_vl
        
        self.bounds = bounds
        self.depth = depth
        self.y = y  # 标签 (如果需要)
        
    def __inc__(self, key, value, *args, **kwargs):
        """
        处理 PyG 索引递增规则，以确保正确拼接多个图。
        """
        if key == 'edge_index_cv':
            return torch.tensor([[self.variable_features.size(0)], [self.constraint_features.size(0)]])
        elif key == 'edge_index_vl':
            return torch.tensor([[self.leaf_features.size(0)], [self.variable_features.size(0)]])  # 变量 <-> 叶子
        else:
            return super().__inc__(key, value, *args, **kwargs)
        

class BipartiteGraphPairData(torch_geometric.data.Data):
    """
    This class encode a pair of node bipartite graphs observation, s is graph0, t is graph1 
    """
    def __init__(self, constraint_features_s=None, edge_indices_s=None, edge_features_s=None, variable_features_s=None, bounds_s=None, depth_s=None, 
                 constraint_features_t=None, edge_indices_t=None, edge_features_t=None, variable_features_t=None,  bounds_t=None, depth_t=None,
                 y=None): 
        
        super().__init__()
        
        self.variable_features_s, self.constraint_features_s, self.edge_index_s, self.edge_attr_s, self.bounds_s, self.depth_s =  (
            variable_features_s, constraint_features_s, edge_indices_s, edge_features_s, bounds_s, depth_s)
        
        self.variable_features_t, self.constraint_features_t, self.edge_index_t, self.edge_attr_t, self.bounds_t, self.depth_t  = (
            variable_features_t, constraint_features_t, edge_indices_t, edge_features_t, bounds_t, depth_t)
        
        self.y = y
        

   
    def __inc__(self, key, value, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs 
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == 'edge_index_s':
            return torch.tensor([[self.variable_features_s.size(0)], [self.constraint_features_s.size(0)]])
        elif key == 'edge_index_t':
            return torch.tensor([[self.variable_features_t.size(0)], [self.constraint_features_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, idx):
        data = torch.load(self.sample_files[idx])
        return data
