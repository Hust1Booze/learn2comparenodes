from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import os
import re
import random
import torch
from data_type import TripartiteGraphData  # 确保你已经正确导入了你的类定义
import numpy as np
from pathlib import Path
import torch_geometric

def create_dataset(data_dir):

    all_lp_files = []

    for dirpath, dirnames, filenames in os.walk(data_dir):
        if dirpath.endswith(".lp"):
            full_path = os.path.join(data_dir, dirpath)
            all_lp_files.append(full_path)

    print(f"total {len(all_lp_files)}  trajectors in  {data_dir}")

    random.shuffle(all_lp_files)

    states = []
    selnode_actions = []
    branch_actions = []
    returns = []
    done_idxs = []
    stepwise_returns = []
    candidate_nodes = []
    candidate_variables = []

    
    selnode_entries = []
    branch_dict = {}
    for trajector in all_lp_files:
        returns.append(0)
        stepwise_returns.append(0)
        # 获取所有 .pt 文件（带路径）
        pt_files = [os.path.join(trajector, f) for f in os.listdir(trajector) if f.endswith(".pt")]

        # 按文件名中的前缀时间戳排序
        pt_files.sort(key=lambda x: float(os.path.basename(x).split('_')[0]))

        for pt_path in pt_files:
            if "selected" in pt_path:
                selnode_state = torch.load(pt_path)
                states.append([selnode_state._features()])
                selnode_actions.append(selnode_state.selnode)
                branch_actions.append(-1)
                returns.append(returns[-1] -0.01)
                stepwise_returns.append(-0.01)
                candidate_nodes.append(selnode_state.candidate_nodes)
                candidate_variables.append(-1)
                
            elif "branched" in pt_path:
                branch_state = torch.load(pt_path)
                states.append([branch_state._features()])
                selnode_actions.append(-1)
                branch_actions.append(branch_state.branch_index)
                returns.append(returns[-1] -0.01)
                stepwise_returns.append(-0.01)
                candidate_nodes.append(-1)
                candidate_variables.append(branch_state.cand_vars)
            elif "bestsolfound" in pt_path:
                returns[-1] += 1
                stepwise_returns[-1] +=1
            elif "nodeinfeasible" in pt_path:
                returns[-1] += 0.1
                stepwise_returns[-1] +=0.1
        #remove last returns to keep size
        returns.pop()
        stepwise_returns.pop()
        done_idxs.append(len(states)-1)

    return states,selnode_actions,branch_actions, returns,stepwise_returns ,candidate_nodes,candidate_variables,done_idxs


class BnBDataset(torch_geometric.data.Dataset):
    def __init__(self, data_dir, block_size=4):
        self.data_dir = data_dir
        self.block_size = block_size

        self.states,self.selnode_actions,self.branch_actions,self.returns,self.stepwise_returns,self.candidate_nodes,self.candidate_variables,self.done_idxs = create_dataset(data_dir)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):

        _done_idx = idx + self.block_size

        for done_idx in self.done_idxs:
            if done_idx>idx:
                _done_idx = min(int(done_idx), _done_idx)
                break
        
        states = self.states[idx:_done_idx]
        selnode_actions = self.selnode_actions[idx:_done_idx]
        branch_actions = self.branch_actions[idx:_done_idx]
        returns = self.returns[idx:_done_idx]
        stepwise_returns = self.stepwise_returns[idx:_done_idx]
        candidate_nodes = self.candidate_nodes[idx:_done_idx]
        candidate_variables = self.candidate_variables[idx:_done_idx]

        return states,selnode_actions,branch_actions,returns, stepwise_returns,candidate_nodes,candidate_variables
