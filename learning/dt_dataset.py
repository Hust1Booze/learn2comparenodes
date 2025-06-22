import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob
import re
import torch.nn.functional as F
import random

def simple_collate_fn(batch):
    """
    简单的collate函数，只pad sequence_data到相同长度
    Args:
        batch: list of sequence_data from dataset
    Returns:
        padded_sequence_data: list of padded sequences
    """
    # 找到batch中的最大序列长度
    max_seq_len = max(len(seq) for seq in batch)
    
    # 对每个序列进行padding
    padded_batch = []
    for seq in batch:
        seq_len = len(seq)
        # 用None填充到最大长度
        padded_seq = seq + [None] * (max_seq_len - seq_len)
        padded_batch.append(padded_seq)
    
    return padded_batch

class BnBSequentialDataset(Dataset):
    def __init__(self, data_dir, max_samples=None):
        self.data_dir = data_dir
        self.max_samples = max_samples
        self.trajectories = self.collect_trajectories()

    def collect_trajectories(self):
        # Each folder is a trajectory (e.g., a MILP instance run)
        all_dirs = [d for d in Path(self.data_dir).iterdir() if d.is_dir()]
        
        # 统计每个文件夹的文件数量
        dir_file_counts = []
        for d in all_dirs:
            file_count = len(list(d.glob("*.pt")))  # 只计算 .pt 文件
            dir_file_counts.append((d, file_count))
        
        # 按文件数量排序
        dir_file_counts.sort(key=lambda x: x[1])
        
        # 计算要保留的文件夹数量（去掉最大的10%）
        num_to_keep = int(len(dir_file_counts) * 0.8)
        filtered_dirs = [d for d, _ in dir_file_counts[:num_to_keep]]
        
        print(f"Total directories: {len(all_dirs)}")
        print(f"Directories after removing top 20%: {len(filtered_dirs)}")
        print(f"File count range in kept directories: {dir_file_counts[0][1]} - {dir_file_counts[num_to_keep-1][1]}")
        
        if self.max_samples is not None:
            # 随机选择指定数量的样本
            random.shuffle(filtered_dirs)
            filtered_dirs = filtered_dirs[:self.max_samples]
        
        return filtered_dirs

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        dir_path = self.trajectories[idx]
        pt_files = sorted(list(Path(dir_path).glob("*.pt")),
                        key=lambda p: float(p.name.split("_")[0]))

        # 存储原始数据而不是嵌入向量
        sequence_data = []  # 存储原始数据和类型信息
        type_ids = []
        actions = []  
        candidates = []
        branch_scores = []
        reward_accum = 0.0
        state_embedded = False

        # 第一遍遍历计算总奖励
        for pt in pt_files:
            name = pt.name
            if name.endswith("origin_milp.pt") and not state_embedded:
                state_embedded = True
            elif "selected" in name and state_embedded:
                reward_accum += -1
            elif "branch_on" in name and state_embedded:
                reward_accum += -1
            elif "bestsolfound" in name and state_embedded:
                reward_accum += 100
            elif "nodeinfeasible" in name and state_embedded:
                reward_accum += 10

        # 重置状态
        state_embedded = False
        state_length = 0
        
        for pt in pt_files:
            name = pt.name

            if name.endswith("origin_milp.pt") and not state_embedded:
                # 存储原始GNN输入数据
                gnn_input = torch.load(pt)
                sequence_data.append({
                    'type': 'state',
                    'data': gnn_input,
                    'reward': reward_accum
                })
                
                state_embedded = True

                #default select node 1
                # sequence_data.append({
                #     'type': 'select',
                #     'data': 1,
                #     'candidates' : []
                # })

                # 添加奖励token
                sequence_data.append({
                    'type': 'reward',
                    'data': reward_accum
                })



            elif "select" in name and state_embedded:
                node_idx = int(re.findall(r'select_(\d+)', name)[0])
                
                cand_nodes = torch.load(pt),

                # 存储节点索引数据
                sequence_data.append({
                    'type': 'select',
                    'data': node_idx,
                    'candidate' : cand_nodes
                })

                reward_accum += 1
                # 添加奖励token
                sequence_data.append({
                    'type': 'reward',
                    'data': reward_accum
                })

            elif "bestsolfound" in name and state_embedded:
                if sequence_data[-1]['type'] != 'reward':
                    print("error in reward")
                reward_accum -= 100
                # 更新最后的奖励数据
                sequence_data[-1]['data'] = reward_accum
                
            elif "nodeinfeasible" in name and state_embedded:
                if sequence_data[-1]['type'] != 'reward':
                    print("error in reward")
                reward_accum -= 10
                # 更新最后的奖励数据
                sequence_data[-1]['data'] = reward_accum

            elif "branch" in name and state_embedded:
                info = torch.load(pt)
                #node_number = info["node_number"]
                branch_var_idx = info["selected_var_index"]
                cand_var_idx = info["candidate_indices"]
                scores = info["scores"]
                if len(cand_var_idx) != len(scores):
                    print("error in branch scores")

                # 存储分支变量数据
                sequence_data.append({
                    'type': 'branch',
                    #'node_number':node_number,
                    'data': branch_var_idx,
                    'candidate': cand_var_idx,
                    'score': scores
                })
                reward_accum += 1

            elif "parent" in name and state_embedded:
                match = re.search(r'parent_\d+_to_(\d+)\.pt', name)
                node_idx = int(match.group(1))
                #parent_node_idx = int(match.group(0))  
                child_node = torch.load(pt)
                # 存储子节点数据
                sequence_data.append({
                    'type': 'node',
                    'data': child_node,
                    'node_id': node_idx,
                    #'parent_node_number': parent_node_idx
                })


        return sequence_data
    

def calculate_average_reward_static(dataset):
    """
    计算平均奖励的静态版本，不需要模型前向传播
    """
    total_reward = 0.0
    for i in range(len(dataset)):
        dir_path = dataset.trajectories[i]
        pt_files = sorted(list(Path(dir_path).glob("*.pt")),
                        key=lambda p: float(p.name.split("_")[0]))

        reward_accum = 0.0
        state_embedded = False

        for pt in pt_files:
            name = pt.name
            if name.endswith("origin_milp.pt") and not state_embedded:
                state_embedded = True
            elif "selected" in name and state_embedded:
                reward_accum += -1
            elif "branch_on" in name and state_embedded:
                reward_accum += -1
            elif "bestsolfound" in name and state_embedded:
                reward_accum += 100
            elif "nodeinfeasible" in name and state_embedded:
                reward_accum += 10

        total_reward += reward_accum

    avg_reward = total_reward / len(dataset)
    return avg_reward

def calculate_average_reward(dataset):
    total_reward = 0.0
    for i in range(len(dataset)):
        _ = dataset[i]  # This will compute reward_accum during __getitem__
        dir_path = dataset.trajectories[i]
        pt_files = sorted(list(Path(dir_path).glob("*.pt")),
                        key=lambda p: float(p.name.split("_")[0]))

        reward_accum = 0.0
        state_embedded = False

        for pt in pt_files:
            name = pt.name
            if name.endswith("origin_milp.pt") and not state_embedded:
                state_embedded = True
            elif "selected" in name and state_embedded:
                reward_accum += -1
            elif "branch_on" in name and state_embedded:
                reward_accum += -1
            elif "bestsolfound" in name and state_embedded:
                reward_accum += 100
            elif "nodeinfeasible" in name and state_embedded:
                reward_accum += 10

        total_reward += reward_accum

    avg_reward = total_reward / len(dataset)
    return avg_reward