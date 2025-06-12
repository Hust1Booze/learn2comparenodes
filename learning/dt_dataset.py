import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
import glob
import re
import torch.nn.functional as F
import random

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
                    'type': 'gnn_state',
                    'data': gnn_input,
                    'reward': reward_accum
                })
                
                # 暂时用占位符长度，后续会被模型的实际输出替换
                # 注意：这里我们只添加一个token代表整个状态
                state_length = len(sequence_data)  # 记录状态token的位置
                type_ids.append(1)  # type 1 = state
                actions.append(-1)
                candidates.append([-1])
                state_embedded = True
                
                # 添加奖励token
                sequence_data.append({
                    'type': 'reward',
                    'data': reward_accum
                })
                type_ids.append(3)  # type 3 = reward      
                actions.append(-1)       
                candidates.append([-1])

            elif "selected" in name and state_embedded:
                node_idx = int(re.findall(r'node_(\d+)_selected', name)[0])
                
                # 存储节点索引数据
                sequence_data.append({
                    'type': 'node_idx',
                    'data': node_idx
                })
                type_ids.append(2)  # type 2 = select

                # 找到 node_idx 在 sequence 中的位置
                select_token_position = 1
                for i in range(len(actions)):
                    if type_ids[i] == 5 and actions[i] == node_idx:
                        select_token_position = i
                        break
                
                actions.append(select_token_position)

                cand_nodes = torch.load(pt)
                cand_node_idx_in_sequence = []
                for i in range(len(type_ids)):
                    if type_ids[i] == 5 and actions[i] in cand_nodes:
                        cand_node_idx_in_sequence.append(i)

                if len(cand_node_idx_in_sequence) != len(cand_nodes) and node_idx != 1:
                    print("error in select candidates")
                
                candidates.append(cand_node_idx_in_sequence)

                reward_accum += 1
                # 添加奖励token
                sequence_data.append({
                    'type': 'reward',
                    'data': reward_accum
                })
                type_ids.append(3)  # type 3 = reward
                actions.append(-1)  
                candidates.append([-1])

            elif "bestsolfound" in name and state_embedded:
                if type_ids[-1] != 3:
                    print("error in reward")
                reward_accum -= 100
                # 更新最后的奖励数据
                sequence_data[-1]['data'] = reward_accum
                
            elif "nodeinfeasible" in name and state_embedded:
                if type_ids[-1] != 3:
                    print("error in reward")
                reward_accum -= 10
                # 更新最后的奖励数据
                sequence_data[-1]['data'] = reward_accum

            elif "branchinfo" in name and state_embedded:
                info = torch.load(pt)
                branch_var_idx = info["selected_var_index"]
                cand_var_idx = info["candidate_indices"]
                scores = info["scores"]
                if len(cand_var_idx) != len(scores):
                    print("error in branch scores")
                branch_scores.append(scores)

                # 存储分支变量数据
                sequence_data.append({
                    'type': 'branch_var',
                    'data': branch_var_idx
                })
                type_ids.append(4)  # type 4 = branch
                actions.append(branch_var_idx) 
                candidates.append(cand_var_idx) 
                reward_accum += 1

            elif "branch_on" in name and state_embedded:
                match = re.search(r'branch_on_\d+_to_(\d+)\.pt', name)
                node_idx = int(match.group(1))
                  
                child_node = torch.load(pt)
                # 存储子节点数据
                sequence_data.append({
                    'type': 'child_node',
                    'data': child_node
                })
                type_ids.append(5)  # type 5 = node
                actions.append(node_idx) 
                candidates.append([-1])

        # 转换为tensor
        type_ids_tensor = torch.tensor(type_ids, dtype=torch.long)
        actions_tensor = torch.tensor(actions, dtype=torch.long)

        # 验证 type_ids_tensor 为 4 的位置对应的 candidates 和 branch_scores 长度是否一致
        branch_positions = (type_ids_tensor == 4).nonzero(as_tuple=True)[0]
        branch_step = 0
        for pos in branch_positions:
            pos_idx = pos.item()
            if len(candidates[pos_idx]) != len(branch_scores[branch_step]):
                print(f"Error: At position {pos_idx} (branch step {branch_step}), candidates length ({len(candidates[pos_idx])}) != branch_scores length ({len(branch_scores[branch_step])})")
                print(f"Candidates: {candidates[pos_idx]}")
                print(f"Branch scores: {branch_scores[branch_step]}")
            branch_step += 1

        return sequence_data, type_ids_tensor, actions_tensor, candidates, branch_scores
    
def bnb_collate(batch, pad_value=0.0):
    """
    batch: list of (sequence_data, type_ids_tensor, actions_tensor, candidates, branch_scores)
    现在sequence_data是原始数据列表，需要在训练时通过模型处理
    """
    sequence_data_list, type_ids, actions, candidates, branch_scores = zip(*batch)

    max_len = max(len(seq_data) for seq_data in sequence_data_list)

    padded_typeids = []
    padded_actions = []
    padded_candidates = []
    padded_branch_scores = []
    attention_masks = []
    padded_sequence_data = []

    for seq_data, tid, act, cand, scores in zip(sequence_data_list, type_ids, actions, candidates, branch_scores):
        pad_len = max_len - len(seq_data)

        # Pad type_ids and actions
        padded_tid = F.pad(tid, pad=(0, pad_len), value=pad_value)
        padded_act = F.pad(act, pad=(0, pad_len), value=pad_value)

        # Pad sequence_data with dummy entries
        padded_seq_data = seq_data + [{'type': 'padding', 'data': 0}] * pad_len

        # pad candidates: list[list[int]] → list of list
        padded_cand = cand + [[-1]] * pad_len  # keep consistent with structure
        
        # pad branch scores: list[list[float]] → list of list
        padded_scores = scores + [[0.0]] * pad_len # pad with zeros

        mask = torch.cat([torch.ones(len(seq_data)), torch.zeros(pad_len)])

        padded_sequence_data.append(padded_seq_data)
        padded_typeids.append(padded_tid)
        padded_actions.append(padded_act)
        padded_candidates.append(padded_cand)
        padded_branch_scores.append(padded_scores)
        attention_masks.append(mask)

    return (
        padded_sequence_data,             # [B, L] (list[list[dict]])
        torch.stack(padded_typeids),      # [B, L]
        torch.stack(padded_actions),      # [B, L]
        padded_candidates,                # [B, L] (list[list])
        padded_branch_scores,             # [B, L] (list[list])
        torch.stack(attention_masks),     # [B, L]
    )

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