import os
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob
import re
import torch.nn.functional as F
import random
import numpy as np

def simple_collate_fn(batch):
    """
    简单的collate函数，处理包含branch_sequence和select_sequence的batch
    Args:
        batch: list of (state, select_sequence, branch_sequence, select_cand, branch_cand, label) from dataset
    Returns:
        batched_data: dictionary containing batched tensors with masks
    """
    # 分离batch中的各个组件
    states = [item[0] for item in batch]
    select_sequences = [item[1] for item in batch]
    branch_sequences = [item[2] for item in batch]
    select_cands = [item[3] for item in batch]
    branch_cands = [item[4] for item in batch]
    node_ids = [item[5] for item in batch]
    types = [item[6] for item in batch]
    select_labels = [item[7] for item in batch]
    branch_actions = [item[8] for item in batch]
    branch_labels = [item[9] for item in batch]

    # 找到batch中的最大序列长度（只遍历一次）
    max_select_seq_len = max(len(seq) for seq in select_sequences)
    max_branch_seq_len = max(len(seq) for seq in branch_sequences)
    max_select_cands = max(len(seq) for seq in select_cands)
    max_branch_cands = max(len(seq) for seq in branch_cands)
    max_node_id = max(len(node_id) for node_id in node_ids)
    max_type = max(len(type) for type in types)
    max_branch_actions = max(len(action) for action in branch_actions)

    # 获取序列的维度信息
    if len(select_sequences) > 0 and len(select_sequences[0]) > 0:
        select_seq_dim = select_sequences[0].shape[1] if len(select_sequences[0].shape) > 1 else 1
        branch_seq_dim = branch_sequences[0].shape[1] if len(branch_sequences[0].shape) > 1 else 1
    else:
        select_seq_dim = 1
        branch_seq_dim = 1
    
    # 初始化结果列表
    padded_select_sequences = []
    padded_branch_sequences = []
    padded_select_cands = []
    padded_branch_cands = []
    padded_node_ids = []
    padded_types = []
    select_sequence_masks = []
    branch_sequence_masks = []
    select_cand_masks = []
    branch_cand_masks = []
    node_id_masks = []
    padded_branch_actions = []
    # 只遍历一次batch，完成所有padding和mask生成
    for seq, branch_seq, select_cand, branch_cand, node_id, type, branch_action in zip(
        select_sequences, branch_sequences, select_cands, branch_cands, node_ids, types, branch_actions
    ):
        # 处理select_sequences
        seq_len = len(seq)
        if len(seq.shape) > 1:
            padded_seq = torch.zeros(max_select_seq_len, select_seq_dim, dtype=seq.dtype)
        else:
            padded_seq = torch.zeros(max_select_seq_len, dtype=seq.dtype)
        padded_seq[:seq_len] = seq
        padded_select_sequences.append(padded_seq)
        
        # 生成select_sequence mask
        mask = torch.ones(max_select_seq_len, dtype=torch.bool)
        mask[:seq_len] = False
        select_sequence_masks.append(mask)
        
        # 处理branch_sequences
        branch_seq_len = len(branch_seq)
        if len(branch_seq.shape) > 1:
            padded_branch_seq = torch.zeros(max_branch_seq_len, branch_seq_dim, dtype=branch_seq.dtype)
        else:
            padded_branch_seq = torch.zeros(max_branch_seq_len, dtype=branch_seq.dtype)
        padded_branch_seq[:branch_seq_len] = branch_seq
        padded_branch_sequences.append(padded_branch_seq)
        
        # 生成branch_sequence mask
        branch_mask = torch.ones(max_branch_seq_len, dtype=torch.bool)
        branch_mask[:branch_seq_len] = False
        branch_sequence_masks.append(branch_mask)
        
        # 处理select_cands
        select_cand_len = len(select_cand)
        if len(select_cand.shape) > 1:
            padded_select_cand = torch.full((max_select_cands, select_cand.shape[1]), -1, dtype=select_cand.dtype)
        else:
            padded_select_cand = torch.full((max_select_cands,), -1, dtype=select_cand.dtype)
        padded_select_cand[:select_cand_len] = select_cand
        padded_select_cands.append(padded_select_cand)
        
        # 生成select_cand mask
        select_cand_mask = torch.zeros(max_select_cands, dtype=torch.bool)
        select_cand_mask[:select_cand_len] = True
        select_cand_masks.append(select_cand_mask)
        
        # 处理branch_cands
        branch_cand_len = len(branch_cand)
        if len(branch_cand.shape) > 1:
            padded_branch_cand = torch.full((max_branch_cands, branch_cand.shape[1]), -1, dtype=branch_cand.dtype)
        else:
            padded_branch_cand = torch.full((max_branch_cands,), -1, dtype=branch_cand.dtype)
        padded_branch_cand[:branch_cand_len] = branch_cand
        padded_branch_cands.append(padded_branch_cand)
        
        # 生成branch_cand mask
        branch_cand_mask = torch.zeros(max_branch_cands, dtype=torch.bool)
        branch_cand_mask[:branch_cand_len] = True
        branch_cand_masks.append(branch_cand_mask)
        
        # 处理node_ids
        node_id_len = len(node_id)
        if len(node_id.shape) > 1:
            padded_node_id = torch.full((max_node_id, node_id.shape[1]), -1, dtype=node_id.dtype)
        else:
            padded_node_id = torch.full((max_node_id,), -1, dtype=node_id.dtype)
        padded_node_id[:node_id_len] = node_id
        padded_node_ids.append(padded_node_id)
        
        # 生成node_id mask
        node_id_mask = torch.zeros(max_node_id, dtype=torch.bool)
        node_id_mask[:node_id_len] = True
        node_id_masks.append(node_id_mask)

        # 处理type
        type_len = len(type)
        if len(type.shape) > 1:
            padded_type = torch.full((max_type, type.shape[1]), 0, dtype=type.dtype)
        else:
            padded_type = torch.full((max_type,), 0, dtype=type.dtype)
        padded_type[:type_len] = type
        padded_types.append(padded_type)

        # 处理branch_actions
        branch_action_len = len(branch_action)
        if len(branch_action.shape) > 1:
            padded_branch_action = torch.full((max_branch_actions, branch_action.shape[1]), -1, dtype=branch_action.dtype)
        else:
            padded_branch_action = torch.full((max_branch_actions,), -1, dtype=branch_action.dtype)
        padded_branch_action[:branch_action_len] = branch_action
        padded_branch_actions.append(padded_branch_action)
    # 堆叠所有tensor
    batched_data = {
        'select_sequences': torch.stack(padded_select_sequences),
        'branch_sequences': torch.stack(padded_branch_sequences),
        'select_cands': torch.stack(padded_select_cands),
        'branch_cands': torch.stack(padded_branch_cands),
        'node_ids': torch.stack(padded_node_ids),
        'types' : torch.stack(padded_types),
        'select_labels': torch.stack(select_labels).to(torch.int64),
        'branch_labels': torch.stack(branch_labels).to(torch.int64),
        'branch_actions': torch.stack(padded_branch_actions).to(torch.int64),
        'select_sequence_masks': torch.stack(select_sequence_masks),
        'branch_sequence_masks': torch.stack(branch_sequence_masks),
        'select_cand_masks': torch.stack(select_cand_masks),
        'branch_cand_masks': torch.stack(branch_cand_masks),
        'node_id_masks': torch.stack(node_id_masks),
    
    }
    
    return states, batched_data

class BnBSequentialDataset(Dataset):
    def __init__(self, data_dir, max_samples=None):
        self.data_dir = data_dir
        self.max_samples = max_samples
        self.trajectories = self.collect_trajectories()

    def collect_trajectories(self):
        # Each folder is a trajectory (e.g., a MILP instance run)
        all_dirs = [d for d in Path(self.data_dir).iterdir() if d.is_dir()]
        
        print(f"Total directories: {len(all_dirs)}")

        sequence_lengths = []
        valid_dirs = []
        
        for i, dir_path in enumerate(all_dirs):
            try:
                # 获取序列数据
                sequence_data = torch.load(dir_path / 'data.pt')
                type_data = torch.load(dir_path / 'type.pt')
                
                # 应用与 __getitem__ 相同的处理逻辑
                sequence_data = sequence_data[2:]  # 跳过前两个元素
                type_data = type_data[2:]
                
                # 记录序列长度和对应的目录
                sequence_lengths.append(sequence_data.shape[0])
                valid_dirs.append(dir_path)
                
            except Exception as e:
                print(f"处理样本 {dir_path} 时出错: {e}")
                continue
        
        if not sequence_lengths:
            print("没有找到有效的序列数据")
            return []
        
        # 计算统计信息
        sequence_lengths = np.array(sequence_lengths)
        
        # 计算95%分位数，剔除5%最大长度的数据
        length_threshold = np.percentile(sequence_lengths, 95)
        print(f"序列长度95%分位数: {length_threshold:.2f}")
        
        # 筛选出长度小于等于95%分位数的数据
        filtered_indices = sequence_lengths <= length_threshold
        filtered_dirs = [valid_dirs[i] for i in range(len(valid_dirs)) if filtered_indices[i]]
        filtered_lengths = sequence_lengths[filtered_indices]
        
        print(f"原始样本数: {len(all_dirs)}")
        print(f"有效样本数: {len(valid_dirs)}")
        print(f"剔除异常后样本数: {len(filtered_dirs)}")
        print(f"剔除的样本数: {len(valid_dirs) - len(filtered_dirs)}")
        
        # 输出统计信息
        stats = {
            'mean_length': np.mean(filtered_lengths),
            'max_length': np.max(filtered_lengths),
            'median_length': np.median(filtered_lengths),
            'min_length': np.min(filtered_lengths),
            'std_length': np.std(filtered_lengths),
            'total_samples': len(filtered_lengths),
            'length_distribution': {
                '0-10': np.sum(filtered_lengths <= 10),
                '11-50': np.sum((filtered_lengths > 10) & (filtered_lengths <= 50)),
                '51-100': np.sum((filtered_lengths > 50) & (filtered_lengths <= 100)),
                '101-200': np.sum((filtered_lengths > 100) & (filtered_lengths <= 200)),
                '201-500': np.sum((filtered_lengths > 200) & (filtered_lengths <= 500)),
                '500+': np.sum(filtered_lengths > 500),
                '1000+': np.sum(filtered_lengths > 1000),
                '2000+': np.sum(filtered_lengths > 2000)
            }
        }
        
        print(f"\n序列长度统计结果 (剔除异常后):")
        print(f"总样本数: {stats['total_samples']}")
        print(f"平均长度: {stats['mean_length']:.2f}")
        print(f"最大长度: {stats['max_length']}")
        print(f"最小长度: {stats['min_length']}")
        print(f"中位数长度: {stats['median_length']:.2f}")
        print(f"标准差: {stats['std_length']:.2f}")
        print(f"\n长度分布:")
        for range_name, count in stats['length_distribution'].items():
            percentage = (count / stats['total_samples']) * 100
            print(f"  {range_name}: {count} 个样本 ({percentage:.1f}%)")

        if self.max_samples is not None:
            # 随机选择指定数量的样本
            random.shuffle(filtered_dirs)
            filtered_dirs = filtered_dirs[:self.max_samples]
            print(f"随机选择后样本数: {len(filtered_dirs)}")
        
        return filtered_dirs
    

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        dir_path = self.trajectories[idx]

        state = torch.load(dir_path / 'state.pt')
        sequence_data = torch.load(dir_path / 'data.pt')
        type = torch.load(dir_path / 'type.pt')
        cand = torch.load(dir_path / 'cand.pt')
        node_id = torch.load(dir_path / 'node_id.pt')
        branch_label = torch.load(dir_path / 'branch_label.pt')
        branch_action = torch.load(dir_path / 'branch_action.pt')
        select_action = torch.load(dir_path / 'select_action.pt')
        select_label = torch.load(dir_path / 'select_label.pt')

        if sequence_data.shape[0]> 2000 :
            print(f"sequence_data.shape[0]> 2000: {sequence_data.shape[0]}, path: {dir_path}")
        
        # 暂时这样做，不知道为什么 SETCOVER收集的数据会选择节点1三次,甚至多次
        sequence_data = sequence_data[2:]
        type = type[2:]
        cand = cand[2:]
        node_id = node_id[2:]
        branch_label = branch_label[2:]
        branch_action = branch_action[2:]
        select_action = select_action[2:]
        select_label = select_label[2:]

        # 找到type=1和type=0的位置
        type_1_indices = torch.where(type == 1)[0][1:]  # select positions, not choose first selct

        type_2_indices = torch.where(type == 2)[0] # branch positions

        select_idx = random.choice(type_1_indices.tolist())
        select_sequence = sequence_data[:select_idx]
        select_action = select_action[select_idx]
        select_label = select_label[select_idx]
        
        branch_idx = random.choice(type_2_indices.tolist())
        branch_sequence = sequence_data[:branch_idx]
        branch_action = branch_action[:branch_idx]
        branch_label = branch_label[branch_idx]

        if select_action not in node_id:
            print(f"select_idx not in node_id: {select_idx}")

        return state, select_sequence, branch_sequence, cand[select_idx], cand[branch_idx] ,node_id[:select_idx], type, select_action, branch_action, branch_label
    

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