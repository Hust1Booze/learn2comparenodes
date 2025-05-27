import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
import glob
import re
import torch.nn.functional as F

class BnBSequentialDataset(Dataset):
    def init(self, data_dir, model, device):
        self.data_dir = data_dir
        self.device = device
        self.model = model
        self.trajectories = self.collect_trajectories()

    def __init__(self, data_dir, model, device):
        self.data_dir = data_dir
        self.device = device
        self.model = model
        self.trajectories = self.collect_trajectories()

    def collect_trajectories(self):
        # Each folder is a trajectory (e.g., a MILP instance run)
        all_dirs = [d for d in Path(self.data_dir).iterdir() if d.is_dir()]
        return all_dirs

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        dir_path = self.trajectories[idx]
        pt_files = sorted(list(Path(dir_path).glob("*.pt")),
                        key=lambda p: float(p.name.split("_")[0]))

        sequence = [] # [states, selected node, reward, branch, node, node, selected node ...]
        type_ids = []
        actions = []  
        candidates = []
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

        state_embedded = False
        state_length = 0
        for pt in pt_files:
            name = pt.name

            if name.endswith("origin_milp.pt") and not state_embedded:
                gnn_input = torch.load(pt)
                gnn_input = [x.to(self.device) for x in gnn_input]
                state_emb = self.model.gnn_encoder(*gnn_input)  # Tensor [D]
                sequence.append(state_emb)
                state_length = state_emb.shape[0]
                type_ids.extend([torch.tensor([1])] * state_emb.size(0))  # type 1 = state
                actions.extend([torch.tensor([-1])] * state_emb.size(0))
                candidates.extend([torch.tensor([-1])] * state_emb.size(0))
                state_embedded = True
                reward_embd = self.model.reward_embedding(torch.tensor([reward_accum], dtype=torch.float32,device=self.device))
                sequence.append(reward_embd.unsqueeze(0))
                type_ids.append(torch.tensor([3]))  # type 3 = reward      
                actions.append(torch.tensor([-1],dtype=torch.long,device=self.device))       
                candidates.append(torch.tensor([-1],dtype=torch.long,device=self.device)) 

            elif "selected" in name and state_embedded:
                node_idx = int(re.findall(r'node_(\d+)_selected', name)[0])
                node_idx_embed = self.model.node_idx_embedding(torch.tensor([node_idx], dtype=torch.long,device=self.device))
                sequence.append(node_idx_embed)
                type_ids.append(torch.tensor([2]))  # type 2 = select
                #actions.append(torch.tensor([node_idx],dtype=torch.long,device=self.device)) 

                # ✅ 找到 node_idx 在 sequence 中的位置（之前被作为 branch_on 加入）
                select_token_position = 1
                for i in range(len(actions)):
                    if type_ids[i] == 5 and actions[i].item() == node_idx:
                        select_token_position = i
                        break
                #assert select_token_position is not None, f"Cannot find node_idx {node_idx} in previous node tokens"
                actions.append(torch.tensor([select_token_position], dtype=torch.long, device=self.device))


                cand_nodes =  torch.load(pt)

                cand_node_idx_in_sequence = []
                for i in range(state_length, len(type_ids)):
                    if type_ids[i] == 5 and actions[i].item() in cand_nodes:
                        cand_node_idx_in_sequence.append(i)

                candidates.append(cand_node_idx_in_sequence)

                reward_accum += 1
                reward_embd = self.model.reward_embedding(torch.tensor([reward_accum], dtype=torch.float32,device=self.device))
                sequence.append(reward_embd.unsqueeze(0))
                type_ids.append(torch.tensor([3]))  # type 3 = reward
                actions.append(torch.tensor([-1],dtype=torch.long,device=self.device))  
                candidates.append(torch.tensor([-1],dtype=torch.long,device=self.device)) 

                

            elif "bestsolfound" in name and state_embedded:
                if(type_ids[-1] != 3):
                    print("error in reward")
                reward_accum -= 100
                reward_embd = self.model.reward_embedding(torch.tensor([reward_accum], dtype=torch.float32,device=self.device))
                sequence[-1] = reward_embd.unsqueeze(0)
            elif "nodeinfeasible" in name and state_embedded:
                if(type_ids[-1] != 3):
                    print("error in reward")
                reward_accum -= 10
                reward_embd = self.model.reward_embedding(torch.tensor([reward_accum], dtype=torch.float32,device=self.device))
                sequence[-1] = reward_embd.unsqueeze(0)

            elif "branchinfo" in name and state_embedded:
                info = torch.load(pt)  # Tensor [2, 8]
                branch_var_idx = info["selected_var_index"]
                cand_var_idx = info["candidate_indices"]

                branch_var_embd = self.model.branch_var_embedding(torch.tensor([branch_var_idx], dtype=torch.long,device=self.device))
                sequence.append(branch_var_embd)
                type_ids.append(torch.tensor([4]))  # type 4 = branch
                actions.append(torch.tensor([branch_var_idx],dtype=torch.long,device=self.device)) 
                candidates.append(cand_var_idx) 
                reward_accum += 1

            elif "branch_on" in name and state_embedded:
                match = re.search(r'branch_on_\d+_to_(\d+)\.pt', name)
                node_idx =int(match.group(1))
                  
                child_node = torch.load(pt)  # Tensor [2, 8]
                child_node_embd = self.model.node_embedding(child_node.to(self.device))
                sequence.append(child_node_embd)
                type_ids.append(torch.tensor([5]))  # type 5 = node
                actions.append(torch.tensor([node_idx])) 
                candidates.append(torch.tensor([-1],dtype=torch.long,device=self.device)) 

        sequence_tensor = torch.stack(sequence) if sequence[0].dim() == 1 else torch.cat(sequence).view(len(type_ids), -1)
        type_ids_tensor = torch.tensor(type_ids, dtype=torch.long)
        actions_tensor = torch.tensor(actions, dtype=torch.long)

        return sequence_tensor, type_ids_tensor, actions_tensor, candidates
    
def bnb_collate(batch, pad_value=0.0):
    """
    batch: list of (sequence_tensor, type_ids_tensor, actions_tensor, candidates)
    """
    sequences, type_ids, actions, candidates = zip(*batch)

    max_len = max(seq.shape[0] for seq in sequences)
    embed_dim = sequences[0].shape[1]

    padded_sequences = []
    padded_typeids = []
    padded_actions = []
    padded_candidates = []
    attention_masks = []

    for seq, tid, act, cand in zip(sequences, type_ids, actions, candidates):
        pad_len = max_len - seq.shape[0]

        padded_seq = F.pad(seq, pad=(0, 0, 0, pad_len), value=pad_value)
        padded_tid = F.pad(tid, pad=(0, pad_len), value=pad_value)
        padded_act = F.pad(act, pad=(0, pad_len), value=pad_value)

        # pad candidates: list[list[int]] → list of list
        padded_cand = cand + [[-1]] * pad_len  # keep consistent with structure

        mask = torch.cat([torch.ones(seq.shape[0]), torch.zeros(pad_len)])

        padded_sequences.append(padded_seq)
        padded_typeids.append(padded_tid)
        padded_actions.append(padded_act)
        padded_candidates.append(padded_cand)
        attention_masks.append(mask)

    return (
        torch.stack(padded_sequences),    # [B, L, D]
        torch.stack(padded_typeids),      # [B, L]
        torch.stack(padded_actions),      # [B, L]
        padded_candidates,                # [B, L] (list[list])
        torch.stack(attention_masks),     # [B, L]
    )



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