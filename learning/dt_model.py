import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dt_dataset import BnBSequentialDataset, bnb_collate
from torch_geometric.nn import GraphConv
import random

class GNNEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.emb_size = emb_size = 32 #uniform node feature embedding dim
        
        hidden_dim1 = 32
        hidden_dim2 = 32
        hidden_dim3 = 32
        
        # static data
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 6
        

        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
        )
        
        self.bounds_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(2),
            torch.nn.Linear(2,2),
            torch.nn.ReLU(),
        )

        self.convs = []
        self.conv1 = GraphConv((emb_size, emb_size), hidden_dim1 )
        self.conv2 = GraphConv((hidden_dim1, hidden_dim1), hidden_dim2 )
        self.conv3 = GraphConv((hidden_dim2, hidden_dim2), hidden_dim3 )
        
        self.convs = [self.conv1, self.conv2]
        
        out_size = hidden_dim3 if len(self.convs)==3 else emb_size
        
        self.final_mlp = torch.nn.Sequential( 
                            torch.nn.Linear(2*out_size+2, 1, bias=False),
                            torch.nn.Sigmoid()
                            )
        
       
    def forward(self, constraint_features, edge_indices, edge_features, 
                       variable_features, bbounds, depth):

        #Assume edge indice var to cons, constraint_mask of shape [Nconvs]       
        variable_features = self.var_embedding(variable_features)
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        bbounds = self.bounds_embedding(bbounds)
         
        edge_indices_reversed = torch.stack([edge_indices[1], edge_indices[0]], dim=0)
              
        for conv in self.convs:           
            #Var to cons
            constraint_features_next = F.relu(conv((variable_features, constraint_features), 
                                              edge_indices,
                                              edge_weight=edge_features,
                                              size=(variable_features.size(0), constraint_features.size(0))))
            
            #cons to var 
            variable_features = F.relu(conv((constraint_features, variable_features), 
                                      edge_indices_reversed,
                                      edge_weight=edge_features,
                                      size=(constraint_features.size(0), variable_features.size(0))))
            
            constraint_features = constraint_features_next
            
            constraint_avg = torch.mean(constraint_features, axis=0, keepdim=True)
            variable_avg = torch.mean(variable_features, axis=0, keepdim=True)
            
        #return torch.cat((variable_avg, constraint_avg, bbounds), dim=1)
        return variable_features
    


class DTModel(nn.Module):
    def __init__(self,d_model=32, n_heads=4, n_layers=2, dropout=0.1, type_vocab_size=6, temperature = 1000.0, use_soft_score_label = False):
        super().__init__()
        self.d_model = d_model  # 保存d_model参数
        self.token_proj = nn.Linear(8, d_model)  # project all input tokens to d_model dim
        self.type_embedding = nn.Embedding(type_vocab_size, d_model, padding_idx=0)
        self.pos_embedding = nn.Embedding(10000, d_model) # support 10000 sequence length 
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.max_nodes = 10000
        self.max_vars = 10000

        self.temperature = temperature
        self.use_soft_score_label = use_soft_score_label

        self.node_idx_embedding = nn.Embedding(self.max_nodes, d_model)         # for node_idx
        self.branch_var_embedding = nn.Embedding(self.max_vars, d_model)        # for branch var
        self.reward_embedding = nn.Linear(1, d_model)                           # for reward (scalar → vector)
        self.node_embedding = nn.Linear(8, d_model)  

        self.gnn_encoder = GNNEncoder()
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)  # example: predict score or class
        )

        self.select_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        self.branch_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def generate_causal_mask(self, sz, device):
        """生成因果掩码，防止看到未来的token"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask
    
    def process_sequence_data(self, sequence_data_batch, type_ids, device, actions=None, candidates=None):
        """
        将原始数据batch转换为嵌入向量batch，并处理type_ids展平和padding
        同时处理actions和candidates的重映射
        sequence_data_batch: [B, L] list of list of dict
        type_ids: [B, L] tensor
        actions: [B, L] tensor (optional)
        candidates: [B, L] list of list (optional)
        returns: (padded_embeddings, padded_type_ids, padded_attention_mask, position_mapping, padded_actions, padded_candidates)
        """
        B = len(sequence_data_batch)
        L = len(sequence_data_batch[0])
        D = self.d_model
        
        # 将所有嵌入展平并记录索引映射
        flattened_embeddings = []
        position_mapping = []  # 记录每个原始位置对应的展平后的索引范围
        flattened_type_ids = []
        flattened_actions = []
        flattened_candidates = []
        
        for b in range(B):
            batch_flat = []
            batch_mapping = []
            batch_type_ids = []
            batch_actions = []
            batch_candidates = []
            current_pos = 0
            
            for t in range(L):
                data_item = sequence_data_batch[b][t]
                data_type = data_item['type']
                
                if data_type == 'gnn_state':
                    gnn_input = data_item['data']
                    # 将GNN输入移到正确的设备
                    gnn_input = [x.to(device) for x in gnn_input]
                    state_emb = self.gnn_encoder(*gnn_input)  # [num_vars, D]
                    # 保持原始维度，不取平均
                    emb = state_emb
                    
                elif data_type == 'reward':
                    reward_value = data_item['data']
                    reward_tensor = torch.tensor([reward_value], dtype=torch.float32, device=device)
                    reward_emb = self.reward_embedding(reward_tensor)  # [1, D]
                    emb = reward_emb  # [1, D]
                    
                elif data_type == 'node_idx':
                    node_idx = data_item['data']
                    node_idx_tensor = torch.tensor([node_idx], dtype=torch.long, device=device)
                    node_idx_emb = self.node_idx_embedding(node_idx_tensor)  # [1, D]
                    emb = node_idx_emb  # [1, D]
                    
                elif data_type == 'branch_var':
                    branch_var_idx = data_item['data']
                    branch_var_tensor = torch.tensor([branch_var_idx], dtype=torch.long, device=device)
                    branch_var_emb = self.branch_var_embedding(branch_var_tensor)  # [1, D]
                    emb = branch_var_emb  # [1, D]
                    
                elif data_type == 'child_node':
                    child_node_data = data_item['data']
                    child_node_tensor = child_node_data.to(device)
                    child_node_emb = self.node_embedding(child_node_tensor)  # [D]
                    emb = child_node_emb  # [1, D]
                    
                elif data_type == 'padding':
                    # 对于padding，添加零向量
                    emb = torch.zeros(1, D, device=device)  # [1, D]
                
                # 确保所有嵌入都是2D的
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                
                batch_flat.append(emb)
                
                # 记录位置映射
                start_pos = current_pos
                end_pos = current_pos + emb.size(0)
                batch_mapping.append((start_pos, end_pos))
                current_pos = end_pos
                
                # 为这个时间步的所有token重复相同的type_id
                token_len = emb.size(0)
                batch_type_ids.extend([type_ids[b, t].item()] * token_len)
                
                # 处理actions和candidates的重映射
                if actions is not None and candidates is not None:
                    original_action = actions[b, t].item()
                    original_candidates = candidates[b][t]
                    
                    # 为这个时间步的所有token重复相同的action和candidates
                    batch_actions.extend([original_action] * token_len)
                    batch_candidates.extend([original_candidates] * token_len)
                else:
                    batch_actions.extend([-1] * token_len)
                    batch_candidates.extend([[-1]] * token_len)
            
            # 将这个batch的所有嵌入和相关数据拼接
            flattened_embeddings.append(torch.cat(batch_flat, dim=0))  # [total_tokens, D]
            flattened_type_ids.append(batch_type_ids)
            flattened_actions.append(batch_actions)
            flattened_candidates.append(batch_candidates)
            position_mapping.append(batch_mapping)
        
        # 找到最大长度并进行padding
        max_len = max(emb.size(0) for emb in flattened_embeddings)
        padded_embeddings = torch.zeros(B, max_len, D, device=device)
        padded_attention_mask = torch.zeros(B, max_len, device=device)
        padded_type_ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
        padded_actions = torch.zeros(B, max_len, dtype=torch.long, device=device) if actions is not None else None
        padded_candidates = []
        
        for b in range(B):
            seq_len = flattened_embeddings[b].size(0)
            padded_embeddings[b, :seq_len] = flattened_embeddings[b]
            padded_attention_mask[b, :seq_len] = 1
            
            # 填充type_ids
            for i, type_id in enumerate(flattened_type_ids[b]):
                padded_type_ids[b, i] = type_id
            
            # 填充actions
            if actions is not None:
                for i, action in enumerate(flattened_actions[b]):
                    padded_actions[b, i] = action
                # 填充剩余位置为-1
                padded_actions[b, seq_len:] = -1
            
            # 填充candidates
            batch_candidates = flattened_candidates[b] + [[-1]] * (max_len - seq_len)
            padded_candidates.append(batch_candidates)
                    
        return padded_embeddings, padded_type_ids, padded_attention_mask, position_mapping, padded_actions, padded_candidates

    def forward(self, sequence_data_batch, type_ids, attention_mask, actions, candidates, branch_scores):
        # 首先将原始数据转换为嵌入向量
        device = type_ids.device
        padded_embeddings, padded_type_ids, padded_attention_mask, position_mapping, padded_actions, padded_candidates = self.process_sequence_data(sequence_data_batch, type_ids, device, actions, candidates)
        
        # 使用padded embeddings进行transformer编码
        type_embed = self.type_embedding(padded_type_ids)
        pos_ids = torch.arange(padded_embeddings.size(1), device=device).unsqueeze(0).expand(padded_embeddings.size(0), -1)
        pos_embed = self.pos_embedding(pos_ids)

        x = padded_embeddings + type_embed + pos_embed

        select_loss = torch.tensor(0.0, device=device)
        branch_loss = torch.tensor(0.0, device=device)
        select_steps = 0
        branch_steps = 0
        select_correct = 0
        branch_correct = 0

        # 现在使用padded的sequence进行处理
        B, T = padded_type_ids.shape
        
        for b in range(B):  # 遍历 batch 中的每个 sample
            branch_steps = 0
            branch_ids = -1
            for t in range(T):  # 遍历该 sample 的所有时间步
                token_type = padded_type_ids[b, t].item()
                if token_type ==4: 
                    branch_ids += 1
                if token_type not in [2, 4]:
                    continue
                if padded_actions[b, t] < 0:
                    continue
                if not isinstance(padded_candidates[b][t], list) or len(padded_candidates[b][t]) <= 1:
                    continue

                x_prefix = x[b:b+1, :t, :]  # [1, t, D]

                encoded = self.transformer(x_prefix)  # [1, t, D]

                candidate_indices = padded_candidates[b][t]
                candidate_tensor = torch.tensor(candidate_indices, device=device, dtype=torch.long)
                candidate_repr = encoded[0, candidate_tensor]  # [num_cand, D]

                if token_type == 2:  # 节点选择
                    logits = self.select_head(candidate_repr).squeeze(-1)  # [num_cand]
                    target_pos = (candidate_tensor == padded_actions[b, t]).nonzero(as_tuple=True)[0]
                    if len(target_pos) > 0:
                        select_loss += F.cross_entropy(logits.unsqueeze(0), target_pos)
                        select_steps += 1
                    
                    pred_pos = logits.argmax().item()
                    if pred_pos == target_pos.item():
                        select_correct += 1

                elif token_type == 4:  # 分支变量选择

                    # 使用 assert 检查长度是否一致
                    assert len(candidate_repr) == len(branch_scores[b][branch_steps]), \
                    f"\n Length mismatch in B - T :{b} - {t} branch_steps: {branch_steps} \
                        \n candidate_indices : {candidate_indices} and\n scores ({branch_scores[b][branch_steps]})"
                    
                    if len(candidate_repr) != len(branch_scores[b][branch_steps]):
                        print("error in train branch, candidiates not equal to socres")
                    logits = self.branch_head(candidate_repr).squeeze(-1)  # [num_cand]
                    scores = torch.tensor(branch_scores[b][branch_steps], device=device, dtype=torch.float32)
                    
                    if self.use_soft_score_label:
                        # 使用 KL 散度损失
                        branch_loss += F.kl_div(
                            F.log_softmax(logits, dim=-1),
                            F.softmax(scores, dim=-1),
                            reduction='batchmean'
                        )
                    else:
                        # 使用硬标签计算损失（交叉熵）
                        target_pos = (candidate_tensor == padded_actions[b, t]).nonzero(as_tuple=True)[0]
                        if len(target_pos) > 0:
                            branch_loss += F.cross_entropy(logits.unsqueeze(0), target_pos)

                    branch_steps += 1

                    # 计算准确率（使用硬标签）
                    target_pos = (candidate_tensor == padded_actions[b, t]).nonzero(as_tuple=True)[0]
                    if len(target_pos) > 0:
                        pred_pos = logits.argmax().item()
                        if pred_pos == target_pos.item():
                            branch_correct += 1

        # 计算平均损失
        if select_steps > 0:
            select_loss = select_loss / select_steps
        if branch_steps > 0:
            branch_loss = branch_loss / branch_steps

        # 计算准确率
        select_acc = select_correct / max(select_steps, 1)
        branch_acc = branch_correct / max(branch_steps, 1)

        return select_loss, branch_loss, select_steps, branch_steps, select_acc, branch_acc
    

    
    def get_select_node_decision(self, sequence, type_ids, candidates, actions):
        device = sequence.device
        T,D = sequence.shape
        type_embed = self.type_embedding(type_ids)
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        pos_embed = self.pos_embedding(pos_ids)

        #pos_embed = pos_embed.squeeze(0)
        x = sequence + type_embed + pos_embed

        encoded = self.transformer(x)
        candidate_indices = candidates[-1]

        logits = self.select_head(encoded).squeeze(-1)  # [num_cand]


        mask = (type_ids == 5)
        if mask.any():
            selected_logits = logits.squeeze(0)[mask]  # 取出 type==5 对应的 logits
            sorted_indices = torch.argsort(selected_logits, descending=True)  # 排序索引（大到小）
            
            # 找出原始 logits 中满足 mask 的 indices
            original_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
            
            # 根据排序后的 logits 索引到对应的 action
            sorted_actions = actions[original_indices[sorted_indices]]
            return sorted_actions
        else:
            print("why no nodes!")
            return []
        
    def get_branch_var_decision(self, sequence, type_ids, candidates, actions):
        device = sequence.device
        T,D = sequence.shape
        type_embed = self.type_embedding(type_ids)
        pos_ids = torch.arange(T, device=device).unsqueeze(0)
        pos_embed = self.pos_embedding(pos_ids)

        #pos_embed = pos_embed.squeeze(0)
        x = sequence + type_embed + pos_embed

        encoded = self.transformer(x)
        candidate_indices = candidates[-1]

        logits = self.branch_head(encoded).squeeze(-1)  # [num_cand]


        mask = (type_ids == 1)
        if mask.any():
            vars_logits = logits.squeeze(0)[mask]  # 取出 type==5 对应的 logits
            
            return vars_logits
        else:
            print("why no vars!")
            return []