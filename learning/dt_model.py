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
    def __init__(self,d_model=32, n_heads=4, n_layers=2, dropout=0.1, type_vocab_size=6, temperature = 1000.0, use_soft_score_label = True):
        super().__init__()
        self.token_proj = nn.Linear(8, d_model)  # project all input tokens to d_model dim
        self.type_embedding = nn.Embedding(type_vocab_size, d_model, padding_idx=-1)
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

    def generate_causal_mask(self, length, device):
        return torch.triu(torch.ones(length, length, device=device), diagonal=1).bool()

    def forward(self, batch_tokens, type_ids, attention_mask, actions, candidates, branch_scores):
        B, T, D = batch_tokens.shape
        device = batch_tokens.device

        type_embed = self.type_embedding(type_ids)
        pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_embed = self.pos_embedding(pos_ids)

        x = batch_tokens + type_embed + pos_embed

        # 🔥 优化：一次性计算整个批次的编码，使用因果掩码
        causal_mask = self.generate_causal_mask(T, device)
        pad_mask = attention_mask == 0
        
        # 使用因果掩码进行批量编码，避免重复计算
        encoded = self.transformer(x, mask=causal_mask, src_key_padding_mask=pad_mask)  # [B, T, D]

        select_loss = torch.tensor(0.0, device=device)
        branch_loss = torch.tensor(0.0, device=device)
        select_steps = 0
        branch_steps = 0
        select_correct = 0
        branch_correct = 0

        # 🔥 优化：批量收集所有需要处理的位置
        select_positions = []  # 存储 (b, t, candidate_indices)
        branch_positions = []  # 存储 (b, t, candidate_indices, branch_step)
        
        for b in range(B):
            cur_branch_steps = 0
            for t in range(T):
                token_type = type_ids[b, t].item()
                if token_type not in [2, 4] or actions[b, t] < 0:
                    continue
                if not isinstance(candidates[b][t], list) or len(candidates[b][t]) <= 1:
                    continue

                candidate_indices = candidates[b][t]
                
                if token_type == 2:  # 节点选择
                    select_positions.append((b, t, candidate_indices))
                elif token_type == 4:  # 分支变量选择
                    branch_positions.append((b, t, candidate_indices, cur_branch_steps))
                    cur_branch_steps += 1

        # 🔥 批量处理节点选择任务
        if select_positions:
            for b, t, candidate_indices in select_positions:
                candidate_tensor = torch.tensor(candidate_indices, device=device, dtype=torch.long)
                # 直接使用预计算的编码，而不是重新计算transformer
                candidate_repr = encoded[b, candidate_tensor]  # [num_cand, D]
                
                logits = self.select_head(candidate_repr).squeeze(-1)
                target_pos = (candidate_tensor == actions[b, t]).nonzero(as_tuple=True)[0]
                
                if len(target_pos) > 0:
                    select_loss += F.cross_entropy(logits.unsqueeze(0), target_pos)
                    select_steps += 1
                    
                    pred_pos = logits.argmax().item()
                    if pred_pos == target_pos.item():
                        select_correct += 1

        # 🔥 批量处理分支变量选择任务
        if branch_positions:
            for b, t, candidate_indices, branch_step in branch_positions:
                candidate_tensor = torch.tensor(candidate_indices, device=device, dtype=torch.long)
                # 直接使用预计算的编码，而不是重新计算transformer
                candidate_repr = encoded[b, candidate_tensor]  # [num_cand, D]
                
                # 检查长度一致性
                assert len(candidate_repr) == len(branch_scores[b][branch_step]), \
                    f"Length mismatch in B-T: {b}-{t} branch_steps: {branch_step}"

                logits = self.branch_head(candidate_repr).squeeze(-1)

                # 使用软标签或硬标签
                if self.use_soft_score_label:
                    scores = torch.tensor(branch_scores[b][branch_step], device=device, dtype=torch.float32)
                    branch_loss += F.kl_div(
                        F.log_softmax(logits, dim=-1),
                        F.softmax(scores/self.temperature, dim=-1),
                        reduction='batchmean'
                    )
                else:
                    target_pos = (candidate_tensor == actions[b, t]).nonzero(as_tuple=True)[0]
                    if len(target_pos) > 0:
                        branch_loss += F.cross_entropy(logits.unsqueeze(0), target_pos)

                branch_steps += 1
                
                # 计算准确率
                target_pos = (candidate_tensor == actions[b, t]).nonzero(as_tuple=True)[0]
                if len(target_pos) > 0:
                    pred_pos = logits.argmax().item()
                    if pred_pos == target_pos.item():
                        branch_correct += 1

        # 计算平均损失和准确率
        if select_steps > 0:
            select_loss = select_loss / select_steps
        if branch_steps > 0:
            branch_loss = branch_loss / branch_steps

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