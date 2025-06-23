import torch
import torch.nn as nn
import torch.nn.functional as F
from gnnencoder import GNNEncoder


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
            nn.Linear(2*d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

        self.branch_head = nn.Sequential(
            nn.Linear(2*d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    
    def forward(self, sequence_data, device):
        
        select_loss = torch.tensor(0.0, device=device)
        branch_loss = torch.tensor(0.0, device=device)
        select_steps = 0
        branch_steps = 0
        select_corrects = 0
        branch_corrects = 0

        for sequence in sequence_data:  # 遍历 batch 中的每个 sample
            squence_emb = None
            # record each item in sequence is node or not, node with its id, else -1
            sequence_node_id = []
            state_emb = None
            for item in sequence:
                if item is None:
                    continue
                type = item['type']
                if type == 'state':
                    gnn_input = item['data']
                    # 将GNN输入移到正确的设备
                    gnn_input = [x.to(device) for x in gnn_input]
                    state_emb = self.gnn_encoder(*gnn_input)  # [num_vars, D]

                    
                elif type == 'select':
                    candidates = item['candidate']
                    select_node_id = item['data']
                    if select_node_id == 1:
                        node_id_tensor = torch.tensor([1], dtype=torch.long, device=device)
                        node_id_emb = self.node_idx_embedding(node_id_tensor)  # [1, D]

                        #first token
                        squence_emb = node_id_emb
                        sequence_node_id.append(-1)
                    else:
                        select_logits = self.cal_select_logits(state_emb, squence_emb)
                        loss = self.cal_select_loss(select_logits, sequence_node_id, select_node_id, candidates[0], device)
                        select_loss += loss
                        select_steps += 1
                        
                        # 计算select正确率
                        if loss is not None:
                            # 获取预测结果
                            pred_logits = select_logits[candidates[0]].squeeze(-1)  # [num_candidates]
                            pred_idx = pred_logits.argmax().item()
                            true_idx = candidates[0].index(select_node_id)
                            
                            if pred_idx == true_idx:
                                select_corrects += 1
                    
                        node_id_tensor = torch.tensor([select_node_id], dtype=torch.long, device=device)
                        node_id_emb = self.node_idx_embedding(node_id_tensor)  # [1, D]
                        squence_emb = torch.cat([squence_emb, node_id_emb], dim=0)
                        sequence_node_id.append(-1)
                    
                elif type == 'branch':
                    candidates = item['candidate']
                    branch_var_idx = torch.tensor([item['data']], dtype=torch.long, device=device)

                    branch_logits = self.cal_branch_logits(state_emb, squence_emb)

                    candidates = torch.tensor(candidates, device=device, dtype=torch.long)
                    cand_branch_logits = branch_logits[candidates]
                    branch_loss += F.cross_entropy(cand_branch_logits.permute(1, 0), branch_var_idx)
                    branch_steps += 1
                    
                    # 计算branch正确率
                    pred_idx = cand_branch_logits.argmax().item()

                    if pred_idx == branch_var_idx.item():
                        branch_corrects += 1

                    branch_var_tensor = torch.tensor([candidates[branch_var_idx]], dtype=torch.long, device=device)
                    branch_var_emb = self.branch_var_embedding(branch_var_tensor)  # [1, D]
                    squence_emb = torch.cat([squence_emb, branch_var_emb], dim=0)
                    sequence_node_id.append(-1)
                    
                elif type == 'node':
                    node_id = item['node_id']
                    child_node_emb = self.node_embedding(item['data'].to(device))  # [D]
                    squence_emb = torch.cat([squence_emb, child_node_emb], dim=0)
                    sequence_node_id.append(node_id)

                elif type == 'reward':
                    continue
                    reward_value = item['data']
                    reward_tensor = torch.tensor([reward_value], dtype=torch.float32, device=device)
                    reward_emb = self.reward_embedding(reward_tensor)  # [1, D]
                    squence_emb = torch.cat([squence_emb, reward_emb], dim=0)

        return select_loss, branch_loss, select_steps, branch_steps, select_corrects, branch_corrects
    

    def cross_attention(self, query, key, value, mask=None):
        """
        Cross attention function
        Args:
            query: [batch_size, seq_len, d_model] - sequence embeddings
            key: [batch_size, num_vars, d_model] - state embeddings  
            value: [batch_size, num_vars, d_model] - state embeddings
            mask: [batch_size, seq_len, num_vars] - attention mask
        Returns:
            attended_output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, seq_len, num_vars]
        """
        seq_len, d_model = query.shape
        num_vars, _ = key.shape
        
        # Calculate attention scores
        # [batch_size, seq_len, num_vars]
        attention_scores = torch.matmul(query, key.transpose(-1, 0)) / (d_model ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to values
        attended_output = torch.matmul(attention_weights, value)
        
        return attended_output

    def cal_select_loss(self, select_logits, sequence_node_id, select_node_id, candidates, device):
        """
        计算选择节点的交叉熵损失
        Args:
            select_logits: [N, 1] - 每个位置的logits
            sequence_node_id: list - 每个位置对应的节点ID
            select_node_id: int - 目标节点ID
            candidates: list - 候选位置索引
            device: torch.device
        Returns:
            loss: torch.Tensor - 交叉熵损失，如果无法计算则返回None
        """
        # 找到select_node_id在sequence_node_id中的位置
        target_positions = [i for i, node_id in enumerate(sequence_node_id) if node_id == select_node_id]
        
        if not target_positions:
            print(f"Warning: select_node_id {select_node_id} not found in sequence_node_id")
            return None
            
        if not candidates:
            print(f"Warning: candidates list is empty")
            return None
         
        # 找到目标在候选中的索引
        target_candidate_idx = candidates.index(select_node_id)
        
        # 提取候选位置的logits
        candidate_logits = select_logits[candidates].squeeze(-1)  # [num_candidates]
        
        # 计算交叉熵损失
        loss = F.cross_entropy(
            candidate_logits.unsqueeze(0),  # [1, num_candidates]
            torch.tensor([target_candidate_idx], device=device)  # [1]
        )
        
        return loss

    def cal_branch_logits(self, state, sequence):
        x = self.transformer(sequence.unsqueeze(0)).squeeze(0)  # [N, F]
        attended_output = self.cross_attention(state, x, x) #[V,F]

        # 方法1: 使用平均池化 (Mean Pooling)
        attended_output = attended_output.mean(dim=0, keepdim=True)  # [1, F]
        # 将 attended_output 从 [1, F] 扩展为 [N, F]
        N = state.size(0)  # 获取序列长度
        attended_output = attended_output.expand(N, -1)  # [N, F]
        # 将 x 和 attended_output 拼接
        combined_features = torch.cat([state, attended_output], dim=1)  # [N, 2F]        

        logits = self.branch_head(combined_features) #[N,1]

        return logits

    def cal_select_logits(self, state, sequence):

        x = self.transformer(sequence.unsqueeze(0)).squeeze(0)  # [N, F]
        attended_output = self.cross_attention(x, state, state) #[V,F]
        # 方法1: 使用平均池化 (Mean Pooling)
        attended_output = attended_output.mean(dim=0, keepdim=True)  # [1, F]
        # 将 attended_output 从 [1, F] 扩展为 [N, F]
        N = x.size(0)  # 获取序列长度
        attended_output = attended_output.expand(N, -1)  # [N, F]
        # 将 x 和 attended_output 拼接
        combined_features = torch.cat([x, attended_output], dim=1)  # [N, 2F]

        logits = self.select_head(combined_features) #[N,1]

        return logits

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

    def print_model_info(self):
        """
        打印模型的参数量信息
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"=" * 50)
        print(f"DTModel 参数统计:")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (假设float32)")
        print(f"=" * 50)
        
        # 详细打印每个模块的参数量
        print(f"\n详细参数分布:")
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # 只打印叶子模块
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    print(f"{name}: {params:,} 参数")
        
        return total_params, trainable_params