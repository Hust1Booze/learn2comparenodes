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

    
    def forward(self, states, sequence_data, device):
        
        select_loss = torch.tensor(0.0, device=device)
        branch_loss = torch.tensor(0.0, device=device)
        select_steps = 0
        branch_steps = 0
        select_corrects = 0
        branch_corrects = 0

        select_sequences = sequence_data["select_sequences"].to(device)
        branch_sequences = sequence_data["branch_sequences"].to(device)
        select_cands = sequence_data["select_cands"].to(device)
        branch_cands = sequence_data["branch_cands"].to(device)
        node_ids = sequence_data["node_ids"].to(device)
        types = sequence_data["types"].to(device)
        select_labels = sequence_data["select_labels"]
        branch_labels = sequence_data["branch_labels"]

        select_sequence_masks = sequence_data["select_sequence_masks"].to(device)
        branch_sequence_masks = sequence_data["branch_sequence_masks"].to(device)
        select_cand_masks = sequence_data["select_cand_masks"].to(device)
        branch_cand_masks = sequence_data["branch_cand_masks"].to(device)
        node_id_masks = sequence_data["node_id_masks"].to(device)


        states_embd, states_mask = self.deal_states(states, device)
        
        _select_sequence_embd,_branch_sequence_embd = self.combine_sequence_embd(select_sequences, branch_sequences, types, device)

        select_sequence_embd = self.transformer(_select_sequence_embd, src_key_padding_mask=select_sequence_masks)
        branch_sequence_embd = self.transformer(_branch_sequence_embd, src_key_padding_mask=branch_sequence_masks)

        select_logits = self.deal_select(select_sequence_embd, select_sequence_masks, states_embd, states_mask)
        branch_logits = self.deal_branch(branch_sequence_embd, branch_sequence_masks, states_embd, states_mask)

        # cal branch loss
        branch_loss, branch_top1, branch_top5, branch_top10 = self.cal_branch_loss(branch_logits, branch_cands, branch_labels)
        select_loss, select_top1, select_top5, select_top10 = self.cal_select_loss(select_logits, select_cands, select_labels, node_ids)

        return branch_loss, select_loss, branch_top1, branch_top5, branch_top10, select_top1, select_top5, select_top10
    

    def deal_states(self, states, device):
        max_state_emb_len = 0
        states_emb = []
        for state in states:
            gnn_input = [x.to(device) for x in state]
            state_emb = self.gnn_encoder(*gnn_input)  # [num_vars, D]
            states_emb.append(state_emb)
            max_state_emb_len = max(max_state_emb_len,state_emb.shape[0])

        # 对states_emb进行padding并转换为tensor
        padded_states_emb = []
        states_emb_masks = []
        
        for state_emb in states_emb:
            state_emb_len = state_emb.shape[0]
            d_model = state_emb.shape[1]
            
            # 创建padded tensor (用0填充)
            padded_state_emb = torch.zeros(max_state_emb_len, d_model, dtype=state_emb.dtype, device=device)
            padded_state_emb[:state_emb_len] = state_emb
            
            # 创建mask (True表示mask位置，False表示正常位置)
            state_emb_mask = torch.ones(max_state_emb_len, dtype=torch.bool, device=device)
            state_emb_mask[:state_emb_len] = False
            
            padded_states_emb.append(padded_state_emb)
            states_emb_masks.append(state_emb_mask)
        
        # 堆叠成batch tensor
        states_emb_tensor = torch.stack(padded_states_emb)  # [batch_size, max_state_emb_len, d_model]
        states_emb_masks_tensor = torch.stack(states_emb_masks)  # [batch_size, max_state_emb_len]

        return states_emb_tensor, states_emb_masks_tensor
    

    def combine_sequence_embd(self,select_sequences, branch_sequences, types, device):
        select_sequence_embd = self.token_proj(select_sequences)
        branch_sequence_embd = self.token_proj(branch_sequences)

        types_embd = self.type_embedding(types)

        # 创建position embedding
        batch_size, select_seq_len, _ = select_sequence_embd.shape
        _, branch_seq_len, _ = branch_sequence_embd.shape
        
        # 创建position indices
        select_positions = torch.arange(select_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        branch_positions = torch.arange(branch_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # 获取position embeddings
        select_pos_embd = self.pos_embedding(select_positions)
        branch_pos_embd = self.pos_embedding(branch_positions)

        # 组合所有embeddings: token + type + position
        select_sequence_embd = select_sequence_embd + types_embd[:,:select_sequence_embd.shape[1],:] + select_pos_embd
        branch_sequence_embd = branch_sequence_embd + types_embd[:,:branch_sequence_embd.shape[1],:] + branch_pos_embd

        return select_sequence_embd,branch_sequence_embd


    def deal_branch(self, branch_sequence_embd, branch_sequence_masks, states_embd, states_mask):
        """
        处理branch序列和state的交互
        Args:
            branch_sequence_embd: [batch_size, seq_len, d_model] - branch序列embeddings
            branch_sequence_masks: [batch_size, seq_len] - branch序列的padding mask
            states_embd: [batch_size, num_vars, d_model] - state embeddings
            states_mask: [batch_size, num_vars] - state的padding mask
        Returns:
            branch_logits: [batch_size,num_vars, 1]
        """
        
        attended_output, attention_weights = \
            self.cross_attention(states_embd, branch_sequence_embd, branch_sequence_embd, states_mask, branch_sequence_masks)
        
        combine_branch_embd = torch.cat([states_embd, attended_output], dim=-1) #[batch, numvars, d_model*2]

        branch_logits = self.branch_head(combine_branch_embd)

        return branch_logits 


    def deal_select(self, select_sequence_embd, select_sequence_masks, states_embd, states_mask):
        """
        处理branch序列和state的交互
        Args:
            branch_sequence_embd: [batch_size, seq_len, d_model] - branch序列embeddings
            branch_sequence_masks: [batch_size, seq_len] - branch序列的padding mask
            states_embd: [batch_size, num_vars, d_model] - state embeddings
            states_mask: [batch_size, num_vars] - state的padding mask
        Returns:
            branch_logits: [batch_size,num_vars, 1]
        """
        
        attended_output, attention_weights = \
            self.cross_attention(select_sequence_embd, states_embd, states_embd, select_sequence_masks, states_mask)
        
        combine_select_embd = torch.cat([select_sequence_embd, attended_output], dim=-1) #[batch, numvars, d_model*2]

        select_logits = self.select_head(combine_select_embd)

        return select_logits 


    def cal_branch_loss(self, branch_logits, branch_cands, branch_labels):
        """
        calculate branch loss
        Args:
            branch_logits: [batch_size, num_vars, 1] - branch logits
            branch_cands: [batch_size, num_candidates] - branch candidates, each is a index of branch_logits
            branch_labels: [batch_size] - branch labels
        Returns:
            branch_loss: scalar - branch loss
        """
        batch_size, num_vars, _ = branch_logits.shape
        num_candidates = branch_cands.shape[1]
        
        # 创建candidate mask: [batch_size, num_vars]
        # True表示该位置是不是candidate， 需要mask
        candidate_mask = torch.ones(batch_size, num_vars, dtype=torch.bool, device=branch_logits.device)
        
        # 使用scatter_来设置candidate位置为True
        batch_indices = torch.arange(batch_size, device=branch_logits.device).unsqueeze(1).expand(-1, num_candidates)
        
        candidate_mask[batch_indices, branch_cands] = False
        
        # 获取logits并应用mask
        logits = branch_logits.squeeze(-1)  # [batch_size, num_vars]
        
        # 将非candidate位置的logits设为-inf
        masked_logits = logits.masked_fill(candidate_mask, float('-inf'))

        batch_indices = torch.arange(batch_size, device=branch_logits.device)
        # labels 就表示label在cands中的位置
        target = branch_cands[batch_indices,branch_labels]      
        # 计算交叉熵损失
        loss = F.cross_entropy(masked_logits, target, reduction='none')  # [batch_size]
        
        # 计算top1, top5, top10准确率
        _, top_indices = masked_logits.topk(k=10, dim=-1)  # [batch_size, 10]
        
        # 检查top1是否包含标签
        top1_correct = (top_indices[:, 0] == target).float().mean().item()
        
        # 检查top5是否包含标签
        top5_correct = torch.any(top_indices[:, :5] == target.unsqueeze(1), dim=1).float().mean().item()
        
        # 检查top10是否包含标签
        top10_correct = torch.any(top_indices == target.unsqueeze(1), dim=1).float().mean().item()
        
        return loss.mean(), top1_correct, top5_correct, top10_correct

    def cal_select_loss(self, select_logits, select_cands, select_labels, node_ids):
        """
        calculate select loss
        Args:
            select_logits: [batch_size, num_seq, 1] - select logits
            select_cands: [batch_size, num_candidates] - select candidates, each is a index of select_logits
            select_labels: [batch_size] - select labels (node id)
            node_ids: [batch_size * seq] - node ids
        Returns:
            select_loss: scalar - select loss
        """

        select_logits = select_logits.squeeze(-1)
        mask = node_ids == -1
        select_logits = select_logits.masked_fill(mask, float('-inf'))

        # 找到select labels 在node_ids中的位置作为target
        batch_size = select_logits.shape[0]
        target = torch.zeros(batch_size, dtype=torch.long, device=select_logits.device)
        
        for i in range(batch_size):
            sample_node_ids = node_ids[i]  # [seq_len]
            label = select_labels[i]  # node id
            
            # 找到action在node_ids中的位置
            action_positions = (sample_node_ids == label).nonzero(as_tuple=True)[0]
            if len(action_positions) > 0:
                target[i] = action_positions[0]  # 取第一个匹配的位置
            else:
                target[i] = 0  # 默认值，会被mask掉
                print(f'label node id {label} not found in node_ids {sample_node_ids}')

        loss = F.cross_entropy(select_logits, target, reduction='none')
        
        # 只计算top1准确率
        _, top_indices = select_logits.topk(k=1, dim=-1)  # [batch_size, 1]
        
        # 检查top1是否包含标签
        top1_correct = (top_indices[:, 0] == target).float().mean().item()
        
        # select任务只关注top1，top5和top10设为0
        top5_correct = 0.0
        top10_correct = 0.0
        
        return loss.mean(), top1_correct, top5_correct, top10_correct
    

    def cross_attention(self, query, key, value, query_mask=None, key_mask=None):
        """
        Cross attention function with padding masks for both query and key
        Args:
            query: [batch_size, seq_len, d_model] - query embeddings
            key: [batch_size, num_vars, d_model] - key embeddings  
            value: [batch_size, num_vars, d_model] - value embeddings
            query_mask: [batch_size, seq_len] - query padding mask
            key_mask: [batch_size, num_vars] - key padding mask (True=padding)
        Returns:
            attended_output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, seq_len, num_vars]
        """
        batch_size, seq_len, d_model = query.shape
        _, num_vars, _ = key.shape
        
        # Calculate attention scores: [batch_size, seq_len, num_vars]
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / (d_model ** 0.5)
        
        # Apply padding masks if provided
        if query_mask is not None or key_mask is not None:
            # 创建attention mask: [batch_size, seq_len, num_vars]
            if query_mask is not None:
                query_mask_expanded = query_mask.unsqueeze(-1)  # [batch_size, seq_len, 1]
            else:
                query_mask_expanded = torch.ones(batch_size, seq_len, 1, dtype=torch.bool, device=query.device)
            
            if key_mask is not None:
                key_mask_expanded = key_mask.unsqueeze(1)  # [batch_size, 1, num_vars]
            else:
                key_mask_expanded = torch.ones(batch_size, 1, num_vars, dtype=torch.bool, device=query.device)
            
            # 只有当query和key都不是padding时，attention score才有效
            attention_mask = query_mask_expanded & key_mask_expanded  # [batch_size, seq_len, num_vars]
            
            # Apply mask
            attention_scores = attention_scores.masked_fill(attention_mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to values
        attended_output = torch.matmul(attention_weights, value)
        
        return attended_output, attention_weights

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

    def get_select_node_decision(self, state_embd, sequence_tensor, types, node_id):
        sequence_embd = self.get_inference_sequence_embd(sequence_tensor, types, node_id)
        sequence_embd = self.transformer(sequence_embd.unsqueeze(0))
        select_logits = self.deal_select(sequence_embd.unsqueeze(0), None, state_embd.unsqueeze(0), None)

        return select_logits
        
    def get_branch_var_decision(self, state_embd, sequence_tensor, types, node_id):

        with torch.no_grad():  # 用于推理但不训练
            sequence_embd = self.get_inference_sequence_embd(sequence_tensor, types, node_id)
            sequence_embd = self.transformer(sequence_embd)
            branch_logits = self.deal_branch(sequence_embd, None, state_embd.unsqueeze(0), None)

        return branch_logits

    def get_inference_sequence_embd(self, sequence_tensor, types, node_id):

        ######### step 1 merge input embd ############
        sequence_embd = self.token_proj(sequence_tensor)

        types_embd = self.type_embedding(types)

        seq_len, _ = sequence_embd.shape
        
        # 创建position indices
        positions = torch.arange(seq_len, device=sequence_embd.device).unsqueeze(0)
        
        # 获取position embeddings
        pos_embd = self.pos_embedding(positions)

        # 组合所有embeddings: token + type + position
        sequence_embd = sequence_embd + types_embd + pos_embd

        return sequence_embd

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