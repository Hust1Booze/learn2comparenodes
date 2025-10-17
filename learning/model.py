import torch
import torch.nn as nn
import torch.nn.functional as F
from gnn_encoder import GNNEncoder
import random
import math
# from xformers.ops import memory_efficient_attention


class DTModel(nn.Module):
    def __init__(self,d_model=32, n_heads=2, n_layers=1, dropout=0.1, type_vocab_size=6, temperature = 1000.0, use_soft_score_label = False):
        super().__init__()
        self.d_model = d_model  # 保存d_model参数
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads
        # 自定义多头注意力所需的投影层
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        self.node_embedding = nn.Linear(8, d_model)  
        self.lp_embedding = nn.Linear(19, d_model)
        self.gnn_encoder = GNNEncoder()

        self.branch_head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, 1, bias=False),
        )

        self.node_branch_concat = torch.nn.Linear(d_model*2, d_model)
        
        # RoPE相关参数
        self.max_seq_len = 10000
        self.rope_theta = 10000.0

        self.gcnn_logic = True

    def forward(self, batch, device):
        
        branch_loss = torch.tensor(0.0, device=device)
        select_loss = torch.tensor(0.0, device=device)
        branch_top1 = 0.0
        branch_top5 = 0.0
        branch_top10 = 0.0
        select_top1 = 0.0
        select_top5 = 0.0
        select_top10 = 0.0

        branch_count = 0
        for sequence in batch:
            if self.gcnn_logic:
                _branch_loss, _branch_top1, _branch_top5 ,_branch_count = self.GCNN_train_logic(sequence, device)
            else:
                _branch_loss, _branch_top1, _branch_top5 ,_branch_count = self.sequence_train_logic(sequence, device)     
                
            branch_loss += _branch_loss
            branch_top1 += _branch_top1
            branch_top5 += _branch_top5
            branch_count += _branch_count         

        return branch_loss / branch_count, select_loss, branch_top1/branch_count, branch_top5/branch_count, branch_top10, select_top1, select_top5, select_top10
        
    def GCNN_train_logic(self, sequence, device):
        # GCNN logits
        # 1️⃣ 先筛出这个 sequence 里所有 branch 数据
        branch_top1 =0;
        branch_top5 =0;
        branch_count =0;
        loss = torch.tensor(0.0, device=device)
        branch_data_list = [d for d in sequence if d['type'] == 'branch']
        if len(branch_data_list) == 0:
            return loss, branch_top1, branch_top5 ,branch_count
        # 3️⃣ 随机选一个 branch
        data = random.choice(branch_data_list)
        branch_label = data['branch_label']
        branch_cand = data['branch_cand']
        col_features = data['col_features'].to(device)
        row_features = data['row_features'].to(device)
        edge_attr = data['edge_attr'].to(device)
        edge_index = data['edge_index'].to(device)
        branch_node = data['branch_node']

        logits, variable_embd = self.gnn_encoder(row_features, edge_index, edge_attr, col_features)
        cand_logits = logits[branch_cand]
        loss = torch.nn.functional.cross_entropy(
            cand_logits.unsqueeze(0),
            torch.tensor(branch_label, dtype=torch.long, device=device).unsqueeze(0)
        )
        # --- 计算准确率 ---
        pred = cand_logits.argmax().item()
        if pred == branch_label:
            branch_top1 += 1
        # top-5 正确率
        k = min(5, cand_logits.size(0))
        top5_preds = cand_logits.topk(k).indices.tolist()
        if branch_label in top5_preds:
            branch_top5 += 1

        branch_count +=1
        
        return loss, branch_top1, branch_top5 ,branch_count

    def sequence_train_logic(self, sequence, device):
        branch_top1 =0;
        branch_top5 =0;
        branch_count =0;        
        total_loss = torch.tensor(0.0, device=device)

        sequence_embd_list = []
        node_embd_dict = {}
        for data in sequence:
            if data['type'] == 'node':
                node_embd = self.node_embedding(data['node_data'].to(device))
                node_number = data['node_number']
                sequence_embd_list.append(node_embd)
                node_embd_dict[node_number] = node_embd
            elif data['type'] == 'branch':
                branch_label = data['branch_label']
                branch_cand = data['branch_cand']
                col_features = data['col_features'].to(device)
                row_features = data['row_features'].to(device)
                edge_attr = data['edge_attr'].to(device)
                edge_index = data['edge_index'].to(device)
                branch_node = data['branch_node']

                logits, variable_embd = self.gnn_encoder(row_features, edge_index, edge_attr, col_features)

                if len(sequence_embd_list) != 0:
                    # use sequence as KV
                    cur_sequence_embd = self.deal_sequence(sequence_embd_list)
                    attention_variable_embd = self.cross_attention(variable_embd, cur_sequence_embd)
                    logits = self.branch_head(attention_variable_embd).squeeze(-1)

                cand_logits = logits[branch_cand]
                loss = torch.nn.functional.cross_entropy(
                    cand_logits.unsqueeze(0),
                    torch.tensor(branch_label, dtype=torch.long, device=device).unsqueeze(0)
                )

                total_loss += loss
                branch_count += 1

                # 选出logits最大的候选，并取其对应的variable embedding
                best_cand_idx = cand_logits.argmax().item()
                best_var_idx = branch_cand[best_cand_idx]
                selected_variable_embd = variable_embd[best_var_idx]
                sequence_embd_list.append(self.node_branch_concat(torch.cat([selected_variable_embd, node_embd_dict[branch_node]])))
                

                # --- 计算准确率 ---
                pred = cand_logits.argmax().item()
                if pred == branch_label:
                    branch_top1 += 1
                # top-5 正确率
                k = min(5, cand_logits.size(0))
                top5_preds = cand_logits.topk(k).indices.tolist()
                if branch_label in top5_preds:
                    branch_top5 += 1

        return total_loss, branch_top1, branch_top5 ,branch_count
    def _rope_cache(self, seq_len, device):
        # 生成cos/sin缓存，形状: (1, 1, seq_len, head_dim/2) -> 之后repeat_interleave到head_dim
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, half_dim, device=device).float() / half_dim))
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum('n,d->nd', t, inv_freq)  # (seq_len, half_dim)
        cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,half_dim)
        sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,half_dim)
        # 扩展到head_dim
        cos = torch.repeat_interleave(cos, 2, dim=-1)  # (1,1,seq_len,head_dim)
        sin = torch.repeat_interleave(sin, 2, dim=-1)  # (1,1,seq_len,head_dim)
        return cos, sin

    @staticmethod
    def _rotate_half(x):
        # 交换偶/奇维度并对奇维度取负: (.., 2i) , (.., 2i+1)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack((-x_odd, x_even), dim=-1)
        return x_rot.reshape(x.shape)

    def _apply_rope_qk(self, q, k):
        # q,k: (b, h, n, head_dim)
        b, h, n, d = q.shape
        device = q.device
        cos, sin = self._rope_cache(n, device)
        # 广播到(b,h,n,d)
        cos = cos.expand(b, h, n, d)
        sin = sin.expand(b, h, n, d)
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)
        return q_rot, k_rot

    def _apply_rope_x(self, x):
        # 仅对单个张量应用RoPE，x: (b, h, n, d)
        b, h, n, d = x.shape
        device = x.device
        cos, sin = self._rope_cache(n, device)
        cos = cos.expand(b, h, n, d)
        sin = sin.expand(b, h, n, d)
        return (x * cos) + (self._rotate_half(x) * sin)

    def self_attention(self, sequence_embd):
        """
        多头自注意力（包含RoPE到Q/K）
        Args:
            sequence_embd: (seq_len, d_model)
        Returns:
            (seq_len, d_model)
        """
        seq_len, _ = sequence_embd.shape
        x = sequence_embd.unsqueeze(0)  # (1, seq_len, d_model)
        # 线性投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # 变形为多头 (b, n, h, d_h) -> (b, h, n, d_h)
        b = q.size(0)
        h = self.n_heads
        d_h = self.head_dim
        q = q.view(b, seq_len, h, d_h).transpose(1, 2)
        k = k.view(b, seq_len, h, d_h).transpose(1, 2)
        v = v.view(b, seq_len, h, d_h).transpose(1, 2)
        # 应用RoPE到Q/K
        q, k = self._apply_rope_qk(q, k)
        # 注意力分数 (b, h, n, n)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_h)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        # 聚合
        context = torch.matmul(attn_probs, v)  # (b, h, n, d_h)
        # 合并heads
        context = context.transpose(1, 2).contiguous().view(b, seq_len, h * d_h)
        out = self.o_proj(context)  # (b, seq_len, d_model)
        out = self.attn_dropout(out)
        # 残差（对序列自注意力）
        out = out + x
        return out.squeeze(0)

    def cross_attention(self, query_embd, context_embd):
        """
        交叉注意力：Q来自query_embd，K/V来自context_embd；对K应用RoPE。
        Args:
            query_embd: (num_query, d_model)
            context_embd: (seq_len, d_model)
        Returns:
            (num_query, d_model)
        """
        nq = query_embd.size(0)
        ns = context_embd.size(0)
        b = 1
        h = self.n_heads
        d_h = self.head_dim
        # 添加batch维度
        q = self.q_proj(query_embd.unsqueeze(0))  # (1, nq, d_model)
        k = self.k_proj(context_embd.unsqueeze(0))  # (1, ns, d_model)
        v = self.v_proj(context_embd.unsqueeze(0))  # (1, ns, d_model)
        # 变形为多头
        q = q.view(b, nq, h, d_h).transpose(1, 2)  # (1, h, nq, d_h)
        k = k.view(b, ns, h, d_h).transpose(1, 2)  # (1, h, ns, d_h)
        v = v.view(b, ns, h, d_h).transpose(1, 2)  # (1, h, ns, d_h)
        # 对K应用RoPE（保留Q不旋转）
        k = self._apply_rope_x(k)
        # 注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_h)  # (1,h,nq,ns)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        context = torch.matmul(attn_probs, v)  # (1,h,nq,d_h)
        # 合并heads
        context = context.transpose(1, 2).contiguous().view(b, nq, h * d_h)  # (1,nq,d_model)
        out = self.o_proj(context)  # (1,nq,d_model)
        out = self.attn_dropout(out)
        # 残差（对交叉注意力，以Q为残差）
        out = out + query_embd.unsqueeze(0)  # (1,nq,d_model)
        return out.squeeze(0)

    def deal_sequence(self, sequence):
        """
        处理序列embedding列表
        Args:
            sequence: embedding列表
        Returns:
            处理后的tensor
        """
        # 将embedding列表转换为tensor
        sequence_embd = torch.stack(sequence)
        
        # 直接在Q/K上通过RoPE的自注意力
        sequence_embd = self.self_attention(sequence_embd)
        
        return sequence_embd
