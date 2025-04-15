import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv

class TripartiteGNNEncoder(nn.Module):
    def __init__(self, emb_dim=32):
        super().__init__()
        self.emb_dim = emb_dim

        # Feature dimensions
        self.constraint_dim = 4
        self.variable_dim = 3
        self.leaf_dim = 3
        self.edge_cv_dim = 1
        self.edge_vl_dim = 2

        # Embedding layers
        self.constraint_embedding = nn.Sequential(
            nn.LayerNorm(self.constraint_dim),
            nn.Linear(self.constraint_dim, emb_dim),
            nn.ReLU(),
        )

        self.variable_embedding = nn.Sequential(
            nn.LayerNorm(self.variable_dim),
            nn.Linear(self.variable_dim, emb_dim),
            nn.ReLU(),
        )

        self.leaf_embedding = nn.Sequential(
            nn.LayerNorm(self.leaf_dim),
            nn.Linear(self.leaf_dim, emb_dim),
            nn.ReLU(),
        )

        self.edge_cv_embedding = nn.LayerNorm(self.edge_cv_dim)
        #self.edge_vl_embedding = nn.LayerNorm(self.edge_vl_dim)
        self.edge_vl_embedding = torch.nn.Sequential(
            nn.Linear(2, 16),  # 输入是 (bound, btype)
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Message passing
        self.gnn_cv = GraphConv((emb_dim, emb_dim), emb_dim)
        self.gnn_vl = GraphConv((emb_dim, emb_dim), emb_dim)

    def forward(self, constraint_features, variable_features, leaf_features,
                edge_index_cv, edge_attr_cv, edge_index_vl, edge_attr_vl):


        h_c = self.constraint_embedding(constraint_features)
        h_v = self.variable_embedding(variable_features)
        h_l = self.leaf_embedding(leaf_features)

        edge_attr_cv = self.edge_cv_embedding(edge_attr_cv.unsqueeze(1))
        edge_attr_vl = self.edge_vl_embedding(edge_attr_vl)

        # Constraint -> Variable -> Constraint message passing
        h_c = F.relu(self.gnn_cv((h_v, h_c), edge_index_cv, edge_weight=edge_attr_cv))
        h_v = F.relu(self.gnn_cv((h_c, h_v), edge_index_cv.flip(0), edge_weight=edge_attr_cv))

        # Variable -> Leaf message passing
        h_l = F.relu(self.gnn_vl((h_v, h_l), edge_index_vl, edge_weight=edge_attr_vl))

        # Global pooling (average)
        graph_embedding = torch.cat([
            h_c.mean(dim=0),
            h_v.mean(dim=0),
            h_l.mean(dim=0)
        ], dim=-1)

        return graph_embedding  # Shape: [emb_dim * 3]


class DTModel(nn.Module):
    def __init__(self, gnn_embed_dim=32, dt_hidden_dim=128, seq_len=10):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = gnn_embed_dim * 3

        self.gnn_encoder = TripartiteGNNEncoder(emb_dim=gnn_embed_dim)

        self.token_embedding = nn.Linear(self.embed_dim, dt_hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(seq_len, dt_hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=dt_hidden_dim, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.output_head = nn.Sequential(
            nn.Linear(dt_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出每个候选节点/变量的 score
        )

    def forward(self, batch):
        """
        trajectory_batch: List of trajectory steps, each with graph data
        (e.g., list of [constraints, variables, leafs, edge_index_cv, edge_attr_cv, edge_index_vl, edge_attr_vl])
        """

        states = batch[0]
        # selnode_actions,
        # branch_actions
        # returns, 
        # stepwise_returns,
        # candidate_nodes,
        # candidate_variables
        embeddings = []
        for step in states:
            embedding = self.gnn_encoder(*step)  # shape: [embed_dim]
            embeddings.append(embedding)

        x = torch.stack(embeddings, dim=0)  # [seq_len, embed_dim]
        x = self.token_embedding(x) + self.pos_embedding[:len(states)]
        x = self.transformer(x.unsqueeze(1)).squeeze(1)  # [seq_len, dt_hidden_dim]
        x = self.output_head(x)  # [seq_len, 1]

        return x.squeeze(-1)  # 每个 step 的 score
