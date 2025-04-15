import torch


def collate_fn(batch, gnn_encoder, device): 
    max_len = max(len(item[0]) for item in batch)
    batch_state_embeds = []
    batch_returns = []
    batch_actions = []

    for states, selnode_actions, branch_actions, returns, stepwise_returns, candidate_nodes, candidate_variables in batch:
        actions = selnode_actions  # or switch to branch_actions if needed

        # encode each TripartiteGraphData into vector
        state_embeds = []
        for g in states:
            # g = g.to(device)
            # emb = gnn_encoder(
            #     g.constraint_features, g.variable_features, g.leaf_features,
            #     g.edge_index_cv, g.edge_attr_cv, g.edge_index_vl, g.edge_attr_vl
            # )
            emb = gnn_encoder(
                g[0][0].to(device), g[0][1].to(device), g[0][2].to(device),
                g[0][3].to(device), g[0][4].to(device), g[0][5].to(device), g[0][6].to(device)
            )
            state_embeds.append(emb)

        pad_len = max_len - len(state_embeds)
        if pad_len > 0:
            pad_embed = torch.zeros_like(state_embeds[0])
            state_embeds.extend([pad_embed] * pad_len)
            returns.extend([0.0] * pad_len)
            actions.extend([-1] * pad_len)

        batch_state_embeds.append(torch.stack(state_embeds))  # [T, D]
        batch_returns.append(torch.tensor(returns))
        batch_actions.append(torch.tensor(actions))

    return {
        "states": torch.stack(batch_state_embeds),   # [B, T, D]
        "returns": torch.stack(batch_returns),       # [B, T]
        "actions": torch.stack(batch_actions),       # [B, T]
    }
