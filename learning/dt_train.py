import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dt_dataset import BnBSequentialDataset,bnb_collate
from dt_model import DTModel

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DTModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dataset = BnBSequentialDataset("/data/ltf/dt_foundation/dt_bnb/node_selection/data/GISP", model, device)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=bnb_collate)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch_tokens, type_ids, attention_mask in dataloader:
            type_ids = type_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_tokens = [b.to(device) for b in batch_tokens]

            out = model(batch_tokens, type_ids, attention_mask)
            target = torch.zeros_like(out)  # placeholder loss target

            loss = F.mse_loss(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss:.4f}")

if __name__ == "__main__":
    train()