import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dt_dataset import BnBSequentialDataset,bnb_collate
from dt_model import DTModel
import time

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    # if torch.cuda.is_available():
    #     device = torch.cuda.current_device()
    model = DTModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dataset = BnBSequentialDataset("/data/ltf/dt_foundation/dt_bnb/node_selection/data/GISP", model, device)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=bnb_collate)

    start_time = time.time()  # ⏱️ 开始计时

    for epoch in range(100000):
        model.train()
        total_loss = 0

        for batch_tokens, type_ids, actions, candidates, attention_mask in dataloader:
            batch_tokens = batch_tokens.to(device)
            type_ids = type_ids.to(device)
            actions = actions.to(device)
            attention_mask = attention_mask.to(device)

            loss = model(batch_tokens, type_ids, attention_mask, actions, candidates)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        end_time = time.time()  # ⏱️ 结束计时
        duration = end_time - start_time

        print(f"Epoch {epoch}: Loss = {total_loss:.4f} | Time: {duration:.2f} seconds")


if __name__ == "__main__":
    train()