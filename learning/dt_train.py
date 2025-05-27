import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dt_dataset import BnBSequentialDataset,bnb_collate,calculate_average_reward
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
    avg_reward = calculate_average_reward(dataset)
    print(f"Average reward: {avg_reward}")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=bnb_collate)

    # 添加垃圾回收
    import gc

    for epoch in range(100000):
        model.train()
        total_loss = 0

        start_time = time.time()  # ⏱️ 开始计时

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
            
            del loss
            torch.cuda.empty_cache()

        end_time = time.time()  # ⏱️ 结束计时
        duration = end_time - start_time

        print(f"Epoch {epoch}: Loss = {total_loss:.4f} | Time: {duration:.2f} seconds")
        # 每个 epoch 结束后进行垃圾回收
        gc.collect()
        torch.cuda.empty_cache()

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"models/dt_model_{epoch}.pth")


if __name__ == "__main__":
    train()