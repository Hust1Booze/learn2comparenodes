import csv
import logging
# make deterministic
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from collections import deque
import random
import torch
import pickle
import argparse

from dt_dataset import BnBDataset
from dt_model import DTModel
import tqdm

from torch.utils.data.dataloader import DataLoader
import torch_geometric
from dt_utiles import collate_fn

def train():
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = BnBDataset(data_dir='/data/ltf/foundation/learn2comparenodes/node_selection/data/GISP')
    model = DTModel().to(device)
    gnn_encoder = model.gnn_encoder

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True,
                            collate_fn=lambda b: collate_fn(b, gnn_encoder, device))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm.tqdm(train_loader):
            for k in batch:
                batch[k] = batch[k].to(device)

            preds, target = model(batch)
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), target.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {total_loss:.4f}")


if __name__ == "__main__":
    train()