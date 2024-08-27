#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 10:38:45 2021

@author: abdel
"""


import os
import sys
import torch
import torch_geometric
from pathlib import Path
from model import GNNPolicy
from data_type import GraphDataset
from utils import process
import datetime

if __name__ == "__main__":
    debug_model = True
    with_root_info = True
    problem = "WPMS"#"GISP" "WPMS"
    lr = 0.005
    n_epoch = 2
    n_sample = -1
    patience = 10
    early_stopping = 20
    normalize = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_train = 512
    batch_valid  = 512
    
    loss_fn = torch.nn.BCELoss()

    optimizer_fn = torch.optim.Adam
    
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-problem':
            problem = str(sys.argv[i + 1])
        if sys.argv[i] == '-lr':
            lr = float(sys.argv[i + 1])
        if sys.argv[i] == '-n_epoch':
            n_epoch = int(sys.argv[i + 1])
        if sys.argv[i] == '-n_sample':
            n_sample = int(sys.argv[i + 1])
        if sys.argv[i] == '-patience':
            patience = int(sys.argv[i + 1])
        if sys.argv[i] == '-early_stopping':
            early_stopping = int(sys.argv[i + 1])
        if sys.argv[i] == '-normalize':
            normalize = bool(int(sys.argv[i + 1]))
        if sys.argv[i] == '-device':
            device = str(sys.argv[i + 1])
        if sys.argv[i] == '-batch_train':
            batch_train = int(sys.argv[i + 1])
        if sys.argv[i] == '-batch_valid':
            batch_valid = int(sys.argv[i + 1])
            
  
    
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []


    train_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                            f"../node_selection/data/{problem}/train")).glob("*.pt") ][:n_sample]
    
    valid_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                            f"../node_selection/data/{problem}/test")).glob("*.pt") ][:int(0.2*n_sample if n_sample != -1 else -1)]
    if with_root_info:
        train_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                                f"../node_selection/data/{problem}/train_cl")).glob("*.pt") ][:n_sample]
        
        valid_files = [ str(path) for path in Path(os.path.join(os.path.dirname(__file__), 
                                                                f"../node_selection/data/{problem}/test_cl")).glob("*.pt") ][:int(0.2*n_sample if n_sample != -1 else -1)]
    if problem == 'FCMCNF':
        train_files = train_files + valid_files[3000:]
        valid_files = valid_files[:3000]

    train_data = GraphDataset(train_files)
    valid_data = GraphDataset(valid_files)
    
    
# TO DO : learn something from the data
    if not with_root_info:
        train_loader = torch_geometric.loader.DataLoader(train_data, 
                                                        batch_size=batch_train, 
                                                        shuffle=True, 
                                                        follow_batch=['constraint_features_s', 
                                                                    'constraint_features_t',
                                                                    'variable_features_s',
                                                                    'variable_features_t'])
        
        valid_loader = torch_geometric.loader.DataLoader(valid_data, 
                                                        batch_size=batch_valid, 
                                                        shuffle=False, 
                                                        follow_batch=['constraint_features_s',
                                                                    'constraint_features_t',
                                                                    'variable_features_s',
                                                                    'variable_features_t'])
    else:
        train_loader = torch_geometric.loader.DataLoader(train_data, 
                                                    batch_size=batch_train, 
                                                    shuffle=True, 
                                                    follow_batch=['constraint_features_s', 
                                                                'constraint_features_t',
                                                                'constraint_features_root',
                                                                'variable_features_s',
                                                                'variable_features_t',
                                                                'variable_features_root'])
    
        valid_loader = torch_geometric.loader.DataLoader(valid_data, 
                                                        batch_size=batch_valid, 
                                                        shuffle=False, 
                                                        follow_batch=['constraint_features_s',
                                                                    'constraint_features_t',
                                                                    'constraint_features_root',
                                                                    'variable_features_s',
                                                                    'variable_features_t',
                                                                    'variable_features_root'])
    
    policy = GNNPolicy().to(device)
    optimizer = optimizer_fn(policy.parameters(), lr=lr) #ADAM is the best
    
    print("-------------------------")
    print(f"GNN for problem {problem}")
    print(f"Training on:          {len(train_data)} samples")
    print(f"Validating on:        {len(valid_data)} samples")
    print(f"Batch Size Train:     {batch_train}")
    print(f"Batch Size Valid      {batch_valid}")
    print(f"Learning rate:        {lr} ")
    print(f"Number of epochs:     {n_epoch}")
    print(f"Normalize:            {normalize}")
    print(f"Device:               {device}")
    print(f"Loss fct:             {loss_fn}")
    print(f"Optimizer:            {optimizer_fn}")  
    print(f"Model's Size:         {sum(p.numel() for p in policy.parameters())} parameters ")
    print("-------------------------") 
    
    best_valid_acc = -1
    for epoch in range(n_epoch):
        print(f"Epoch {epoch + 1}")
        
        train_loss, train_acc = process(policy, 
                                        train_loader, 
                                        loss_fn,
                                        device,
                                        optimizer=optimizer, 
                                        normalize=normalize,
                                        with_root_info = with_root_info)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        print(f"Train loss: {train_loss:0.3f}, accuracy {train_acc:0.3f}" )
    
        valid_loss, valid_acc = process(policy, 
                                        valid_loader, 
                                        loss_fn, 
                                        device,
                                        optimizer=None,
                                        normalize=normalize,
                                        with_root_info = with_root_info)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        
        # if valid_acc > best_valid_acc:
        #     best_valid_acc = valid_acc
        #     torch.save(policy.state_dict(),f'policy_{problem}.pkl')
        #     print('save model')
        print(f"Valid loss: {valid_loss:0.3f}, accuracy {valid_acc:0.3f}" )

    torch.save(policy.state_dict(),f'policy_{problem}.pkl')
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(policy.state_dict(),f'pkls/policy_{problem}_{current_time}.pkl')
    print(f'Model saved at pkls/policy_{problem}_{current_time}.pkl')





