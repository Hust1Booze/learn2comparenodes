#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import deepspeed
import argparse
import json

# 简单的数据集示例
class SimpleDataset(Dataset):
    def __init__(self, size=1000, input_dim=128):
        self.size = size
        self.input_dim = input_dim
        # 生成一些随机数据用于分类任务
        self.data = torch.randn(size, input_dim)
        self.labels = torch.randint(0, 10, (size,))  # 10类分类
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 简单的模型示例
class SimpleModel(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, num_classes=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

def create_dataloaders(batch_size=32):
    """创建训练和验证数据加载器"""
    train_dataset = SimpleDataset(size=8000)
    val_dataset = SimpleDataset(size=2000)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Simple DeepSpeed Example')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    
    # DeepSpeed会自动添加其参数
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    # 创建模型
    model = SimpleModel()
    
    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(args.batch_size)
    
    # 初始化DeepSpeed
    model_engine, optimizer, train_loader, __ = deepspeed.initialize(
        args=args,
        model=model,
        training_data=train_loader.dataset,
        config_params=args.deepspeed_config
    )
    
    # 获取设备和rank信息
    device = model_engine.device
    local_rank = model_engine.local_rank
    
    print(f"Local rank: {local_rank}, Device: {device}")
    
    # 训练循环
    for epoch in range(args.epochs):
        model_engine.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 将数据移到正确的设备
            data = data.to(device)
            target = target.to(device)
            
            # 前向传播
            outputs = model_engine(data)
            loss = F.cross_entropy(outputs, target)
            
            # 反向传播
            model_engine.backward(loss)
            model_engine.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 打印训练进度
            if batch_idx % 100 == 0 and local_rank == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        
        # 验证
        if local_rank == 0:  # 只在主进程进行验证
            model_engine.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data = data.to(device)
                    target = target.to(device)
                    
                    outputs = model_engine(data)
                    val_loss += F.cross_entropy(outputs, target).item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Accuracy: {accuracy:.2f}%')
        
        # 保存检查点
        if epoch % 2 == 0:
            model_engine.save_checkpoint(f'./checkpoints', tag=f'epoch_{epoch}')
    
    print("Training completed!")

if __name__ == '__main__':
    main() 