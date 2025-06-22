import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dt_dataset import BnBSequentialDataset, calculate_average_reward, calculate_average_reward_static, simple_collate_fn
from dt_model import DTModel
import time
import gc
import os

def train():
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型并移动到设备
    model = DTModel()
    model = model.to(device)
    
    # 创建数据集
    dataset = BnBSequentialDataset("/data/ltf/dt_transformer/learn2comparenodes/node_selection/data/GISP", max_samples=1000)
    
    # 计算平均奖励
    avg_reward = calculate_average_reward_static(dataset)
    print(f"Average reward: {avg_reward}")
    
    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True,collate_fn=simple_collate_fn)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    

    select_loss_weight = 0.1
    branch_loss_weight = 1
    
    # 创建检查点目录
    os.makedirs("./checkpoints", exist_ok=True)
    
    for epoch in range(100000):
        model.train()
        total_select_loss = 0
        total_branch_loss = 0
        total_select_steps = 0
        total_branch_steps = 0
        total_branch_corrects = 0
        total_select_corrects = 0

        start_time = time.time()

        for batch in dataloader:
            # batch现在包含: sequence_data, type_ids, actions, candidates, branch_scores, attention_mask
            sequence_data = batch
            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            select_loss, branch_loss, select_steps, branch_steps, select_corrects, branch_corrects  = model(sequence_data, device)

            # 计算总损失
            total_loss = select_loss * select_loss_weight / select_steps + branch_loss * branch_loss_weight / branch_steps

            # 反向传播
            total_loss.backward()
            
            # 更新参数
            optimizer.step()

            total_select_loss += select_loss.item()
            total_branch_loss += branch_loss.item()
            total_select_steps += select_steps
            total_branch_steps += branch_steps
            total_branch_corrects += branch_corrects
            total_select_corrects += select_corrects

        # 计算平均指标
        avg_select_loss = total_select_loss / total_select_steps
        avg_branch_loss = total_branch_loss / total_branch_steps
        avg_select_acc = total_select_corrects / total_select_steps
        avg_branch_acc = total_branch_corrects / total_branch_steps

        end_time = time.time()
        duration = end_time - start_time

        # 打印训练信息
        print(f"Epoch {epoch}:")
        print(f"  Select Loss: {avg_select_loss:.4f}, Branch Loss: {avg_branch_loss:.4f}")
        print(f"  Select Acc: {avg_select_acc:.4f}, Branch Acc: {avg_branch_acc:.4f}")
        print(f"  Time: {duration:.2f} seconds", flush=True)


    print("Training completed!")

if __name__ == "__main__":
    train() 