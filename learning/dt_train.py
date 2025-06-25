import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dt_dataset import BnBSequentialDataset, calculate_average_reward, calculate_average_reward_static, simple_collate_fn
from dt_model import DTModel
import time
import gc
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime

def train():
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型并移动到设备
    model = DTModel()
    model = model.to(device)
    
    # 创建数据集
    dataset = BnBSequentialDataset("/data/ltf/batch_transformer/learn2comparenodes/node_selection/data/GISP/train", max_samples=1000)
    
    valid_dataset = BnBSequentialDataset("/data/ltf/batch_transformer/learn2comparenodes/node_selection/data/GISP/valid", max_samples=1000)
    # 计算平均奖励
    # avg_reward = calculate_average_reward_static(dataset)
    # print(f"Average reward: {avg_reward}")
    
    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,collate_fn=simple_collate_fn)

    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True,collate_fn=simple_collate_fn)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    

    select_loss_weight = 0.1
    branch_loss_weight = 1
    
    # 创建检查点目录
    os.makedirs("./checkpoints", exist_ok=True)
    
    # 创建TensorBoard writer
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = f'./logs/train_{current_time}'
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard日志将保存到: {log_dir}")
    
    for epoch in range(100000):
        model.train()
        total_select_loss = 0
        total_branch_loss = 0
        total_step = 0
        total_select_acc = []
        total_branch_acc = []
        start_time = time.time()

        for batch in dataloader:
            # batch现在包含: sequence_data, type_ids, actions, candidates, branch_scores, attention_mask
            states, sequence_data = batch
            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            select_loss, branch_loss ,select_acc, branch_acc = model(states, sequence_data, device)

            # 计算总损失
            total_loss = select_loss * select_loss_weight + branch_loss * branch_loss_weight

            # 反向传播
            total_loss.backward()
            
            # 更新参数
            optimizer.step()

            total_select_loss += select_loss.item()
            total_branch_loss += branch_loss.item()
            total_select_acc.append(select_acc)
            total_branch_acc.append(branch_acc)
            total_step += 1
        # 计算平均指标
        avg_select_loss = total_select_loss / total_step
        avg_branch_loss = total_branch_loss / total_step
        avg_select_acc = np.mean(total_select_acc)
        avg_branch_acc = np.mean(total_branch_acc)
        end_time = time.time()
        duration = end_time - start_time

        # 打印训练信息
        print(f"Epoch {epoch}: Select Loss: {avg_select_loss:.4f}, Branch Loss: {avg_branch_loss:.4f}, Select Acc: {avg_select_acc:.4f}, Branch Acc: {avg_branch_acc:.4f}, Time: {duration:.2f}s", flush=True)

        # 记录训练指标到TensorBoard
        writer.add_scalar('Train_Loss/Select', avg_select_loss, epoch)
        writer.add_scalar('Train_Loss/Branch', avg_branch_loss, epoch)
        writer.add_scalar('Train_Loss/Total', avg_select_loss + avg_branch_loss, epoch)
        writer.add_scalar('Train_Accuracy/Select', avg_select_acc, epoch)
        writer.add_scalar('Train_Accuracy/Branch', avg_branch_acc, epoch)
        writer.add_scalar('Train_Count/Total_Steps', total_step, epoch)
        writer.add_scalar('Train_Time/Duration', duration, epoch)

        # eval
        if epoch % 10 ==0:
            model.eval()
            valid_select_loss = 0
            valid_branch_loss = 0
            valid_step = 0
            valid_select_acc = []
            valid_branch_acc = []
            for batch in valid_dataloader:
                # batch现在包含: sequence_data, type_ids, actions, candidates, branch_scores, attention_mask
                states, sequence_data = batch
                # 前向传播
                select_loss, branch_loss ,select_acc, branch_acc = model(states, sequence_data, device)
                valid_select_loss += select_loss.item()
                valid_branch_loss += branch_loss.item()
                valid_select_acc.append(select_acc)
                valid_branch_acc.append(branch_acc)
                valid_step += 1
            # 计算平均指标
            avg_valid_select_loss = valid_select_loss / valid_step
            avg_valid_branch_loss = valid_branch_loss / valid_step
            avg_valid_select_acc = np.mean(valid_select_acc)
            avg_valid_branch_acc = np.mean(valid_branch_acc)

            print(f"Epoch eval {epoch}: Select Loss: {avg_valid_select_loss:.4f}, Branch Loss: {avg_valid_branch_loss:.4f}, Select Acc: {avg_valid_select_acc:.4f}, Branch Acc: {avg_valid_branch_acc:.4f}")

            # 记录验证指标到TensorBoard
            writer.add_scalar('Valid_Loss/Select', avg_valid_select_loss, epoch)
            writer.add_scalar('Valid_Loss/Branch', avg_valid_branch_loss, epoch)
            writer.add_scalar('Valid_Loss/Total', avg_valid_select_loss + avg_valid_branch_loss, epoch)
            writer.add_scalar('Valid_Accuracy/Select', avg_valid_select_acc, epoch)
            writer.add_scalar('Valid_Accuracy/Branch', avg_valid_branch_acc, epoch)

    print("Training completed!")
    
    # 关闭TensorBoard writer
    writer.close()
    print(f"TensorBoard日志已保存到: {log_dir}")

if __name__ == "__main__":
    train() 