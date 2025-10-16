import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import BnBSequentialDataset, collate_fn
from model import DTModel
import time
import gc
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
import yaml


def train():

    with open('./learning/train.yaml', 'r') as f:
        config = yaml.safe_load(f)
    # print info
    print('\n\n\n')
    print(f'~'*80)
    print(f'Config:\n{config}')
    print(f'~'*80)


    batch_size = config['batch_size']
    problem = config['problem']
    max_samples = config['max_samples']
    select_loss_weight = config['select_loss_weight']
    branch_loss_weight = config['branch_loss_weight']
    lr = config['lr']

    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型并移动到设备
    model = DTModel()
    model = model.to(device)
    
    # 创建数据集
    dataset = BnBSequentialDataset(f"/lab/shiyh_lab/12332470/code/bnb_gasses/learn2comparenodes/node_selection/data/{problem}/train", max_samples=max_samples)
    valid_dataset = BnBSequentialDataset(f"/lab/shiyh_lab/12332470/code/bnb_gasses/learn2comparenodes/node_selection/data/{problem}/valid", max_samples=max_samples)
    
    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 跟踪最佳验证branch_top1准确率
    best_branch_top1 = 0.0
    
    # 创建TensorBoard writer
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = f'./logs/train_{current_time}'
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("./checkpoints", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    writer = SummaryWriter(log_dir)

    print(f"TensorBoard日志将保存到: {log_dir}")
    
    for epoch in range(100000):
        model.train()
        total_select_loss = 0
        total_branch_loss = 0
        total_step = 0
        total_select_top1 = []
        total_select_top5 = []
        total_select_top10 = []
        total_branch_top1 = []
        total_branch_top5 = []
        total_branch_top10 = []
        start_time = time.time()

        for batch in dataloader:

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            branch_loss, select_loss, branch_top1, branch_top5, branch_top10, select_top1, select_top5, select_top10 = model(batch, device)

            # 计算总损失
            total_loss = select_loss * select_loss_weight + branch_loss * branch_loss_weight

            # 反向传播
            total_loss.backward()
            
            # 更新参数
            optimizer.step()

            total_select_loss += select_loss.item()
            total_branch_loss += branch_loss.item()
            total_select_top1.append(select_top1)
            total_select_top5.append(select_top5)
            total_select_top10.append(select_top10)
            total_branch_top1.append(branch_top1)
            total_branch_top5.append(branch_top5)
            total_branch_top10.append(branch_top10)
            total_step += 1
        # 计算平均指标
        avg_select_loss = total_select_loss / total_step
        avg_branch_loss = total_branch_loss / total_step
        avg_select_top1 = np.mean(total_select_top1)
        avg_select_top5 = np.mean(total_select_top5)
        avg_select_top10 = np.mean(total_select_top10)
        avg_branch_top1 = np.mean(total_branch_top1)
        avg_branch_top5 = np.mean(total_branch_top5)
        avg_branch_top10 = np.mean(total_branch_top10)
        end_time = time.time()
        duration = end_time - start_time

        # 打印训练信息
        print(f"Epoch {epoch}: Select Loss: {avg_select_loss:.4f}, Branch Loss: {avg_branch_loss:.4f}, Select Top1: {avg_select_top1:.4f}, Select Top5: {avg_select_top5:.4f}, Select Top10: {avg_select_top10:.4f}, Branch Top1: {avg_branch_top1:.4f}, Branch Top5: {avg_branch_top5:.4f}, Branch Top10: {avg_branch_top10:.4f}, Time: {duration:.2f}s", flush=True)

        # 记录训练指标到TensorBoard
        writer.add_scalar('Train_Loss/Select', avg_select_loss, epoch)
        writer.add_scalar('Train_Loss/Branch', avg_branch_loss, epoch)
        writer.add_scalar('Train_Loss/Total', avg_select_loss + avg_branch_loss, epoch)
        writer.add_scalar('Train_Accuracy/Select_Top1', avg_select_top1, epoch)
        writer.add_scalar('Train_Accuracy/Select_Top5', avg_select_top5, epoch)
        writer.add_scalar('Train_Accuracy/Select_Top10', avg_select_top10, epoch)
        writer.add_scalar('Train_Accuracy/Branch_Top1', avg_branch_top1, epoch)
        writer.add_scalar('Train_Accuracy/Branch_Top5', avg_branch_top5, epoch)
        writer.add_scalar('Train_Accuracy/Branch_Top10', avg_branch_top10, epoch)
        writer.add_scalar('Train_Count/Total_Steps', total_step, epoch)
        writer.add_scalar('Train_Time/Duration', duration, epoch)

        # eval
        if epoch % 100 ==0:
            model.eval()
            valid_select_loss = 0
            valid_branch_loss = 0
            valid_step = 0
            valid_select_top1 = []
            valid_select_top5 = []
            valid_select_top10 = []
            valid_branch_top1 = []
            valid_branch_top5 = []
            valid_branch_top10 = []
            for i in range(10):
                for batch in valid_dataloader:
                    # batch现在包含: sequence_data, type_ids, actions, candidates, branch_scores, attention_mask
                    # 前向传播
                    branch_loss, select_loss, branch_top1, branch_top5, branch_top10, select_top1, select_top5, select_top10 = model(batch, device)
                    valid_branch_loss += branch_loss.item()
                    valid_select_top1.append(select_top1)
                    valid_select_top5.append(select_top5)
                    valid_select_top10.append(select_top10)
                    valid_branch_top1.append(branch_top1)
                    valid_branch_top5.append(branch_top5)
                    valid_branch_top10.append(branch_top10)
                    valid_step += 1
            # 计算平均指标
            avg_valid_select_loss = valid_select_loss / valid_step
            avg_valid_branch_loss = valid_branch_loss / valid_step
            avg_valid_select_top1 = np.mean(valid_select_top1)
            avg_valid_select_top5 = np.mean(valid_select_top5)
            avg_valid_select_top10 = np.mean(valid_select_top10)
            avg_valid_branch_top1 = np.mean(valid_branch_top1)
            avg_valid_branch_top5 = np.mean(valid_branch_top5)
            avg_valid_branch_top10 = np.mean(valid_branch_top10)

            print(f"Valid {epoch}: Select Loss: {avg_valid_select_loss:.4f}, Branch Loss: {avg_valid_branch_loss:.4f}, Select Top1: {avg_valid_select_top1:.4f}, Select Top5: {avg_valid_select_top5:.4f}, Select Top10: {avg_valid_select_top10:.4f}, Branch Top1: {avg_valid_branch_top1:.4f}, Branch Top5: {avg_valid_branch_top5:.4f}, Branch Top10: {avg_valid_branch_top10:.4f}")

            # 记录验证指标到TensorBoard
            writer.add_scalar('Valid_Loss/Select', avg_valid_select_loss, epoch)
            writer.add_scalar('Valid_Loss/Branch', avg_valid_branch_loss, epoch)
            writer.add_scalar('Valid_Loss/Total', avg_valid_select_loss + avg_valid_branch_loss, epoch)
            writer.add_scalar('Valid_Accuracy/Select_Top1', avg_valid_select_top1, epoch)
            writer.add_scalar('Valid_Accuracy/Select_Top5', avg_valid_select_top5, epoch)
            writer.add_scalar('Valid_Accuracy/Select_Top10', avg_valid_select_top10, epoch)
            writer.add_scalar('Valid_Accuracy/Branch_Top1', avg_valid_branch_top1, epoch)
            writer.add_scalar('Valid_Accuracy/Branch_Top5', avg_valid_branch_top5, epoch)
            writer.add_scalar('Valid_Accuracy/Branch_Top10', avg_valid_branch_top10, epoch)

            # 检查是否达到新的最佳验证branch_top1准确率
            if avg_valid_branch_top1 > best_branch_top1:
                best_branch_top1 = avg_valid_branch_top1
                # 保存模型
                model_path = f'./models/best_model_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pth'
                torch.save(model.state_dict(), model_path)
                print(f"新的最佳验证branch_top1准确率: {best_branch_top1:.4f}，模型已保存到: {model_path}")

    print("Training completed!")
    
    # 关闭TensorBoard writer
    writer.close()
    print(f"TensorBoard日志已保存到: {log_dir}")

if __name__ == "__main__":
    train() 