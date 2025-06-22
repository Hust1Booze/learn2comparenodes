import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dt_dataset import BnBSequentialDataset, calculate_average_reward_static, simple_collate_fn
from dt_model import DTModel
import time
import gc
import deepspeed
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import datetime

# CUDA_VISIBLE_DEVICES = int(os.environ[“LOCAL_RANK”])
                          
def train():

    batch_size = 2
    
    # 解析命令行参数（DeepSpeed需要）
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-4
            }
        },
        "fp16": {
            "enabled": False
        },
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {
                "device": "cpu"
            }
        }
    }

    model = DTModel()

    # 先创建数据集（在DeepSpeed初始化之前）
    dataset = BnBSequentialDataset("/lab/shiyh_lab/12332470/code/foundation/learn2comparenodes/node_selection/data/GISP", max_samples=1000)
    
    # DeepSpeed 初始化
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    # 只在主进程创建TensorBoard writer
    writer = None
    if model_engine.global_rank == 0:
        # 创建TensorBoard writer
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = f'./logs/train_{current_time}'
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard日志将保存到: {log_dir}")
    
    # 计算平均奖励（使用静态方法，不需要模型前向传播）
    if model_engine.global_rank == 0:  # 只在主进程打印
        avg_reward = calculate_average_reward_static(dataset)
        print(f"Average reward: {avg_reward}")
    
    # DataLoader的batch_size应该等于DeepSpeed配置中的train_micro_batch_size_per_gpu
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=simple_collate_fn)
   

    select_loss_weight = 0.05
    branch_loss_weight = 1
    
    for epoch in range(100000):
        model_engine.train()
        total_select_loss = 0
        total_branch_loss = 0
        total_select_steps = 0
        total_branch_steps = 0
        total_branch_corrects = 0
        total_select_corrects = 0

        start_time = time.time()

        for step, batch in enumerate(dataloader):
            sequence_data = batch
        
            select_loss, branch_loss, select_steps, branch_steps, select_corrects, branch_corrects  = model_engine(sequence_data, model_engine.device)

            total_loss = select_loss*select_loss_weight + branch_loss*branch_loss_weight

            model_engine.backward(total_loss)
            model_engine.step()

            total_select_loss += select_loss.item()
            total_branch_loss += branch_loss.item()
            total_select_steps += select_steps
            total_branch_steps += branch_steps
            total_select_corrects += select_corrects
            total_branch_corrects  += branch_corrects


        # 计算平均指标
        avg_select_loss = total_select_loss / total_select_steps
        avg_branch_loss = total_branch_loss / total_branch_steps
        avg_select_acc = total_select_corrects / total_select_steps
        avg_branch_acc = total_branch_corrects / total_branch_steps

        end_time = time.time()
        duration = end_time - start_time

        # 只在主进程打印
        if model_engine.global_rank == 0:
            print(f"Epoch {epoch}:")
            print(f"  Select Loss: {avg_select_loss:.4f}, Branch Loss: {avg_branch_loss:.4f}")
            print(f"  Select Acc: {avg_select_acc:.4f}, Branch Acc: {avg_branch_acc:.4f}")
            print(f"  Time: {duration:.2f} seconds", flush= True)
            
            # 记录epoch级别的指标到TensorBoard（只在主进程）
            if writer is not None:
                writer.add_scalar('Epoch_Loss/Select', avg_select_loss, epoch)
                writer.add_scalar('Epoch_Loss/Branch', avg_branch_loss, epoch)
                writer.add_scalar('Epoch_Loss/Total', avg_select_loss + avg_branch_loss, epoch)
                writer.add_scalar('Epoch_Accuracy/Select', avg_select_acc, epoch)
                writer.add_scalar('Epoch_Accuracy/Branch', avg_branch_acc, epoch)
                writer.add_scalar('Epoch_Count/Total_Select_Steps', total_select_steps, epoch)
                writer.add_scalar('Epoch_Count/Total_Branch_Steps', total_branch_steps, epoch)
                writer.add_scalar('Epoch_Time/Duration', duration, epoch)
                

    
    # 关闭TensorBoard writer（只在主进程）
    if model_engine.global_rank == 0 and writer is not None:
        writer.close()
        print(f"TensorBoard日志已保存到: {log_dir}")

if __name__ == "__main__":
    train()