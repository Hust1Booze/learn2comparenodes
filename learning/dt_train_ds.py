import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dt_dataset import BnBSequentialDataset, calculate_average_reward_static, simple_collate_fn
from dt_model import DTModel
import time
import deepspeed
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
# CUDA_VISIBLE_DEVICES = int(os.environ[“LOCAL_RANK”])
                          
def train():

    batch_size = 8
    
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
            "stage": 2,
            # "offload_optimizer": {
            #     "device": "cpu"
            # }
        }
    }

    model = DTModel()
    print(model)

    model.print_model_info()
    
    # 先创建数据集（在DeepSpeed初始化之前）
    dataset = BnBSequentialDataset("/lab/shiyh_lab/12332470/code/transformer_foundation/learn2comparenodes/node_selection/data/GISP", max_samples=1000)
    valid_dataset = BnBSequentialDataset("/lab/shiyh_lab/12332470/code/transformer_foundation/learn2comparenodes/node_selection/data/GISP", max_samples=1000)
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
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=simple_collate_fn)

    select_loss_weight = 0.05
    branch_loss_weight = 1
    
    for epoch in range(100000):
        model_engine.train()
        total_select_loss = 0
        total_branch_loss = 0
        total_step = 0
        total_select_acc = []
        total_branch_acc = []

        start_time = time.time()

        for batch in dataloader:
            states, sequence_data = batch
            # 前向传播
            select_loss, branch_loss ,select_acc, branch_acc = model_engine(states, sequence_data, model_engine.device)

            total_loss = select_loss*select_loss_weight + branch_loss*branch_loss_weight

            model_engine.backward(total_loss)
            model_engine.step()

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
                writer.add_scalar('Epoch_Count/Total_Steps', total_step, epoch)
                writer.add_scalar('Epoch_Time/Duration', duration, epoch)
                

        if epoch % 10 ==0 :
            model_engine.eval()
            valid_select_loss = 0
            valid_branch_loss = 0
            valid_step = 0
            valid_select_acc = []
            valid_branch_acc = []
            for batch in valid_dataloader:
                states, sequence_data = batch
                # 前向传播
                select_loss, branch_loss ,select_acc, branch_acc = model_engine(states, sequence_data, model_engine.device)

                valid_select_loss += select_loss.item()
                valid_branch_loss += branch_loss.item()
                valid_select_acc.append(select_acc)
                valid_branch_acc.append(branch_acc)
                valid_step += 1
            avg_valid_select_loss = valid_select_loss / valid_step
            avg_valid_branch_loss= valid_branch_loss / valid_step
            avg_valid_select_acc = np.mean(valid_select_acc)
            avg_valid_branch_acc = np.mean(valid_branch_acc)
            # 只在主进程打印
            if model_engine.global_rank == 0:
                print(f"Epoch valid {epoch}:")
                print(f"  Select Loss: {avg_valid_select_loss:.4f}, Branch Loss: {avg_valid_branch_loss:.4f}")
                print(f"  Select Acc: {avg_valid_select_acc:.4f}, Branch Acc: {avg_valid_branch_acc:.4f}")
                
                # 记录epoch级别的指标到TensorBoard（只在主进程）
                if writer is not None:
                    writer.add_scalar('Valid_Loss/Select', avg_valid_select_loss, epoch)
                    writer.add_scalar('Valid_Loss/Branch', avg_valid_branch_loss, epoch)
                    writer.add_scalar('Valid_Loss/Total', avg_valid_select_loss + avg_valid_branch_loss, epoch)
                    writer.add_scalar('Valid_Accuracy/Select', avg_valid_select_acc, epoch)
                    writer.add_scalar('Valid_Accuracy/Branch', avg_valid_branch_acc, epoch)



    # 关闭TensorBoard writer（只在主进程）
    if model_engine.global_rank == 0 and writer is not None:
        writer.close()
        print(f"TensorBoard日志已保存到: {log_dir}")

if __name__ == "__main__":
    train()