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

    batch_size = 32
    
    # 解析命令行参数（DeepSpeed需要）
    ds_config = {
        "train_micro_batch_size_per_gpu": batch_size,
         #"gradient_accumulation_steps": 4,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 5e-4
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


    #model.print_model_info()
    
    # 先创建数据集（在DeepSpeed初始化之前）
    dataset = BnBSequentialDataset("/lab/shiyh_lab/12332470/code/batch_transformer/learn2comparenodes/node_selection/data/GISP/train/", max_samples=500)
    valid_dataset = BnBSequentialDataset("/lab/shiyh_lab/12332470/code/batch_transformer/learn2comparenodes/node_selection/data/GISP/valid", max_samples=100)
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
    # if model_engine.global_rank == 0:  # 只在主进程打印
    #     avg_reward = calculate_average_reward_static(dataset)
    #     print(f"Average reward: {avg_reward}")
    
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
        total_select_top1 = []
        total_select_top5 = []
        total_select_top10 = []
        total_branch_top1 = []
        total_branch_top5 = []
        total_branch_top10 = []

        start_time = time.time()

        for batch in dataloader:
            states, sequence_data = batch
            # 前向传播
            branch_loss, select_loss, branch_top1, branch_top5, branch_top10, select_top1, select_top5, select_top10 = model_engine(states, sequence_data, model_engine.device)

            total_loss = select_loss*select_loss_weight + branch_loss*branch_loss_weight

            model_engine.backward(total_loss)
            model_engine.step()

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

        # 只在主进程打印
        if model_engine.global_rank == 0:
            print(f"Epoch {epoch}: Select Loss: {avg_select_loss:.4f}, Branch Loss: {avg_branch_loss:.4f}, Select Top1: {avg_select_top1:.4f}, Select Top5: {avg_select_top5:.4f}, Select Top10: {avg_select_top10:.4f}, Branch Top1: {avg_branch_top1:.4f}, Branch Top5: {avg_branch_top5:.4f}, Branch Top10: {avg_branch_top10:.4f}, Time: {duration:.2f}s", flush=True)
            
            # 记录epoch级别的指标到TensorBoard（只在主进程）
            if writer is not None:
                writer.add_scalar('Epoch_Loss/Select', avg_select_loss, epoch)
                writer.add_scalar('Epoch_Loss/Branch', avg_branch_loss, epoch)
                writer.add_scalar('Epoch_Loss/Total', avg_select_loss + avg_branch_loss, epoch)
                writer.add_scalar('Epoch_Accuracy/Select_Top1', avg_select_top1, epoch)
                writer.add_scalar('Epoch_Accuracy/Select_Top5', avg_select_top5, epoch)
                writer.add_scalar('Epoch_Accuracy/Select_Top10', avg_select_top10, epoch)
                writer.add_scalar('Epoch_Accuracy/Branch_Top1', avg_branch_top1, epoch)
                writer.add_scalar('Epoch_Accuracy/Branch_Top5', avg_branch_top5, epoch)
                writer.add_scalar('Epoch_Accuracy/Branch_Top10', avg_branch_top10, epoch)
                writer.add_scalar('Epoch_Count/Total_Steps', total_step, epoch)
                writer.add_scalar('Epoch_Time/Duration', duration, epoch)
                

        if epoch % 10 ==0 :
            model_engine.eval()
            valid_select_loss = 0
            valid_branch_loss = 0
            valid_step = 0
            valid_select_top1 = []
            valid_select_top5 = []
            valid_select_top10 = []
            valid_branch_top1 = []
            valid_branch_top5 = []
            valid_branch_top10 = []
            for batch in valid_dataloader:
                states, sequence_data = batch
                # 前向传播
                branch_loss, select_loss, branch_top1, branch_top5, branch_top10, select_top1, select_top5, select_top10 = model_engine(states, sequence_data, model_engine.device)

                valid_select_loss += select_loss.item()
                valid_branch_loss += branch_loss.item()
                valid_select_top1.append(select_top1)
                valid_select_top5.append(select_top5)
                valid_select_top10.append(select_top10)
                valid_branch_top1.append(branch_top1)
                valid_branch_top5.append(branch_top5)
                valid_branch_top10.append(branch_top10)
                valid_step += 1
            avg_valid_select_loss = valid_select_loss / valid_step
            avg_valid_branch_loss= valid_branch_loss / valid_step
            avg_valid_select_top1 = np.mean(valid_select_top1)
            avg_valid_select_top5 = np.mean(valid_select_top5)
            avg_valid_select_top10 = np.mean(valid_select_top10)
            avg_valid_branch_top1 = np.mean(valid_branch_top1)
            avg_valid_branch_top5 = np.mean(valid_branch_top5)
            avg_valid_branch_top10 = np.mean(valid_branch_top10)
            # 只在主进程打印
            if model_engine.global_rank == 0:
                print(f"Valid {epoch}: Select Loss: {avg_valid_select_loss:.4f}, Branch Loss: {avg_valid_branch_loss:.4f}, Select Top1: {avg_valid_select_top1:.4f}, Select Top5: {avg_valid_select_top5:.4f}, Select Top10: {avg_valid_select_top10:.4f}, Branch Top1: {avg_valid_branch_top1:.4f}, Branch Top5: {avg_valid_branch_top5:.4f}, Branch Top10: {avg_valid_branch_top10:.4f}")
                
                # 记录epoch级别的指标到TensorBoard（只在主进程）
                if writer is not None:
                    writer.add_scalar('Valid_Loss/Select', avg_valid_select_loss, epoch)
                    writer.add_scalar('Valid_Loss/Branch', avg_valid_branch_loss, epoch)
                    writer.add_scalar('Valid_Loss/Total', avg_valid_select_loss + avg_valid_branch_loss, epoch)
                    writer.add_scalar('Valid_Accuracy/Select_Top1', avg_valid_select_top1, epoch)
                    writer.add_scalar('Valid_Accuracy/Select_Top5', avg_valid_select_top5, epoch)
                    writer.add_scalar('Valid_Accuracy/Select_Top10', avg_valid_select_top10, epoch)
                    writer.add_scalar('Valid_Accuracy/Branch_Top1', avg_valid_branch_top1, epoch)
                    writer.add_scalar('Valid_Accuracy/Branch_Top5', avg_valid_branch_top5, epoch)
                    writer.add_scalar('Valid_Accuracy/Branch_Top10', avg_valid_branch_top10, epoch)



    # 关闭TensorBoard writer（只在主进程）
    if model_engine.global_rank == 0 and writer is not None:
        writer.close()
        print(f"TensorBoard日志已保存到: {log_dir}")

if __name__ == "__main__":
    train()