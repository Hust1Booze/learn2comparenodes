import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dt_dataset import BnBSequentialDataset,bnb_collate,calculate_average_reward, calculate_average_reward_static
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
    print_interval = 100  # 每隔100个step打印一次
    
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=bnb_collate)
   
    best_loss = float('inf')
    patience = 1000
    patience_counter = 0

    select_loss_weight = 0.05
    branch_loss_weight = 1
    
    for epoch in range(100000):
        model_engine.train()
        total_select_loss = 0
        total_branch_loss = 0
        total_select_acc = 0
        total_branch_acc = 0
        total_select_steps = 0
        total_branch_steps = 0

        start_time = time.time()

        for step, batch in enumerate(dataloader):
            # batch现在包含: sequence_data, type_ids, actions, candidates, branch_scores, attention_mask
            sequence_data, type_ids, actions, candidates, branch_scores, attention_mask = batch
            
            # 将非tensor数据移动到设备 - sequence_data是list不需要移动
            type_ids = type_ids.to(model_engine.device)
            actions = actions.to(model_engine.device)
            attention_mask = attention_mask.to(model_engine.device)

            select_loss, branch_loss, select_steps, branch_steps, select_acc, branch_acc = model_engine(
                sequence_data, type_ids, attention_mask, actions, candidates, branch_scores
            )

            total_loss = select_loss*select_loss_weight + branch_loss*branch_loss_weight

            model_engine.backward(total_loss)
            model_engine.step()

            total_select_loss += select_loss.item()
            total_branch_loss += branch_loss.item()
            total_select_acc += select_acc * select_steps
            total_branch_acc += branch_acc * branch_steps
            total_select_steps += select_steps
            total_branch_steps += branch_steps

            # 每隔print_interval个step打印一次loss信息
            if step % print_interval == 0 and model_engine.global_rank == 0:
                print(f"Epoch {epoch}, Step {step}:")
                print(f"  Select Loss: {select_loss.item():.4f}, Branch Loss: {branch_loss.item():.4f}")
                print(f"  Total Loss: {total_loss.item():.4f}")
                print(f"  Select Acc: {select_acc:.4f}, Branch Acc: {branch_acc:.4f}")
                print(f"  Select Steps: {select_steps}, Branch Steps: {branch_steps}", flush=True)
                
                # 记录step级别的指标到TensorBoard（只在主进程）
                if writer is not None:
                    global_step = epoch * len(dataloader) + step
                    writer.add_scalar('Step_Loss/Select', select_loss.item(), global_step)
                    writer.add_scalar('Step_Loss/Branch', branch_loss.item(), global_step)
                    writer.add_scalar('Step_Loss/Total', total_loss.item(), global_step)
                    writer.add_scalar('Step_Accuracy/Select', select_acc, global_step)
                    writer.add_scalar('Step_Accuracy/Branch', branch_acc, global_step)
                    writer.add_scalar('Step_Count/Select_Steps', select_steps, global_step)
                    writer.add_scalar('Step_Count/Branch_Steps', branch_steps, global_step)

        # 计算平均指标
        avg_select_loss = total_select_loss / len(dataloader)
        avg_branch_loss = total_branch_loss / len(dataloader)
        avg_select_acc = total_select_acc / max(total_select_steps, 1)
        avg_branch_acc = total_branch_acc / max(total_branch_steps, 1)

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
                
                # 记录学习率
                current_lr = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else 1e-4
                writer.add_scalar('Epoch_Config/Learning_Rate', current_lr, epoch)

    
    # 关闭TensorBoard writer（只在主进程）
    if model_engine.global_rank == 0 and writer is not None:
        writer.close()
        print(f"TensorBoard日志已保存到: {log_dir}")

if __name__ == "__main__":
    train()