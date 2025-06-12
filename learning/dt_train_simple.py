import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dt_dataset import BnBSequentialDataset, bnb_collate, calculate_average_reward, calculate_average_reward_static
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
    dataset = BnBSequentialDataset("/data/ltf/dt_foundation/dt_bnb/node_selection/data/GISP", max_samples=1000)
    
    # 计算平均奖励
    avg_reward = calculate_average_reward_static(dataset)
    print(f"Average reward: {avg_reward}")
    
    # 创建DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=bnb_collate)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练参数
    best_loss = float('inf')
    patience = 1000
    patience_counter = 0
    select_loss_weight = 0
    branch_loss_weight = 10
    
    # 创建检查点目录
    os.makedirs("./checkpoints", exist_ok=True)
    
    for epoch in range(100000):
        model.train()
        total_select_loss = 0
        total_branch_loss = 0
        total_select_acc = 0
        total_branch_acc = 0
        total_select_steps = 0
        total_branch_steps = 0

        start_time = time.time()

        for batch in dataloader:
            # batch现在包含: sequence_data, type_ids, actions, candidates, branch_scores, attention_mask
            sequence_data, type_ids, actions, candidates, branch_scores, attention_mask = batch
            
            # 将tensor数据移动到设备 - sequence_data是list不需要移动
            type_ids = type_ids.to(device)
            actions = actions.to(device)
            attention_mask = attention_mask.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            select_loss, branch_loss, select_steps, branch_steps, select_acc, branch_acc = model(
                sequence_data, type_ids, attention_mask, actions, candidates, branch_scores
            )

            # 计算总损失
            total_loss = select_loss * select_loss_weight + branch_loss * branch_loss_weight

            # 反向传播
            total_loss.backward()
            
            # 更新参数
            optimizer.step()

            # 累积统计信息
            total_select_loss += select_loss.item()
            total_branch_loss += branch_loss.item()
            total_select_acc += select_acc * select_steps
            total_branch_acc += branch_acc * branch_steps
            total_select_steps += select_steps
            total_branch_steps += branch_steps

        # 计算平均指标
        avg_select_loss = total_select_loss / len(dataloader)
        avg_branch_loss = total_branch_loss / len(dataloader)
        avg_select_acc = total_select_acc / max(total_select_steps, 1)
        avg_branch_acc = total_branch_acc / max(total_branch_steps, 1)

        end_time = time.time()
        duration = end_time - start_time

        # 打印训练信息
        print(f"Epoch {epoch}:")
        print(f"  Select Loss: {avg_select_loss:.4f}, Branch Loss: {avg_branch_loss:.4f}")
        print(f"  Select Acc: {avg_select_acc:.4f}, Branch Acc: {avg_branch_acc:.4f}")
        print(f"  Time: {duration:.2f} seconds", flush=True)

        # 早停检查
        current_loss = avg_select_loss + avg_branch_loss
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            # 保存最佳模型
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
                'select_loss': avg_select_loss,
                'branch_loss': avg_branch_loss,
                'select_acc': avg_select_acc,
                'branch_acc': avg_branch_acc
            }
            torch.save(checkpoint, f"./checkpoints/best_model_epoch_{epoch}.pt")
            print(f"  Saved best model at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # 每个 epoch 结束后进行垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Training completed!")

if __name__ == "__main__":
    train() 