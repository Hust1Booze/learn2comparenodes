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

def train():
    # 解析命令行参数（DeepSpeed需要）
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    # 初始化分布式训练
    deepspeed.init_distributed()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DTModel()
    
    # 先创建数据集（在DeepSpeed初始化之前）
    dataset = BnBSequentialDataset("/lab/shiyh_lab/12332470/code/foundation/learn2comparenodes/node_selection/data/GISP", model, device, max_samples= 1000)
    
    # DeepSpeed 初始化
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )
    
    # 注意：不要重复定义optimizer，DeepSpeed已经创建了
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 删除这行

    # 计算平均奖励（使用静态方法，不需要模型前向传播）
    if model_engine.global_rank == 0:  # 只在主进程打印
        avg_reward = calculate_average_reward_static(dataset)
        print(f"Average reward: {avg_reward}")
    
    # DataLoader的batch_size应该等于DeepSpeed配置中的train_micro_batch_size_per_gpu
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=bnb_collate)
   
    best_loss = float('inf')
    patience = 1000
    patience_counter = 0

    select_loss_weight = 0
    branch_loss_weight = 10
    
    for epoch in range(100000):
        model_engine.train()
        total_select_loss = 0
        total_branch_loss = 0
        total_select_acc = 0
        total_branch_acc = 0
        total_select_steps = 0
        total_branch_steps = 0

        start_time = time.time()

        for batch in dataloader:
            batch = [x.to(model_engine.device) if torch.is_tensor(x) else x for x in batch]
            batch_tokens, type_ids, actions, candidates, branch_scores, attention_mask = batch

            select_loss, branch_loss, select_steps, branch_steps, select_acc, branch_acc = model_engine(
                batch_tokens, type_ids, attention_mask, actions, candidates, branch_scores
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

        # 早停检查
        current_loss = avg_select_loss + avg_branch_loss
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            # 保存最佳模型（只在主进程保存）
            if model_engine.global_rank == 0:
                model_engine.save_checkpoint("./checkpoints", f"best_model_epoch_{epoch}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if model_engine.global_rank == 0:
                    print(f"Early stopping at epoch {epoch}")
                break

        # 每个 epoch 结束后进行垃圾回收
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train()