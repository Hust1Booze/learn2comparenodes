#!/usr/bin/env python3
"""
测试脚本：验证修改后的数据集和模型是否正确工作
"""

import torch
from learning.dt_dataset import BnBSequentialDataset, bnb_collate
from learning.dt_model import DTModel

def test_dataset_and_model():
    print("🔍 测试数据集和模型集成...")
    
    # 1. 测试数据集创建
    try:
        dataset = BnBSequentialDataset(
            "/lab/shiyh_lab/12332470/code/foundation/learn2comparenodes/node_selection/data/GISP", 
            max_samples=2  # 使用很小的样本数进行测试
        )
        print(f"✅ 数据集创建成功，包含 {len(dataset)} 个样本")
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        return False
    
    # 2. 测试单个样本
    try:
        if len(dataset) > 0:
            sample = dataset[0]
            sequence_data, type_ids, actions, candidates, branch_scores = sample
            print(f"✅ 单个样本获取成功:")
            print(f"   - sequence_data长度: {len(sequence_data)}")
            print(f"   - type_ids形状: {type_ids.shape}")
            print(f"   - actions形状: {actions.shape}")
            print(f"   - candidates长度: {len(candidates)}")
            print(f"   - branch_scores长度: {len(branch_scores)}")
        else:
            print("⚠️  数据集为空，跳过单个样本测试")
    except Exception as e:
        print(f"❌ 单个样本获取失败: {e}")
        return False
    
    # 3. 测试collate函数
    try:
        if len(dataset) > 0:
            batch_data = [dataset[i] for i in range(min(2, len(dataset)))]
            collated = bnb_collate(batch_data)
            sequence_data_batch, type_ids_batch, actions_batch, candidates_batch, branch_scores_batch, attention_mask = collated
            print(f"✅ Collate函数工作正常:")
            print(f"   - batch sequence_data长度: {len(sequence_data_batch)}")
            print(f"   - batch type_ids形状: {type_ids_batch.shape}")
            print(f"   - batch actions形状: {actions_batch.shape}")
            print(f"   - batch attention_mask形状: {attention_mask.shape}")
    except Exception as e:
        print(f"❌ Collate函数失败: {e}")
        return False
    
    # 4. 测试模型创建
    try:
        model = DTModel()
        print(f"✅ 模型创建成功")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False
    
    # 5. 测试模型前向传播
    try:
        if len(dataset) > 0:
            # 使用CPU进行测试
            device = torch.device('cpu')
            model = model.to(device)
            
            # 准备batch数据
            batch_data = [dataset[i] for i in range(min(2, len(dataset)))]
            collated = bnb_collate(batch_data)
            sequence_data_batch, type_ids_batch, actions_batch, candidates_batch, branch_scores_batch, attention_mask = collated
            
            # 移动到设备
            type_ids_batch = type_ids_batch.to(device)
            actions_batch = actions_batch.to(device)
            attention_mask = attention_mask.to(device)
            
            # 前向传播
            with torch.no_grad():
                select_loss, branch_loss, select_steps, branch_steps, select_acc, branch_acc = model(
                    sequence_data_batch, type_ids_batch, attention_mask, actions_batch, candidates_batch, branch_scores_batch
                )
            
            print(f"✅ 模型前向传播成功:")
            print(f"   - select_loss: {select_loss.item():.4f}")
            print(f"   - branch_loss: {branch_loss.item():.4f}")
            print(f"   - select_steps: {select_steps}")
            print(f"   - branch_steps: {branch_steps}")
            print(f"   - select_acc: {select_acc:.4f}")
            print(f"   - branch_acc: {branch_acc:.4f}")
            
    except Exception as e:
        print(f"❌ 模型前向传播失败: {e}")
        return False
    
    print("🎉 所有测试通过！")
    return True

if __name__ == "__main__":
    success = test_dataset_and_model()
    if success:
        print("\n✅ 代码修改验证成功，可以开始训练！")
    else:
        print("\n❌ 代码修改验证失败，请检查错误信息！") 