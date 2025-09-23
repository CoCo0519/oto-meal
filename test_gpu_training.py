#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU训练功能测试脚本
用于验证修复后的GPU训练功能是否正常工作
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

def test_basic_gpu():
    """测试基本GPU功能"""
    print("🧪 测试基本GPU功能...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    device = torch.device('cuda')
    print(f"✅ 使用GPU: {torch.cuda.get_device_name()}")
    
    # 测试张量运算
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.matmul(x, y)
    
    print("✅ GPU张量运算正常")
    
    # 清理内存
    del x, y, z
    torch.cuda.empty_cache()
    
    return True

def test_mixed_precision():
    """测试混合精度训练"""
    print("🧪 测试混合精度训练...")
    
    try:
        from torch.cuda.amp import GradScaler, autocast
        
        device = torch.device('cuda')
        scaler = GradScaler()
        
        # 创建简单模型
        model = torch.nn.Linear(100, 50).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss()
        
        # 测试混合精度训练步骤
        x = torch.randn(32, 100).to(device)
        y = torch.randn(32, 50).to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            output = model(x)
            loss = criterion(output, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("✅ 混合精度训练正常")
        
        # 清理
        del model, optimizer, criterion, x, y, output, loss
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ 混合精度测试失败: {e}")
        return False

def test_model_compilation():
    """测试模型编译"""
    print("🧪 测试模型编译...")
    
    try:
        if not hasattr(torch, 'compile'):
            print("ℹ️ 模型编译功能不可用（需要PyTorch 2.0+）")
            return True
        
        device = torch.device('cuda')
        
        # 创建简单模型
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10)
        ).to(device)
        
        # 编译模型
        compiled_model = torch.compile(model)
        
        # 测试编译后的模型
        x = torch.randn(32, 100).to(device)
        output = compiled_model(x)
        
        print("✅ 模型编译正常")
        
        # 清理
        del model, compiled_model, x, output
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ 模型编译测试失败: {e}")
        return False

def test_data_loading():
    """测试GPU优化的数据加载"""
    print("🧪 测试GPU优化的数据加载...")
    
    try:
        from torch.utils.data import Dataset, DataLoader
        
        # 创建测试数据集
        class TestDataset(Dataset):
            def __init__(self, size=1000):
                self.data = torch.randn(size, 100)
                self.labels = torch.randint(0, 4, (size,))
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx]
        
        dataset = TestDataset()
        
        # 测试GPU优化的数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        device = torch.device('cuda')
        
        # 测试数据传输
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            
            # 简单计算
            result = torch.sum(batch_data)
            
            break  # 只测试一个batch
        
        print("✅ GPU优化的数据加载正常")
        
        # 清理
        del dataset, dataloader, batch_data, batch_labels, result
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载测试失败: {e}")
        return False

def test_multi_gpu():
    """测试多GPU支持"""
    print("🧪 测试多GPU支持...")
    
    if torch.cuda.device_count() < 2:
        print("ℹ️ 只有一个GPU，跳过多GPU测试")
        return True
    
    try:
        device = torch.device('cuda')
        
        # 创建模型
        model = torch.nn.Linear(100, 50).to(device)
        
        # 使用DataParallel
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            print(f"✅ 多GPU支持正常: {torch.cuda.device_count()} 个GPU")
        else:
            print("ℹ️ 只有一个GPU，无法测试多GPU")
        
        # 清理
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ 多GPU测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("="*60)
    print("GPU训练功能测试")
    print("="*60)
    
    tests = [
        test_basic_gpu,
        test_mixed_precision,
        test_model_compilation,
        test_data_loading,
        test_multi_gpu
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
        print()
    
    print("="*60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有GPU功能测试通过！")
        return True
    else:
        print("⚠️ 部分测试失败，请检查GPU环境")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
