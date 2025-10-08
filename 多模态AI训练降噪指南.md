# 多模态AI训练降噪指南

## 概述

本指南专门针对多模态AI训练需求，介绍了如何使用小波降噪工具对PPG和IMU数据进行预处理，为AI模型提供高质量的训练数据。

## 多模态数据特征

### PPG信号通道
- **绿色通道 (Green)**: 主要心率检测通道，信噪比最高
- **红外通道 (IR)**: 辅助心率特征，对运动伪影较不敏感
- **红色通道 (Red)**: 额外生理信息，提供不同波长的光学特性

### IMU信号通道
- **加速度计 (ACC)**: X/Y/Z三轴，提供运动和姿态信息
- **陀螺仪 (GYRO)**: X/Y/Z三轴，提供旋转和角速度信息

## 降噪方法选择

### 1. 小波降噪 (推荐用于标准训练)
```json
{
  "method": "wavelet",
  "wavelet": "db6",
  "decomposition_level": 4,
  "threshold": {
    "strategy": "universal",
    "scale": 1.0,
    "threshold_mode": "soft"
  }
}
```

**优势**:
- 计算高效，适合大规模训练
- 保持信号的主要特征
- 统一的降噪参数

### 2. Bayesian降噪 (推荐用于高噪声数据)
```json
{
  "method": "bayes",
  "wavelet": "db6",
  "decomposition_level": 4,
  "threshold": {
    "strategy": "bayes",
    "scale": 1.0,
    "threshold_mode": "soft"
  }
}
```

**优势**:
- 在低信噪比情况下表现更好
- 更保守的噪声去除策略
- 适合噪声水平较高的数据

## 批量处理配置

### 标准配置 (multi_channel_config.json)
```json
{
  "fs": 100,
  "channel": "green",
  "method": "bayes",
  "wavelet": "db6",
  "decomposition_level": 4,
  "mode": "symmetric",
  "threshold": {
    "strategy": "bayes",
    "scale": 1.0,
    "manual_value": null,
    "threshold_mode": "soft"
  },
  "plotting": {
    "figsize": [12, 6]
  },
  "ppg_denoise": true,
  "imu_denoise": true
}
```

### 批量处理命令
```bash
# 处理所有数据目录
python batch_all_denoise.py --config multi_channel_config.json

# 或启动GUI界面
python batch_all_denoise.py
```

## 输出数据结构

### 目录结构
```
batch_denoised_results/
└── batch_all_YYYYMMDD-HHMM/
    ├── batch_config.json
    ├── batch_summary.txt
    ├── denoised_hyx_YYYYMMDD-HHMM/
    │   ├── denoising_summary.csv
    │   ├── wavelet_config.json
    │   ├── [file]_green_comparison.png
    │   ├── [file]_ir_comparison.png
    │   ├── [file]_red_comparison.png
    │   ├── [file]_acc_comparison.png
    │   └── [file]_gyro_comparison.png
    ├── denoised_lhr_YYYYMMDD-HHMM/
    └── denoised_lj_YYYYMMDD-HHMM/
```

### 数据格式
每个目录包含：
- **CSV报告**: 详细的降噪效果统计
- **配置文件**: 使用的降噪参数
- **对比图像**: 每个通道的降噪前后对比
- **原始数据**: 保持原始txt格式

## AI训练数据准备

### 1. 数据标准化
```python
# 示例：加载降噪后的数据
import numpy as np

# 加载PPG数据
ppg_green = np.loadtxt('denoised_data/green_denoised.txt')
ppg_ir = np.loadtxt('denoised_data/ir_denoised.txt')
ppg_red = np.loadtxt('denoised_data/red_denoised.txt')

# 加载IMU数据
acc_data = np.loadtxt('denoised_data/acc_denoised.txt')  # N x 3
gyro_data = np.loadtxt('denoised_data/gyro_denoised.txt')  # N x 3

# 组合多模态数据
multimodal_data = np.column_stack([
    ppg_green, ppg_ir, ppg_red,  # PPG channels
    acc_data, gyro_data  # IMU data
])
```

### 2. 特征工程
```python
# 示例：提取时域特征
def extract_features(signal, window_size=100):
    features = []
    for i in range(0, len(signal)-window_size, window_size):
        window = signal[i:i+window_size]
        
        # 时域特征
        features.append([
            np.mean(window),      # 均值
            np.std(window),       # 标准差
            np.var(window),       # 方差
            np.max(window),       # 最大值
            np.min(window),       # 最小值
            np.ptp(window),       # 峰峰值
            np.median(window),    # 中位数
        ])
    
    return np.array(features)
```

### 3. 数据增强
```python
# 示例：数据增强策略
def augment_data(signal, noise_level=0.1):
    # 添加高斯噪声
    noise = np.random.normal(0, noise_level * np.std(signal), len(signal))
    augmented = signal + noise
    return augmented

# 时间扭曲
def time_warp(signal, warp_factor=0.1):
    # 简单的线性时间扭曲
    indices = np.arange(len(signal))
    warped_indices = indices + np.random.normal(0, warp_factor, len(signal))
    warped_indices = np.clip(warped_indices, 0, len(signal)-1)
    return np.interp(indices, warped_indices, signal)
```

## 模型训练建议

### 1. 输入数据维度
- **PPG通道**: 3个通道 (green, ir, red)
- **IMU数据**: 6个通道 (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
- **总维度**: 9个通道

### 2. 网络架构建议
```python
# 示例：多模态融合网络
import torch
import torch.nn as nn

class MultimodalNet(nn.Module):
    def __init__(self, input_channels=9, hidden_dim=128, num_classes=5):
        super().__init__()
        
        # PPG分支
        self.ppg_branch = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # IMU分支
        self.imu_branch = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        ppg = x[:, :3, :]  # 前3个通道是PPG
        imu = x[:, 3:, :]  # 后6个通道是IMU
        
        ppg_feat = self.ppg_branch(ppg)
        imu_feat = self.imu_branch(imu)
        
        # 特征融合
        combined = torch.cat([ppg_feat, imu_feat], dim=1)
        output = self.fusion(combined.squeeze(-1))
        
        return output
```

### 3. 训练策略
- **学习率**: 1e-3 到 1e-4
- **批次大小**: 32-64
- **优化器**: Adam或AdamW
- **正则化**: Dropout (0.5), L2正则化
- **数据增强**: 时间扭曲、噪声添加

## 性能评估

### 1. 降噪效果指标
- **信噪比改善**: SNR improvement (dB)
- **均方误差**: MSE reduction
- **峰值信噪比**: PSNR improvement

### 2. 模型性能指标
- **准确率**: Classification accuracy
- **F1分数**: F1-score for each class
- **混淆矩阵**: Confusion matrix
- **ROC曲线**: ROC curves for each class

## 最佳实践

### 1. 数据预处理
- 使用统一的降噪参数
- 保持数据的时间对齐
- 记录降噪参数和效果

### 2. 模型训练
- 使用交叉验证评估性能
- 监控训练和验证损失
- 早停防止过拟合

### 3. 结果分析
- 分析每个通道的贡献
- 可视化特征重要性
- 比较不同降噪方法的效果

## 故障排除

### 1. 数据质量问题
- 检查降噪前后的信噪比
- 验证数据的时间对齐
- 确认所有通道都正常处理

### 2. 模型训练问题
- 调整学习率和批次大小
- 检查数据标准化
- 验证标签的正确性

### 3. 性能问题
- 比较不同降噪方法的效果
- 尝试不同的网络架构
- 调整特征提取策略

## 总结

通过使用小波降噪工具对PPG和IMU数据进行预处理，可以显著提高多模态AI训练的效果。关键是选择合适的降噪方法、保持数据的一致性，并采用适当的模型架构和训练策略。

建议：
1. 首先使用Bayesian降噪方法处理所有数据
2. 生成统一的训练数据集
3. 使用多模态融合网络架构
4. 采用数据增强提高模型泛化能力
5. 持续监控和优化模型性能

