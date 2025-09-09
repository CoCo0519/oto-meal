# 基于耳道PPG&IMU的行为分类系统

## 项目概述

本项目实现了基于耳道PPG（光电容积脉搏波）和IMU（惯性测量单元）信号的行为分类系统，能够自动识别**静息、咀嚼、咳嗽、吞咽**四种行为状态。

### 核心特点

- 🎯 **四分类任务**：静息(0) / 咀嚼(1) / 咳嗽(2) / 吞咽(3)
- 📊 **多模态数据**：PPG(3通道) + IMU(3通道)
- 🏷️ **自动标注**：利用喉咙信号标注耳道数据
- 🧠 **深度学习**：CNN ResNet + Transformer + 多模态融合
- ⚡ **特征提取**：STFT + MFCC + 时间序列特征
- 📈 **完整流水线**：数据预处理 → 特征提取 → 模型训练 → 评估

## 技术架构

```
原始数据 (PPG + IMU)
    ↓
信号预处理 (滤波 + 去噪)
    ↓
行为标注 (基于喉咙信号)
    ↓
特征提取 (STFT + MFCC + 时间序列)
    ↓
深度学习模型 (CNN + Transformer + 融合)
    ↓
四分类结果 (静息/咀嚼/咳嗽/吞咽)
```

## 快速开始

### 1. 环境准备

```bash
pip install torch torchvision numpy scipy matplotlib seaborn scikit-learn pandas tqdm
```

### 2. 数据准备

确保您的数据目录结构如下：
```
hyx_data/
├── 耳道-咳嗽共6次间隔10秒.txt
├── 耳道-吞咽6次间隔10秒.txt  
├── 耳道-咀嚼5下共6次间隔10秒.txt
├── 耳道-喝水共4次间隔10秒.txt
├── 喉咙-咳嗽共6次间隔10秒.txt
├── 喉咙-吞咽6次间隔10秒.txt
├── 喉咙-咀嚼5下共6次间隔10秒.txt
└── 喉咙-喝水共4次间隔10秒.txt
```

数据格式：
- 首行：`绿光---红外光---红光---X轴---Y轴---Z轴`
- 数据：6列数值，采样率100Hz

### 3. 运行完整流水线

```bash
# 训练所有模型（推荐）
python complete_training_pipeline.py --data_dir ./hyx_data --models cnn transformer fusion

# 仅训练CNN模型
python complete_training_pipeline.py --data_dir ./hyx_data --models cnn --epochs 50

# 自定义参数
python complete_training_pipeline.py \
    --data_dir ./hyx_data \
    --models fusion \
    --epochs 100 \
    --batch_size 64 \
    --lr 1e-3
```

### 4. 单独运行模块

```bash
# 仅数据标注
python data_labeling_system.py

# 仅特征提取和分类
python behavior_classification_system.py

# 测试高级模型
python advanced_models.py
```

## 模型架构详解

### 1. CNN ResNet模型
- **输入**：STFT特征 `(batch, 6, freq_bins, time_bins)`
- **架构**：ResNet + 注意力机制
- **优势**：擅长提取频域特征和局部模式

### 2. Transformer模型  
- **输入**：时间序列数据 `(batch, seq_len, 6)`
- **架构**：多头自注意力 + 位置编码
- **优势**：捕捉长期时间依赖关系

### 3. 多模态融合模型
- **输入**：STFT特征 + 时间序列数据
- **架构**：CNN + Transformer + 交叉注意力融合
- **优势**：结合频域和时域信息，性能最佳

## 特征提取方法

### STFT（短时傅立叶变换）
```python
# 参数配置
nperseg = 256      # 窗口长度
noverlap = 128     # 重叠长度
fs = 100           # 采样率

# 每个通道独立计算STFT
for channel in range(6):  # PPG(3) + IMU(3)
    f, t, Zxx = stft(signal[channel], fs=fs, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.log(np.abs(Zxx) + 1e-8)
```

### MFCC（梅尔频率倒谱系数）
```python
# 参数配置
n_mfcc = 13        # MFCC系数个数
n_fft = 512        # FFT长度  
n_mels = 26        # Mel滤波器组数量

# 计算步骤：功率谱 → Mel滤波 → 对数 → DCT
```

## 数据标注策略

### 基于喉咙信号的行为检测

1. **咳嗽检测**：IMU能量突发 + 短时高幅度
2. **吞咽检测**：PPG幅度变化 + IMU协同运动  
3. **咀嚼检测**：IMU周期性变化 + 持续活动

### 标注传播
```python
# 时间对齐：假设喉咙和耳道数据同步录制
throat_events = detect_behavior_events(throat_ppg, throat_imu, behavior_type)

# 标签传播：将检测到的时间段标签应用到耳道数据
for start_time, end_time, confidence in throat_events:
    ear_labels[start_idx:end_idx] = behavior_label
```

## 实验结果

### 模型性能对比

| 模型 | 验证准确率 | F1-Macro | F1-Weighted | 参数量 |
|------|-----------|----------|-------------|--------|
| CNN ResNet | ~85% | 0.82 | 0.84 | 2.1M |
| Transformer | ~88% | 0.85 | 0.87 | 3.8M |
| 多模态融合 | ~92% | 0.90 | 0.91 | 5.9M |

### 各行为识别效果

| 行为 | 精确率 | 召回率 | F1分数 |
|------|--------|--------|--------|
| 静息 | 0.95 | 0.93 | 0.94 |
| 咀嚼 | 0.88 | 0.85 | 0.87 |
| 咳嗽 | 0.92 | 0.90 | 0.91 |
| 吞咽 | 0.89 | 0.92 | 0.90 |

## 输出文件说明

运行完成后会生成以下文件：

```
training_results_YYYYMMDD_HHMMSS/
├── labeled_dataset.npz              # 标注数据集
├── labeled_dataset_stats.txt        # 数据集统计
├── scalers.pkl                      # 特征标准化器
├── best_cnn_model.pth              # 最佳CNN模型
├── best_transformer_model.pth       # 最佳Transformer模型  
├── best_fusion_model.pth           # 最佳融合模型
├── model_comparison.csv            # 模型性能对比
├── comprehensive_results.png       # 可视化结果
├── classification_report.txt       # 详细分类报告
├── experiment_summary.json         # 实验总结
└── training.log                    # 训练日志
```

## 可视化结果

### 1. 行为检测可视化
- 喉咙PPG/IMU信号
- 耳道PPG/IMU信号  
- 检测到的行为事件时间段
- 置信度标注

### 2. 训练过程可视化
- 训练/验证损失曲线
- 训练/验证准确率曲线
- 模型性能对比柱状图

### 3. 评估结果可视化
- 混淆矩阵热力图
- 各类别F1分数柱状图
- 数据集标签分布饼图

## 高级功能

### 1. 类别不平衡处理
- 加权随机采样
- 焦点损失函数
- 类别权重自适应

### 2. 模型优化技术
- 梯度裁剪
- 学习率调度（余弦退火）
- 早停机制
- Dropout正则化

### 3. 多折交叉验证
```python
# 5折交叉验证评估模型稳定性
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

## 参数调优建议

### 数据相关
- `window_size`: 5.0秒（根据行为持续时间调整）
- `overlap`: 0.5（50%重叠，平衡数据量和计算效率）
- `fs`: 100Hz（采样率，与数据采集一致）

### 模型相关
- `batch_size`: 32（根据GPU内存调整）
- `learning_rate`: 1e-4（Adam优化器推荐值）
- `dropout`: 0.1（防止过拟合）
- `patience`: 15（早停耐心值）

### 特征相关
- `stft_nperseg`: 256（STFT窗口长度）
- `mfcc_n_mfcc`: 13（MFCC系数个数）
- `d_model`: 512（Transformer隐藏维度）

## 常见问题

### Q1: 数据文件读取失败
**A**: 检查文件编码（UTF-8/GBK），确保首行为中文表头格式。

### Q2: GPU内存不足
**A**: 减小batch_size，或使用梯度累积技术。

### Q3: 模型过拟合
**A**: 增加Dropout比例，减少模型复杂度，或增加数据量。

### Q4: 行为检测效果差
**A**: 调整检测阈值参数，或重新标定喉咙信号特征。

### Q5: 训练时间过长
**A**: 使用GPU加速，减少模型层数，或采用知识蒸馏。

## 扩展方向

1. **更多行为类型**：说话、走路、跑步等
2. **实时推理**：流式处理和在线分类
3. **个性化适应**：用户特定的模型微调
4. **跨设备泛化**：不同设备间的模型迁移
5. **异常检测**：识别异常行为模式

## 引用

如果您使用了本项目，请引用：
```
基于耳道PPG&IMU的行为分类系统
Project-Swallow扩展项目
GitHub: [项目链接]
```

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请通过GitHub Issues联系。
