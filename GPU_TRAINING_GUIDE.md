# GPU训练使用指南

## 概述

本指南介绍如何使用修复后的GPU训练功能进行行为分类模型训练。

## 主要修复内容

### 1. GPU设备管理
- ✅ 智能设备选择：自动检测并设置最佳GPU设备
- ✅ 环境变量支持：支持`CUDA_VISIBLE_DEVICES`环境变量
- ✅ 设备ID配置：可通过配置文件或命令行参数指定GPU设备

### 2. 混合精度训练 (AMP)
- ✅ 自动混合精度：根据GPU能力自动启用/禁用
- ✅ 梯度缩放：自动处理梯度缩放和更新
- ✅ 内存优化：减少显存使用，提升训练速度

### 3. 数据加载优化
- ✅ GPU优化参数：`pin_memory`, `persistent_workers`, `prefetch_factor`
- ✅ 智能worker数量：根据CPU核心数自动调整
- ✅ 非阻塞传输：使用`non_blocking=True`提升数据传输效率

### 4. 模型优化
- ✅ 模型编译：支持PyTorch 2.0+的`torch.compile`优化
- ✅ 多GPU支持：自动检测并使用DataParallel
- ✅ 内存监控：实时监控GPU内存使用情况

### 5. 性能优化
- ✅ TF32支持：启用Tensor Float-32加速
- ✅ cuDNN优化：启用`cudnn.benchmark`提升性能
- ✅ 高精度矩阵乘法：优化矩阵运算精度

## 使用方法

### 1. 基本GPU训练

```bash
# 运行完整训练流水线（自动GPU优化）
python run_behavior_classification.py --full-pipeline

# 指定GPU设备
python run_behavior_classification.py --full-pipeline --gpu-id 0

# 启用混合精度训练
python run_behavior_classification.py --full-pipeline --mixed-precision
```

### 2. GPU功能测试

```bash
# 测试GPU功能
python run_behavior_classification.py --test-gpu

# 或使用独立测试脚本
python test_gpu_training.py
```

### 3. 高级配置

```bash
# 自定义批次大小
python run_behavior_classification.py --full-pipeline --batch-size 64

# 限制GPU内存使用
python run_behavior_classification.py --full-pipeline --gpu-memory-fraction 0.8

# 禁用GPU（强制CPU）
python run_behavior_classification.py --full-pipeline --no-gpu
```

### 4. 配置文件方式

创建配置文件`gpu_config.json`：

```json
{
  "data_dir": "./hyx_data",
  "batch_size": 32,
  "learning_rate": 1e-4,
  "num_epochs": 100,
  "model_types": ["cnn", "transformer", "fusion"],
  "use_gpu": true,
  "device_id": 0,
  "use_amp": true,
  "compile_model": true,
  "multi_gpu": false,
  "dataloader_params": {
    "num_workers": 8,
    "pin_memory": true,
    "persistent_workers": true,
    "prefetch_factor": 4,
    "drop_last": true
  }
}
```

然后运行：

```bash
python run_behavior_classification.py --full-pipeline --config gpu_config.json
```

## 环境要求

### 硬件要求
- NVIDIA GPU（支持CUDA）
- 推荐显存：≥4GB（8GB以上效果更佳）
- 多GPU支持：≥2个GPU（可选）

### 软件要求
- Python 3.8+
- PyTorch 1.12+（推荐2.0+以获得完整功能）
- CUDA 11.0+
- cuDNN 8.0+

### 安装命令

```bash
# 安装PyTorch（CUDA版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install numpy pandas matplotlib seaborn scikit-learn tqdm
```

## 性能优化建议

### 1. 显存优化
- 根据显存大小调整批次大小：
  - 4GB显存：batch_size=8
  - 8GB显存：batch_size=32
  - 16GB+显存：batch_size=64+

### 2. 数据加载优化
- 设置合适的`num_workers`（通常为CPU核心数的一半）
- 启用`pin_memory`和`persistent_workers`
- 调整`prefetch_factor`（2-4之间）

### 3. 模型优化
- 启用混合精度训练（AMP）
- 使用模型编译（PyTorch 2.0+）
- 启用TF32加速（Ampere架构GPU）

### 4. 多GPU训练
- 使用DataParallel进行多GPU训练
- 根据GPU数量调整批次大小
- 注意数据加载器的worker数量

## 故障排除

### 1. CUDA内存不足
```bash
# 减少批次大小
python run_behavior_classification.py --full-pipeline --batch-size 8

# 限制GPU内存使用
python run_behavior_classification.py --full-pipeline --gpu-memory-fraction 0.7
```

### 2. GPU不可用
```bash
# 检查CUDA安装
python -c "import torch; print(torch.cuda.is_available())"

# 强制使用CPU
python run_behavior_classification.py --full-pipeline --no-gpu
```

### 3. 混合精度训练失败
- 检查PyTorch版本（需要1.6+）
- 确认GPU支持混合精度
- 检查CUDA版本兼容性

### 4. 数据加载缓慢
- 增加`num_workers`
- 启用`pin_memory`
- 检查数据存储位置（推荐SSD）

## 监控和调试

### 1. GPU状态监控
训练过程中会自动显示：
- GPU内存使用情况
- 训练速度统计
- 设备信息

### 2. 日志记录
- 训练日志保存在`training.log`
- 包含详细的GPU使用信息
- 支持实时监控

### 3. 性能分析
```bash
# 使用nvidia-smi监控GPU状态
nvidia-smi -l 1

# 使用PyTorch profiler分析性能
python -c "import torch; print(torch.cuda.memory_summary())"
```

## 示例命令

### 完整训练示例
```bash
# 使用所有GPU优化功能
python run_behavior_classification.py \
    --full-pipeline \
    --mixed-precision \
    --batch-size 32 \
    --gpu-id 0 \
    --gpu-memory-fraction 0.9 \
    --models cnn transformer fusion
```

### 快速测试示例
```bash
# 快速测试GPU功能
python run_behavior_classification.py --test-gpu

# 快速训练测试
python run_behavior_classification.py --quick-start --batch-size 16
```

### 生产环境示例
```bash
# 生产环境配置
python run_behavior_classification.py \
    --full-pipeline \
    --config production_config.json \
    --data-glob "./*_data" \
    --mixed-precision \
    --batch-size 64
```

## 注意事项

1. **首次运行**：建议先运行`--test-gpu`测试GPU功能
2. **显存管理**：注意监控GPU内存使用，避免OOM错误
3. **数据准备**：确保数据文件格式正确，路径可访问
4. **环境隔离**：建议使用虚拟环境避免依赖冲突
5. **定期清理**：训练完成后及时清理GPU内存缓存

## 技术支持

如果遇到问题，请：
1. 检查GPU驱动和CUDA版本
2. 运行GPU功能测试脚本
3. 查看训练日志文件
4. 确认数据文件完整性

---

**更新日期**: 2024年12月
**版本**: v2.0 (GPU优化版)
