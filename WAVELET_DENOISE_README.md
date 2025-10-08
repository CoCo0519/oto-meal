# 小波降噪批处理工具

这是一个基于 `ReadDirectory.py` 逻辑的小波降噪批处理工具，用于对 PPG 和 IMU 数据进行小波近似降噪处理。

## 功能特点

- **批处理**：自动处理目录下的所有 `.txt` 文件
- **小波降噪**：支持多种小波基函数和参数配置
- **图形化配置**：提供直观的 GUI 界面配置降噪参数
- **对比图像**：生成降噪前后的对比可视化图像
- **结果统计**：输出详细的降噪效果统计报告
- **灵活配置**：支持 PPG 和 IMU 信号的独立降噪控制

## 文件结构

```
wavelet_denoise_batch.py    # 主批处理脚本
run_wavelet_denoise.py      # 便捷运行脚本
default_wavelet_config.json # 默认配置文件
WAVELET_DENOISE_README.md   # 本说明文档
```

## 安装依赖

```bash
pip install numpy matplotlib PyWavelets tkinter
```

## 使用方法

### 1. 基本使用（图形化配置）

```bash
python wavelet_denoise_batch.py --dir ./hyx_data
```

或使用便捷脚本：

```bash
python run_wavelet_denoise.py --dir ./hyx_data
```

### 2. 使用配置文件

```bash
python wavelet_denoise_batch.py --dir ./hyx_data --config my_config.json
```

### 3. 配置参数说明

#### 基本参数
- **采样率 (fs)**: 数据采样频率，默认为 100 Hz
- **PPG通道**: 选择要处理的 PPG 通道 (green/ir/red)

#### 小波参数
- **小波类型**: 选择小波基函数
  - db1-db8: Daubechies 小波
  - sym2-sym8: Symlets 小波
  - coif1-coif5: Coiflets 小波
- **分解层数**: 小波分解的层数，通常 4-6 层
- **边界模式**: 信号边界处理模式

#### 阈值参数
- **阈值策略**:
  - `universal`: 通用阈值（推荐）
  - `manual`: 手动设置阈值
- **阈值缩放**: 通用阈值的缩放因子，默认为 1.0
- **阈值模式**: `soft`（软阈值）或 `hard`（硬阈值）

#### 降噪选项
- **PPG降噪**: 是否对 PPG 信号进行小波降噪
- **IMU降噪**: 是否对 IMU (ACC/GYRO) 信号进行小波降噪

## 输出结果

### 目录结构
```
denoised_xxx_YYYYMMDD-HHMM/     # 输出目录
├── denoising_summary.csv       # 详细统计报告
├── wavelet_config.json         # 使用的配置参数
└── [文件名]_denoising_comparison.png  # 对比图像
```

### 统计报告 (CSV)

包含以下信息：
- 文件基本信息（采样点数、时长等）
- 降噪参数设置
- PPG信号的信噪比改善情况
- IMU信号的能量变化情况
- 处理状态和错误信息

### 可视化图像

每个文件生成一个对比图像，包含：
- PPG信号原始 vs 降噪对比
- PPG残差信号
- ACC信号X/Y/Z轴的原始 vs 降噪对比
- ACC残差信号
- GYRO信号（如果存在）的原始 vs 降噪对比

## 配置示例

### 通用降噪配置
```json
{
  "fs": 100,
  "channel": "green",
  "wavelet": "db6",
  "decomposition_level": 4,
  "mode": "symmetric",
  "threshold": {
    "strategy": "universal",
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

### 自定义阈值配置
```json
{
  "fs": 100,
  "channel": "green",
  "wavelet": "sym8",
  "decomposition_level": 5,
  "mode": "symmetric",
  "threshold": {
    "strategy": "manual",
    "scale": 1.0,
    "manual_value": 0.1,
    "threshold_mode": "soft"
  },
  "plotting": {
    "figsize": [16, 8]
  },
  "ppg_denoise": true,
  "imu_denoise": false
}
```

## 与原 ReadDirectory.py 的区别

1. **降噪方法**: 使用小波近似代替模板相减和ANC
2. **配置方式**: 提供图形化界面和配置文件两种方式
3. **输出格式**: 专门针对降噪效果的对比可视化
4. **统计指标**: 提供信噪比改善等降噪专用指标
5. **处理对象**: 同时处理 PPG 和 IMU 数据

## 性能提示

1. **小波选择**: db6 和 sym8 是常用的较好选择
2. **分解层数**: 通常 4-6 层，过大会过度平滑信号
3. **阈值策略**: universal 策略通常能获得较好效果
4. **阈值模式**: soft 阈值通常比 hard 阈值效果更好

## 故障排除

### 常见问题

1. **内存不足**: 对于大文件，考虑降低分解层数或分批处理
2. **降噪过度**: 降低阈值缩放因子或使用 manual 策略调整阈值
3. **降噪不足**: 提高阈值缩放因子或增加分解层数
4. **GUI界面不显示**: 确保安装了 tkinter (`sudo apt-get install python3-tk`)

### 错误信息

- `pywt is required`: 需要安装 PyWavelets
- `Config file not found`: 检查配置文件路径
- `Input file not found`: 检查输入文件是否存在
- `UnicodeDecodeError`: 文件编码问题，尝试指定编码格式

## 技术原理

### 小波降噪过程

1. **小波分解**: 将信号分解为近似系数和细节系数
2. **阈值处理**: 对细节系数应用阈值收缩
3. **信号重构**: 使用处理后的系数重构信号

### 阈值策略

- **Universal阈值**: σ√(2lnN)，其中σ是噪声标准差
- **Manual阈值**: 用户指定的固定阈值
- **Soft阈值**: 收缩到0，保留连续性
- **Hard阈值**: 小于阈值置零，大于阈值保持不变

## 许可证

本工具遵循与原项目相同的许可证。

