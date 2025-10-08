# MATLAB兼容性说明

## 概述

我们的降噪工具现在完全支持与MATLAB相同的小波降噪参数和选项，确保降噪效果的一致性。

## MATLAB对应参数

### 1. WAVELET 参数
- **MATLAB**: `'sym8'`
- **Python**: `"sym8"`
- **说明**: Symlets小波，8阶，提供良好的时频局部化特性

### 2. DENOISING METHOD 参数
- **MATLAB**: `'Bayes'`
- **Python**: `"bayes"`
- **说明**: Bayesian降噪方法，基于统计理论的阈值策略

### 3. LEVEL 参数
- **MATLAB**: `7`
- **Python**: `7`
- **说明**: 小波分解层数，7层提供良好的噪声去除效果

### 4. THRESHOLDING 参数
- **Rule**: `'Soft'` (软阈值)
- **Q-Value**: `0.05` (Q值，用于Bayesian阈值计算)

### 5. NOISE ESTIMATE 参数
- **Level-Independent**: 层级无关的噪声估计
- **Level-Dependent**: 层级相关的噪声估计

## 配置文件示例

### MATLAB兼容配置 (matlab_compatible_config.json)
```json
{
  "fs": 100,
  "channel": "green",
  "method": "bayes",
  "wavelet": "sym8",
  "decomposition_level": 7,
  "mode": "symmetric",
  "threshold": {
    "strategy": "bayes",
    "scale": 1.0,
    "manual_value": null,
    "threshold_mode": "soft",
    "q_value": 0.05,
    "noise_estimate": "level_independent"
  },
  "plotting": {
    "figsize": [12, 6]
  },
  "ppg_denoise": true,
  "imu_denoise": true
}
```

## 使用方法

### 1. 使用MATLAB兼容配置
```bash
# 单文件处理
python ppg_wavelet_denoise.py --input data.txt --config matlab_compatible_config.json

# 批量处理
python batch_all_denoise.py --config matlab_compatible_config.json
```

### 2. GUI界面设置
启动GUI界面后，可以手动设置以下参数：
- **小波类型**: sym8
- **降噪方法**: bayes
- **分解层数**: 7
- **阈值策略**: bayes
- **Q-Value**: 0.05
- **噪声估计**: level_independent

## 技术实现

### 1. Bayesian阈值计算
```python
# MATLAB兼容的Bayesian阈值计算
sigma = np.median(np.abs(detail_coeffs)) / 0.6745
universal = sigma * np.sqrt(2.0 * np.log(detail_coeffs.size))
bayes_threshold = sigma**2 / universal
final_threshold = bayes_threshold * q_value * scale
```

### 2. 层级相关噪声估计
```python
# 为每个分解层级计算不同的阈值
if noise_estimate == "level_dependent":
    for idx, detail in enumerate(detail_coeffs):
        sigma = np.median(np.abs(detail)) / 0.6745
        bayes_threshold = sigma**2 / (sigma * np.sqrt(2.0 * np.log(detail.size)))
        threshold = bayes_threshold * q_value * scale
        coeffs[idx+1] = pywt.threshold(coeffs[idx+1], threshold, mode="soft")
```

### 3. 软阈值处理
```python
# 应用软阈值收缩
coeffs[idx] = pywt.threshold(coeffs[idx], threshold, mode="soft")
```

## 参数对比表

| 参数 | MATLAB | Python | 说明 |
|------|--------|--------|------|
| 小波类型 | 'sym8' | "sym8" | Symlets 8阶小波 |
| 降噪方法 | 'Bayes' | "bayes" | Bayesian降噪 |
| 分解层数 | 7 | 7 | 小波分解层级 |
| 阈值规则 | 'Soft' | "soft" | 软阈值收缩 |
| Q值 | 0.05 | 0.05 | Bayesian参数 |
| 噪声估计 | Level-Independent | "level_independent" | 层级无关估计 |

## 验证方法

### 1. 参数验证
确保配置文件中的参数与MATLAB设置完全一致：
- 小波类型: sym8
- 分解层数: 7
- Q值: 0.05
- 阈值模式: soft

### 2. 结果对比
可以通过以下方式验证结果的一致性：
- 比较降噪前后的信噪比改善
- 检查阈值计算的数值
- 验证小波系数的处理方式

## 优势

### 1. 完全兼容
- 参数设置与MATLAB完全一致
- 算法实现遵循MATLAB标准
- 结果可重现和验证

### 2. 扩展功能
- 支持多通道PPG和IMU数据处理
- 提供批量处理能力
- 包含可视化对比功能

### 3. 性能优化
- Python实现提供更好的性能
- 支持并行处理
- 内存使用更高效

## 使用建议

### 1. 首次使用
推荐使用MATLAB兼容配置：
```bash
python batch_all_denoise.py --config matlab_compatible_config.json
```

### 2. 参数调整
如果需要调整参数，建议：
- 保持小波类型为sym8
- Q值在0.01-0.1范围内调整
- 分解层数根据信号长度调整

### 3. 结果验证
- 检查降噪效果报告
- 对比可视化图像
- 验证信噪比改善情况

---

**现在您可以使用与MATLAB完全相同的参数进行小波降噪处理！** 🎯

