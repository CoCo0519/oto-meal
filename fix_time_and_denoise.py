
# -*- coding: utf-8 -*-
"""
修复时间轴和降噪效果问题
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 修复中文显示
matplotlib.rcParams['axes.unicode_minus'] = False

from ppg_wavelet_denoise import load_ppg_txt, wavelet_denoise

def auto_detect_sampling_rate(data_length, expected_duration=60):
    """自动检测采样率"""
    return data_length / expected_duration

def improved_wavelet_denoise(signal, wavelet='sym8', level=7, q_value=0.05):
    """改进的小波降噪算法"""
    import pywt
    
    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level, mode='symmetric')
    
    # 计算每个层级的阈值 - 使用更激进的降噪
    thresholds = []
    for i in range(1, len(coeffs)):
        detail = coeffs[i]
        if detail.size == 0:
            thresholds.append(0.0)
            continue
            
        # 使用更小的Q值进行更激进的降噪
        sigma = np.median(np.abs(detail)) / 0.6745
        if sigma == 0:
            thresholds.append(0.0)
            continue
            
        N = detail.size
        # 使用更小的Q值 (0.01-0.03) 进行更激进的降噪
        threshold = sigma * np.sqrt(2 * np.log(N)) * (q_value * 0.5)  # 更激进
        thresholds.append(threshold)
    
    # 应用阈值
    for i in range(1, len(coeffs)):
        if i - 1 < len(thresholds):
            threshold = thresholds[i - 1]
            if threshold > 0:
                coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
    
    # 重构
    try:
        reconstructed = pywt.waverec(coeffs, wavelet=wavelet, mode='symmetric')
    except:
        reconstructed = pywt.waverec(coeffs, wavelet=wavelet, mode='periodization')
    
    # 确保长度一致
    if reconstructed.size > signal.size:
        reconstructed = reconstructed[:signal.size]
    elif reconstructed.size < signal.size:
        pad_width = signal.size - reconstructed.size
        reconstructed = np.pad(reconstructed, (0, pad_width), mode='edge')
    
    avg_threshold = np.mean(thresholds) if thresholds else 0.0
    return reconstructed, avg_threshold

def test_with_real_data():
    """使用真实数据测试"""
    data_file = Path("hyx_data/喉咙- 咳嗽共6次间隔10秒.txt")
    
    if not data_file.exists():
        print(f"数据文件不存在: {data_file}")
        return
    
    print(f"正在处理: {data_file}")
    
    # 先加载数据
    try:
        data = np.loadtxt(data_file, skiprows=1, encoding='gbk')
        print(f"数据形状: {data.shape}")
        print(f"数据行数: {data.shape[0]}")
    except:
        data = np.loadtxt(data_file, skiprows=1, encoding='utf-8')
    
    # 自动检测采样率
    expected_duration = 60  # 期望60秒
    actual_fs = auto_detect_sampling_rate(data.shape[0], expected_duration)
    print(f"自动检测的采样率: {actual_fs:.2f} Hz")
    
    # 使用原始100Hz和检测的采样率分别测试
    fs_options = [100, actual_fs]
    
    for fs in fs_options:
        print(f"\n=== 使用采样率 {fs} Hz ===")
        
        # 计算实际时长
        duration = data.shape[0] / fs
        print(f"实际时长: {duration:.2f} 秒")
        
        if "green" in data:
            signal = data[:, 0]  # green通道
        else:
            signal = data[:, 0]  # 第一列
        
        print(f"信号长度: {len(signal)}")
        print(f"原始信号标准差: {np.std(signal):.2f}")
        
        # 改进的降噪
        denoised_signal, threshold = improved_wavelet_denoise(signal, q_value=0.05)
        
        print(f"降噪阈值: {threshold:.2f}")
        print(f"降噪后标准差: {np.std(denoised_signal):.2f}")
        
        # 计算改善
        residual = signal - denoised_signal
        noise_reduction = (1 - np.std(residual)/np.std(signal)) * 100
        print(f"噪声抑制率: {noise_reduction:.1f}%")
        
        # 绘制对比图
        time_axis = np.arange(len(signal)) / fs
        
        plt.figure(figsize=(15, 10))
        
        # 原始信号
        plt.subplot(3, 1, 1)
        plt.plot(time_axis, signal, 'b-', linewidth=0.8, alpha=0.7, label='原始信号')
        plt.title(f'原始信号 (采样率: {fs} Hz, 时长: {duration:.1f}秒)')
        plt.ylabel('幅度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 降噪信号
        plt.subplot(3, 1, 2)
        plt.plot(time_axis, signal, 'b-', linewidth=0.8, alpha=0.6, label='原始信号')
        plt.plot(time_axis, denoised_signal, 'r-', linewidth=1.2, label='降噪信号')
        plt.title(f'降噪对比 (噪声抑制率: {noise_reduction:.1f}%)')
        plt.ylabel('幅度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 残差
        plt.subplot(3, 1, 3)
        plt.plot(time_axis, residual, 'orange', linewidth=0.8, label='被去除的噪声')
        plt.title('残差信号')
        plt.xlabel('时间 (秒)')
        plt.ylabel('幅度')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'denoise_fix_test_fs{fs:.0f}.png', dpi=150, bbox_inches='tight')
        plt.show()

def test_multiple_q_values():
    """测试不同的Q值效果"""
    data_file = Path("hyx_data/喉咙- 咳嗽共6次间隔10秒.txt")
    
    if not data_file.exists():
        print(f"数据文件不存在: {data_file}")
        return
    
    # 加载数据
    try:
        data = np.loadtxt(data_file, skiprows=1, encoding='gbk')
    except:
        data = np.loadtxt(data_file, skiprows=1, encoding='utf-8')
    
    signal = data[:, 0]
    actual_fs = auto_detect_sampling_rate(len(signal), 60)
    time_axis = np.arange(len(signal)) / actual_fs
    
    # 测试不同的Q值
    q_values = [0.01, 0.03, 0.05, 0.1, 0.2]
    
    plt.figure(figsize=(15, 12))
    
    for i, q_val in enumerate(q_values):
        denoised, threshold = improved_wavelet_denoise(signal, q_value=q_val)
        noise_reduction = (1 - np.std(signal - denoised)/np.std(signal)) * 100
        
        plt.subplot(len(q_values), 1, i+1)
        plt.plot(time_axis, signal, 'b-', linewidth=0.5, alpha=0.5, label='原始')
        plt.plot(time_axis, denoised, 'r-', linewidth=1, label=f'降噪 (Q={q_val})')
        plt.title(f'Q值={q_val}, 噪声抑制率={noise_reduction:.1f}%')
        plt.ylabel('幅度')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.xlabel('时间 (秒)')
    plt.tight_layout()
    plt.savefig('q_value_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("=== 修复时间轴和降噪效果测试 ===")
    
    # 测试1: 真实数据
    test_with_real_data()
    
    # 测试2: 不同Q值效果
    print("\n=== 测试不同Q值效果 ===")
    test_multiple_q_values()
    
    print("\n测试完成！")

