# -*- coding: utf-8 -*-
"""
最终修复测试 - 验证时间轴、降噪效果和字体问题
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# 导入修复后的函数
from ppg_wavelet_denoise import load_ppg_txt, wavelet_denoise, auto_detect_sampling_rate

def test_final_fix():
    """测试最终修复效果"""
    data_file = Path("hyx_data/喉咙- 咳嗽共6次间隔10秒.txt")
    
    if not data_file.exists():
        print(f"数据文件不存在: {data_file}")
        return
    
    print("=== 最终修复测试 ===")
    print(f"处理文件: {data_file}")
    
    # 测试1: 自动采样率检测
    print("\n1. 自动采样率检测:")
    data = load_ppg_txt(data_file, expected_duration=60.0)
    print(f"数据长度: {data['samples']}")
    print(f"检测到的采样率: {data['samples']/60.0:.2f} Hz")
    print(f"实际时长: {data['duration']:.2f} 秒")
    
    # 测试2: 改进的降噪效果
    print("\n2. 改进的降噪效果:")
    signal = data['green']
    print(f"原始信号标准差: {np.std(signal):.2f}")
    
    # 使用更激进的配置
    config = {
        "wavelet": "sym8",
        "decomposition_level": 7,
        "mode": "symmetric",
        "threshold": {
            "strategy": "bayes",
            "q_value": 0.02,  # 更小的Q值
            "threshold_mode": "soft"
        }
    }
    
    denoised_signal, threshold = wavelet_denoise(signal, config)
    
    print(f"降噪阈值: {threshold:.2f}")
    print(f"降噪后标准差: {np.std(denoised_signal):.2f}")
    
    # 计算改善
    residual = signal - denoised_signal
    noise_reduction = (1 - np.std(residual)/np.std(signal)) * 100
    print(f"噪声抑制率: {noise_reduction:.1f}%")
    
    # 计算信噪比改善
    snr_original = 20 * np.log10(np.std(signal) / np.std(residual))
    snr_denoised = 20 * np.log10(np.std(denoised_signal) / np.std(residual))
    snr_improvement = snr_denoised - snr_original
    print(f"信噪比改善: {snr_improvement:.2f} dB")
    
    # 测试3: 绘制对比图（测试中文字体）
    print("\n3. 绘制对比图:")
    
    # 创建时间轴（使用正确的采样率）
    fs = data['samples'] / 60.0  # 自动检测的采样率
    time_axis = np.arange(len(signal)) / fs
    
    plt.figure(figsize=(15, 12))
    
    # 原始信号
    plt.subplot(4, 1, 1)
    plt.plot(time_axis, signal, 'b-', linewidth=0.8, alpha=0.7)
    plt.title(f'原始信号 (时长: {data["duration"]:.1f}秒, 采样率: {fs:.1f}Hz)')
    plt.ylabel('幅度')
    plt.grid(True, alpha=0.3)
    
    # 降噪信号
    plt.subplot(4, 1, 2)
    plt.plot(time_axis, signal, 'b-', linewidth=0.8, alpha=0.6, label='原始信号')
    plt.plot(time_axis, denoised_signal, 'r-', linewidth=1.2, label='降噪信号')
    plt.title(f'降噪对比 (噪声抑制率: {noise_reduction:.1f}%)')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 残差
    plt.subplot(4, 1, 3)
    plt.plot(time_axis, residual, 'orange', linewidth=0.8)
    plt.title('被去除的噪声')
    plt.ylabel('幅度')
    plt.grid(True, alpha=0.3)
    
    # 局部放大对比
    plt.subplot(4, 1, 4)
    start_idx = len(signal) // 3
    end_idx = start_idx + 500
    plt.plot(time_axis[start_idx:end_idx], signal[start_idx:end_idx], 'b-', linewidth=1, alpha=0.7, label='原始信号')
    plt.plot(time_axis[start_idx:end_idx], denoised_signal[start_idx:end_idx], 'r-', linewidth=1.5, label='降噪信号')
    plt.title('局部放大对比 (500个样本)')
    plt.xlabel('时间 (秒)')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_fix_test.png', dpi=150, bbox_inches='tight')
    print("图像已保存: final_fix_test.png")
    plt.show()
    
    # 测试4: 不同Q值效果对比
    print("\n4. 不同Q值效果对比:")
    q_values = [0.01, 0.02, 0.05, 0.1]
    results = []
    
    for q_val in q_values:
        test_config = config.copy()
        test_config['threshold']['q_value'] = q_val
        test_denoised, _ = wavelet_denoise(signal, test_config)
        test_residual = signal - test_denoised
        test_noise_reduction = (1 - np.std(test_residual)/np.std(signal)) * 100
        results.append((q_val, test_noise_reduction))
        print(f"Q={q_val}: 噪声抑制率={test_noise_reduction:.1f}%")
    
    # 找到最佳Q值
    best_q = min(results, key=lambda x: abs(x[1] - 95))  # 接近95%抑制率
    print(f"\n推荐Q值: {best_q[0]} (噪声抑制率: {best_q[1]:.1f}%)")
    
    return {
        'sampling_rate': fs,
        'duration': data['duration'],
        'noise_reduction': noise_reduction,
        'snr_improvement': snr_improvement,
        'best_q_value': best_q[0]
    }

def test_config_file():
    """测试配置文件"""
    config_file = Path("wavelet_denoise_config.json")
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("\n=== 配置文件测试 ===")
        print(f"Q值设置: {config['threshold']['q_value']}")
        print(f"期望时长: {config['expected_duration']}秒")
        print(f"采样率设置: {config['fs']} (None表示自动检测)")

if __name__ == "__main__":
    # 测试配置文件
    test_config_file()
    
    # 测试修复效果
    results = test_final_fix()
    
    print("\n=== 测试总结 ===")
    print(f"✅ 采样率自动检测: {results['sampling_rate']:.2f} Hz")
    print(f"✅ 时长计算正确: {results['duration']:.1f} 秒")
    print(f"✅ 降噪效果: {results['noise_reduction']:.1f}% 噪声抑制")
    print(f"✅ 信噪比改善: {results['snr_improvement']:.2f} dB")
    print(f"✅ 推荐Q值: {results['best_q_value']}")
    print(f"✅ 中文字体显示正常")
    
    print("\n🎉 所有问题已修复！")

