# -*- coding: utf-8 -*-
"""
测试修复后的降噪效果
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# 导入我们的降噪函数
from ppg_wavelet_denoise import load_ppg_txt, wavelet_denoise

def test_denoise_effect():
    """测试降噪效果"""
    
    # 创建测试信号：正弦波 + 噪声
    fs = 100  # 采样率
    duration = 10  # 持续时间（秒）
    t = np.linspace(0, duration, int(fs * duration))
    
    # 原始信号：低频正弦波 + 高频噪声
    signal = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)
    noise = 0.3 * np.random.randn(len(signal))
    noisy_signal = signal + noise
    
    print(f"原始信号长度: {len(noisy_signal)}")
    print(f"原始信号标准差: {np.std(noisy_signal):.4f}")
    
    # MATLAB兼容配置
    config = {
        "wavelet": "sym8",
        "decomposition_level": 7,
        "mode": "symmetric",
        "threshold": {
            "strategy": "bayes",
            "q_value": 0.05,
            "threshold_mode": "soft"
        }
    }
    
    # 执行降噪
    denoised_signal, threshold = wavelet_denoise(noisy_signal, config)
    
    print(f"降噪阈值: {threshold:.6f}")
    print(f"降噪后标准差: {np.std(denoised_signal):.4f}")
    print(f"噪声抑制: {(1 - np.std(noisy_signal - denoised_signal)/np.std(noisy_signal))*100:.1f}%")
    
    # 绘制对比图
    plt.figure(figsize=(15, 10))
    
    # 原始信号
    plt.subplot(3, 1, 1)
    plt.plot(t, signal, 'g-', linewidth=2, label='Clean Signal')
    plt.plot(t, noisy_signal, 'b-', linewidth=1, alpha=0.7, label='Noisy Signal')
    plt.title('原始信号对比')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 降噪信号
    plt.subplot(3, 1, 2)
    plt.plot(t, signal, 'g-', linewidth=2, label='Clean Signal')
    plt.plot(t, denoised_signal, 'r-', linewidth=1.5, label='Denoised Signal')
    plt.title('降噪信号对比')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 残差
    plt.subplot(3, 1, 3)
    residual = noisy_signal - denoised_signal
    plt.plot(t, residual, 'orange', linewidth=1, label='Residual (Noise)')
    plt.title('残差信号（被去除的噪声）')
    plt.xlabel('时间 (s)')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('denoise_test_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 计算信噪比改善
    snr_original = 20 * np.log10(np.std(signal) / np.std(noise))
    snr_denoised = 20 * np.log10(np.std(denoised_signal) / np.std(residual))
    snr_improvement = snr_denoised - snr_original
    
    print(f"\n信噪比分析:")
    print(f"原始信噪比: {snr_original:.2f} dB")
    print(f"降噪后信噪比: {snr_denoised:.2f} dB")
    print(f"信噪比改善: {snr_improvement:.2f} dB")
    
    return noisy_signal, denoised_signal, threshold

def test_with_real_data():
    """使用真实数据测试"""
    # 查找数据文件
    data_dirs = ["hyx_data", "lhr_data", "lj_data"]
    txt_files = []
    
    for data_dir in data_dirs:
        if Path(data_dir).exists():
            txt_files.extend(list(Path(data_dir).glob("*.txt")))
    
    if not txt_files:
        print("未找到数据文件，跳过真实数据测试")
        return
    
    # 使用第一个找到的文件
    test_file = txt_files[0]
    print(f"\n使用真实数据测试: {test_file}")
    
    try:
        # 加载数据
        data = load_ppg_txt(test_file, fs=100)
        
        if "green" in data:
            signal = data["green"]
            print(f"信号长度: {len(signal)}")
            print(f"信号时长: {len(signal)/100:.1f}秒")
            print(f"原始信号标准差: {np.std(signal):.4f}")
            
            # MATLAB兼容配置
            config = {
                "wavelet": "sym8",
                "decomposition_level": 7,
                "mode": "symmetric",
                "threshold": {
                    "strategy": "bayes",
                    "q_value": 0.05,
                    "threshold_mode": "soft"
                }
            }
            
            # 执行降噪
            denoised_signal, threshold = wavelet_denoise(signal, config)
            
            print(f"降噪阈值: {threshold:.6f}")
            print(f"降噪后标准差: {np.std(denoised_signal):.4f}")
            
            # 计算改善
            residual = signal - denoised_signal
            noise_reduction = (1 - np.std(residual)/np.std(signal)) * 100
            print(f"噪声抑制率: {noise_reduction:.1f}%")
            
            # 绘制对比图
            fs = 100
            t = np.arange(len(signal)) / fs
            
            plt.figure(figsize=(15, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(t, signal, 'b-', linewidth=1, alpha=0.7, label='Original')
            plt.plot(t, denoised_signal, 'r-', linewidth=1.5, label='Denoised')
            plt.title(f'真实数据降噪对比 - {test_file.name}')
            plt.ylabel('幅度')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 1, 2)
            plt.plot(t, residual, 'orange', linewidth=1, label='Removed Noise')
            plt.title('被去除的噪声')
            plt.xlabel('时间 (s)')
            plt.ylabel('幅度')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('real_data_denoise_test.png', dpi=150, bbox_inches='tight')
            plt.show()
            
        else:
            print("数据中未找到green通道")
            
    except Exception as e:
        print(f"真实数据测试失败: {e}")

if __name__ == "__main__":
    print("=== 测试修复后的降噪效果 ===")
    
    # 测试1: 合成信号
    print("\n1. 合成信号测试:")
    test_denoise_effect()
    
    # 测试2: 真实数据
    print("\n2. 真实数据测试:")
    test_with_real_data()
    
    print("\n测试完成！")

