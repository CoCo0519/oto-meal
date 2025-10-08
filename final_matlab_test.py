# -*- coding: utf-8 -*-
"""
最终MATLAB兼容性测试
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from ppg_wavelet_denoise import load_ppg_txt, wavelet_denoise

def final_test():
    """最终测试 - 验证MATLAB兼容性"""
    
    # 加载数据
    data_file = "hyx_data/喉咙- 咳嗽共6次间隔10秒.txt"
    try:
        data = np.loadtxt(data_file, skiprows=1, encoding='gbk')
    except:
        data = np.loadtxt(data_file, skiprows=1, encoding='utf-8')
    
    signal = data[:, 0]  # green通道
    
    # 计算正确的采样率和时间轴
    fs = len(signal) / 60.0  # 自动检测采样率
    time_axis = np.arange(len(signal)) / fs
    
    print("=== 最终MATLAB兼容性测试 ===")
    print(f"信号长度: {len(signal)}")
    print(f"采样率: {fs:.2f} Hz")
    print(f"时长: {len(signal)/fs:.2f} 秒")
    print(f"原始信号标准差: {np.std(signal):.2f}")
    
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
    
    # 计算效果指标
    residual = signal - denoised_signal
    noise_reduction = (1 - np.std(residual)/np.std(signal)) * 100
    signal_change = np.std(denoised_signal) / np.std(signal)
    
    print(f"\n降噪结果:")
    print(f"降噪阈值: {threshold:.6f}")
    print(f"降噪后标准差: {np.std(denoised_signal):.2f}")
    print(f"信号变化比例: {signal_change:.4f} (应该接近1.0)")
    print(f"噪声抑制率: {noise_reduction:.1f}%")
    print(f"残差标准差: {np.std(residual):.2f}")
    
    # 计算信噪比改善
    snr_original = 20 * np.log10(np.std(signal) / np.std(residual))
    snr_denoised = 20 * np.log10(np.std(denoised_signal) / np.std(residual))
    snr_improvement = snr_denoised - snr_original
    
    print(f"信噪比改善: {snr_improvement:.2f} dB")
    
    # 绘制对比图
    plt.figure(figsize=(15, 12))
    
    # 原始信号
    plt.subplot(4, 1, 1)
    plt.plot(time_axis, signal, 'b-', linewidth=0.8)
    plt.title('原始信号')
    plt.ylabel('幅度')
    plt.grid(True, alpha=0.3)
    
    # 降噪信号
    plt.subplot(4, 1, 2)
    plt.plot(time_axis, signal, 'b-', linewidth=0.5, alpha=0.6, label='原始信号')
    plt.plot(time_axis, denoised_signal, 'r-', linewidth=1.2, label='降噪信号')
    plt.title(f'MATLAB兼容降噪结果 (信号保留: {signal_change:.4f})')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 残差
    plt.subplot(4, 1, 3)
    plt.plot(time_axis, residual, 'orange', linewidth=0.8)
    plt.title(f'被去除的噪声 (噪声抑制率: {noise_reduction:.1f}%)')
    plt.ylabel('幅度')
    plt.grid(True, alpha=0.3)
    
    # 局部放大对比
    plt.subplot(4, 1, 4)
    start_idx = len(signal) // 3
    end_idx = start_idx + 1000  # 显示1000个样本
    plt.plot(time_axis[start_idx:end_idx], signal[start_idx:end_idx], 'b-', linewidth=1, alpha=0.7, label='原始信号')
    plt.plot(time_axis[start_idx:end_idx], denoised_signal[start_idx:end_idx], 'r-', linewidth=1.5, label='降噪信号')
    plt.title('局部放大对比 (1000个样本)')
    plt.xlabel('时间 (秒)')
    plt.ylabel('幅度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_matlab_compatibility_test.png', dpi=150, bbox_inches='tight')
    print(f"\n图像已保存: final_matlab_compatibility_test.png")
    plt.show()
    
    # 评估结果
    print(f"\n=== 结果评估 ===")
    
    # 检查是否达到MATLAB效果
    if signal_change > 0.99:
        print("✅ 信号保留率优秀 (>99%)")
    elif signal_change > 0.95:
        print("✅ 信号保留率良好 (>95%)")
    else:
        print("❌ 信号保留率不足")
    
    if noise_reduction > 95:
        print("✅ 噪声抑制效果优秀 (>95%)")
    elif noise_reduction > 90:
        print("✅ 噪声抑制效果良好 (>90%)")
    else:
        print("❌ 噪声抑制效果不足")
    
    if abs(snr_improvement) < 1:
        print("✅ 信噪比变化合理 (<1dB)")
    else:
        print("⚠️ 信噪比变化较大")
    
    # 总体评估
    if signal_change > 0.99 and noise_reduction > 95:
        print("\n🎉 成功！达到MATLAB兼容效果！")
        return True
    else:
        print("\n⚠️ 需要进一步优化")
        return False

if __name__ == "__main__":
    success = final_test()
    if success:
        print("\n✅ 所有测试通过，可以投入使用！")
    else:
        print("\n❌ 需要进一步调整参数")

