# -*- coding: utf-8 -*-
"""
PPG信号分析程序 - 确保工作的版本
按照Readtxt.py的方案处理PPG信号，实现降噪和STFT分析
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端，确保能保存图片
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import savgol_filter, medfilt, butter, filtfilt
import os
import sys

def notch_filter_simple(x, fs, f0=50, Q=30):
    """简化的陷波滤波器"""
    nyq = fs / 2.0
    w0 = f0 / nyq
    bw = w0 / Q
    b, a = signal.iirnotch(w0=w0, Q=w0/bw)
    return filtfilt(b, a, x)

def butter_highpass_simple(x, fs, fc=0.1, order=2):
    """简化的高通滤波器"""
    nyq = fs / 2.0
    b, a = butter(order, fc / nyq, btype='high')
    return filtfilt(b, a, x)

def butter_bandpass_simple(x, fs, f1, f2, order=3):
    """简化的带通滤波器"""
    nyq = fs / 2.0
    b, a = butter(order, [f1/nyq, f2/nyq], btype='band')
    return filtfilt(b, a, x)

def analyze_ppg_signal(txt_path, channel='green', save_path=None):
    """
    分析PPG信号
    Args:
        txt_path: txt文件路径
        channel: PPG通道 ('green', 'ir', 'red')
        save_path: 保存路径
    """
    print(f"开始分析PPG信号...")
    print(f"输入文件: {txt_path}")
    print(f"PPG通道: {channel}")
    
    # 参数设置
    fs = 100  # 采样率
    mains = 50  # 工频
    
    # 1. 加载数据
    print("1. 加载数据...")
    try:
        data = np.loadtxt(txt_path, skiprows=1, encoding='utf-8')
        print(f"   数据加载成功 (UTF-8): {data.shape}")
    except:
        try:
            data = np.loadtxt(txt_path, skiprows=1, encoding='gbk')
            print(f"   数据加载成功 (GBK): {data.shape}")
        except Exception as e:
            print(f"   ❌ 数据加载失败: {e}")
            return None
    
    # 2. 选择PPG通道
    channel_map = {'green': 0, 'ir': 1, 'red': 2}
    if channel not in channel_map:
        print(f"   ❌ 无效的通道: {channel}")
        return None
    
    ppg_raw = data[:, channel_map[channel]]
    time_axis = np.arange(len(ppg_raw)) / fs
    
    print(f"   PPG数据: 长度={len(ppg_raw)}, 时长={time_axis[-1]:.1f}秒")
    print(f"   数据范围: {np.min(ppg_raw):.1f} ~ {np.max(ppg_raw):.1f}")
    
    # 3. 信号预处理 (按照Readtxt.py方案)
    print("2. 信号预处理...")
    
    # 工频陷波
    ppg_notched = notch_filter_simple(ppg_raw, fs, f0=mains, Q=30)
    
    # 高通滤波去漂移
    ppg_hp = butter_highpass_simple(ppg_notched, fs, fc=0.1, order=2)
    
    print(f"   预处理完成")
    
    # 4. 高级降噪处理 (类似MATLAB)
    print("3. 降噪处理...")
    
    # Savitzky-Golay滤波
    window_length = min(51, len(ppg_hp) // 4)
    if window_length % 2 == 0:
        window_length += 1
    
    ppg_sg = savgol_filter(ppg_hp, window_length, polyorder=3)
    
    # 中值滤波去脉冲噪声
    ppg_median = medfilt(ppg_hp, kernel_size=5)
    
    # 组合降噪：中值滤波 + SG滤波
    ppg_denoised = savgol_filter(ppg_median, window_length, polyorder=3)
    
    # 心率带通滤波 (0.8-3.5 Hz对应48-210 BPM)
    ppg_hr_band = butter_bandpass_simple(ppg_hp, fs, 0.8, 3.5, order=4)
    
    print(f"   降噪完成")
    
    # 5. STFT分析
    print("4. STFT分析...")
    
    nperseg = 256
    noverlap = nperseg // 2
    
    frequencies, times, Zxx = signal.stft(
        ppg_denoised,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap
    )
    
    magnitude = np.abs(Zxx)
    magnitude_db = 20 * np.log10(magnitude + 1e-12)
    
    print(f"   STFT完成: 频率范围 0-{frequencies[-1]:.1f} Hz")
    
    # 6. 生成对比图
    print("5. 生成图像...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图像
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'PPG信号分析 - {channel.upper()}通道', fontsize=16, fontweight='bold')
    
    # 子图1: 原始信号
    ax1 = plt.subplot(3, 2, (1, 2))
    plt.plot(time_axis, ppg_raw, 'b-', linewidth=1, alpha=0.8, label='原始信号')
    plt.title('1. 原始PPG信号', fontsize=14, fontweight='bold')
    plt.xlabel('时间 (秒)')
    plt.ylabel('幅值')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加统计信息
    stats_text = f'均值: {np.mean(ppg_raw):.1f}\n标准差: {np.std(ppg_raw):.1f}\n峰峰值: {np.ptp(ppg_raw):.1f}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 子图2: 降噪对比
    ax2 = plt.subplot(3, 2, (3, 4))
    plt.plot(time_axis, ppg_hp, 'g-', linewidth=1, alpha=0.6, label='预处理')
    plt.plot(time_axis, ppg_denoised, 'r-', linewidth=1.5, label='组合降噪')
    plt.plot(time_axis, ppg_hr_band, 'm-', linewidth=1, alpha=0.7, label='心率带通')
    plt.title('2. 降噪后PPG信号', fontsize=14, fontweight='bold')
    plt.xlabel('时间 (秒)')
    plt.ylabel('幅值')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 计算降噪效果
    noise_reduction = np.std(ppg_hp) - np.std(ppg_denoised)
    snr_improvement = 20 * np.log10(np.std(ppg_denoised) / (np.std(ppg_hp - ppg_denoised) + 1e-12))
    
    noise_text = f'噪声减少: {noise_reduction:.2f}\nSNR提升: {snr_improvement:.1f} dB'
    ax2.text(0.02, 0.98, noise_text, transform=ax2.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 子图3: STFT时频图
    ax3 = plt.subplot(3, 2, (5, 6))
    
    # 只显示0-10Hz的频率范围
    freq_mask = frequencies <= 10
    
    im = plt.pcolormesh(
        times,
        frequencies[freq_mask],
        magnitude_db[freq_mask, :],
        shading='gouraud',
        cmap='jet'
    )
    
    plt.title('3. STFT时频谱图', fontsize=14, fontweight='bold')
    plt.xlabel('时间 (秒)')
    plt.ylabel('频率 (Hz)')
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('幅度 (dB)', rotation=270, labelpad=20)
    
    # 标注心率频带
    plt.axhspan(0.8, 3.5, alpha=0.2, color='white', label='心率频带')
    plt.legend(loc='upper right')
    
    # 估计心率
    avg_spectrum = np.mean(magnitude[freq_mask, :], axis=1)
    peak_freq_idx = np.argmax(avg_spectrum)
    peak_freq = frequencies[freq_mask][peak_freq_idx]
    estimated_hr = peak_freq * 60
    
    hr_text = f'主频率: {peak_freq:.2f} Hz\n估计心率: {estimated_hr:.0f} BPM'
    ax3.text(0.02, 0.98, hr_text, transform=ax3.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存图像
    if save_path is None:
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
        save_path = f'{base_name}_{channel}_analysis.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ 图像已保存: {save_path}")
    
    # 7. 输出分析结果
    print("\n" + "="*60)
    print("PPG信号分析结果:")
    print(f"  输入文件: {os.path.basename(txt_path)}")
    print(f"  PPG通道: {channel.upper()}")
    print(f"  信号长度: {len(ppg_raw)} 个样本点")
    print(f"  信号时长: {time_axis[-1]:.1f} 秒")
    print(f"  采样率: {fs} Hz")
    print(f"  原始信号统计:")
    print(f"    均值: {np.mean(ppg_raw):.2f}")
    print(f"    标准差: {np.std(ppg_raw):.2f}")
    print(f"    峰峰值: {np.ptp(ppg_raw):.2f}")
    print(f"  降噪效果:")
    print(f"    噪声减少: {noise_reduction:.2f}")
    print(f"    SNR提升: {snr_improvement:.1f} dB")
    print(f"  频谱分析:")
    print(f"    主频率: {peak_freq:.2f} Hz")
    print(f"    估计心率: {estimated_hr:.0f} BPM")
    print(f"  输出图像: {save_path}")
    print("="*60)
    
    return {
        'original': ppg_raw,
        'preprocessed': ppg_hp,
        'denoised': ppg_denoised,
        'hr_band': ppg_hr_band,
        'stft_freq': frequencies,
        'stft_time': times,
        'stft_magnitude': magnitude,
        'estimated_hr': estimated_hr,
        'save_path': save_path
    }

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) > 1:
        txt_path = sys.argv[1]
        channel = sys.argv[2] if len(sys.argv) > 2 else 'green'
    else:
        # 使用默认文件
        txt_path = './hyx_data/喉咙-吞咽6次间隔10秒.txt'
        channel = 'green'
        print("使用默认参数:")
        print(f"  文件: {txt_path}")
        print(f"  通道: {channel}")
    
    # 检查文件是否存在
    if not os.path.exists(txt_path):
        print(f"❌ 文件不存在: {txt_path}")
        return
    
    # 运行分析
    try:
        result = analyze_ppg_signal(txt_path, channel)
        if result:
            print("\n🎉 PPG信号分析完成!")
        else:
            print("\n❌ PPG信号分析失败!")
    except Exception as e:
        print(f"\n❌ 分析过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
