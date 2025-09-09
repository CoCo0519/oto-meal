# -*- coding: utf-8 -*-
"""
PPG信号分析程序 - 详细输出版本
确保每一步都有输出，便于调试
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import savgol_filter, medfilt, butter, filtfilt
import os
import sys

def notch_filter_simple(x, fs, f0=50, Q=30):
    """简化的陷波滤波器"""
    try:
        nyq = fs / 2.0
        w0 = f0 / nyq
        bw = w0 / Q
        b, a = signal.iirnotch(w0=w0, Q=w0/bw)
        return filtfilt(b, a, x)
    except Exception as e:
        print(f"   陷波滤波器错误: {e}")
        return x

def butter_highpass_simple(x, fs, fc=0.1, order=2):
    """简化的高通滤波器"""
    try:
        nyq = fs / 2.0
        b, a = butter(order, fc / nyq, btype='high')
        return filtfilt(b, a, x)
    except Exception as e:
        print(f"   高通滤波器错误: {e}")
        return x

def butter_bandpass_simple(x, fs, f1, f2, order=3):
    """简化的带通滤波器"""
    try:
        nyq = fs / 2.0
        b, a = butter(order, [f1/nyq, f2/nyq], btype='band')
        return filtfilt(b, a, x)
    except Exception as e:
        print(f"   带通滤波器错误: {e}")
        return x

def analyze_ppg_signal_verbose(txt_path, channel='green', save_path=None):
    """
    详细输出版本的PPG信号分析
    """
    print(f"\n{'='*60}")
    print(f"开始分析PPG信号")
    print(f"输入文件: {txt_path}")
    print(f"PPG通道: {channel}")
    print(f"{'='*60}")
    
    # 参数设置
    fs = 100
    mains = 50
    
    # 1. 检查文件
    print("\n1. 检查输入文件...")
    if not os.path.exists(txt_path):
        print(f"   ❌ 文件不存在: {txt_path}")
        return None
    
    file_size = os.path.getsize(txt_path) / 1024  # KB
    print(f"   ✅ 文件存在，大小: {file_size:.1f} KB")
    
    # 2. 加载数据
    print("\n2. 加载数据...")
    try:
        print("   尝试UTF-8编码...")
        data = np.loadtxt(txt_path, skiprows=1, encoding='utf-8')
        print(f"   ✅ UTF-8加载成功: {data.shape}")
    except Exception as e1:
        print(f"   UTF-8失败: {e1}")
        try:
            print("   尝试GBK编码...")
            data = np.loadtxt(txt_path, skiprows=1, encoding='gbk')
            print(f"   ✅ GBK加载成功: {data.shape}")
        except Exception as e2:
            print(f"   ❌ GBK也失败: {e2}")
            return None
    
    # 3. 验证数据
    print("\n3. 验证数据格式...")
    if data.shape[1] < 6:
        print(f"   ❌ 数据列数不足: {data.shape[1]} < 6")
        return None
    
    print(f"   ✅ 数据格式正确: {data.shape[0]}行 × {data.shape[1]}列")
    
    # 4. 选择PPG通道
    print(f"\n4. 选择PPG通道: {channel}")
    channel_map = {'green': 0, 'ir': 1, 'red': 2}
    if channel not in channel_map:
        print(f"   ❌ 无效通道: {channel}")
        return None
    
    ppg_raw = data[:, channel_map[channel]]
    time_axis = np.arange(len(ppg_raw)) / fs
    
    print(f"   ✅ PPG数据提取成功")
    print(f"   数据长度: {len(ppg_raw)} 样本")
    print(f"   信号时长: {time_axis[-1]:.1f} 秒")
    print(f"   数值范围: {np.min(ppg_raw):.1f} ~ {np.max(ppg_raw):.1f}")
    print(f"   均值: {np.mean(ppg_raw):.2f}")
    print(f"   标准差: {np.std(ppg_raw):.2f}")
    
    # 5. 信号预处理
    print(f"\n5. 信号预处理...")
    
    print("   应用工频陷波滤波器...")
    ppg_notched = notch_filter_simple(ppg_raw, fs, f0=mains, Q=30)
    print(f"   陷波后标准差: {np.std(ppg_notched):.2f}")
    
    print("   应用高通滤波器...")
    ppg_hp = butter_highpass_simple(ppg_notched, fs, fc=0.1, order=2)
    print(f"   高通后标准差: {np.std(ppg_hp):.2f}")
    
    # 6. 降噪处理
    print(f"\n6. 降噪处理...")
    
    # SG滤波
    window_length = min(51, len(ppg_hp) // 4)
    if window_length % 2 == 0:
        window_length += 1
    
    print(f"   Savitzky-Golay滤波 (窗口长度: {window_length})...")
    try:
        ppg_sg = savgol_filter(ppg_hp, window_length, polyorder=3)
        print(f"   SG滤波后标准差: {np.std(ppg_sg):.2f}")
    except Exception as e:
        print(f"   SG滤波失败: {e}")
        ppg_sg = ppg_hp
    
    # 中值滤波
    print("   中值滤波...")
    try:
        ppg_median = medfilt(ppg_hp, kernel_size=5)
        print(f"   中值滤波后标准差: {np.std(ppg_median):.2f}")
    except Exception as e:
        print(f"   中值滤波失败: {e}")
        ppg_median = ppg_hp
    
    # 组合降噪
    print("   组合降噪...")
    try:
        ppg_denoised = savgol_filter(ppg_median, window_length, polyorder=3)
        print(f"   组合降噪后标准差: {np.std(ppg_denoised):.2f}")
    except Exception as e:
        print(f"   组合降噪失败: {e}")
        ppg_denoised = ppg_median
    
    # 心率带通
    print("   心率带通滤波 (0.8-3.5 Hz)...")
    try:
        ppg_hr_band = butter_bandpass_simple(ppg_hp, fs, 0.8, 3.5, order=4)
        print(f"   心率带通后标准差: {np.std(ppg_hr_band):.2f}")
    except Exception as e:
        print(f"   心率带通失败: {e}")
        ppg_hr_band = ppg_hp
    
    # 7. STFT分析
    print(f"\n7. STFT分析...")
    
    nperseg = min(256, len(ppg_denoised) // 4)
    noverlap = nperseg // 2
    
    print(f"   STFT参数: nperseg={nperseg}, noverlap={noverlap}")
    
    try:
        frequencies, times, Zxx = signal.stft(
            ppg_denoised,
            fs=fs,
            window='hann',
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        magnitude = np.abs(Zxx)
        magnitude_db = 20 * np.log10(magnitude + 1e-12)
        
        print(f"   ✅ STFT计算成功")
        print(f"   频率范围: 0 - {frequencies[-1]:.1f} Hz")
        print(f"   时间范围: 0 - {times[-1]:.1f} 秒")
        print(f"   频谱矩阵大小: {magnitude.shape}")
        
    except Exception as e:
        print(f"   ❌ STFT计算失败: {e}")
        return None
    
    # 8. 生成图像
    print(f"\n8. 生成图像...")
    
    try:
        # 设置字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        print("   创建图像布局...")
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'PPG Signal Analysis - {channel.upper()} Channel', fontsize=16, fontweight='bold')
        
        # 原始信号
        print("   绘制原始信号...")
        ax1 = plt.subplot(3, 2, (1, 2))
        plt.plot(time_axis, ppg_raw, 'b-', linewidth=1, alpha=0.8, label='Original Signal')
        plt.title('1. Original PPG Signal', fontsize=14, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        stats_text = f'Mean: {np.mean(ppg_raw):.1f}\nStd: {np.std(ppg_raw):.1f}\nPP: {np.ptp(ppg_raw):.1f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 降噪信号
        print("   绘制降噪信号...")
        ax2 = plt.subplot(3, 2, (3, 4))
        plt.plot(time_axis, ppg_hp, 'g-', linewidth=1, alpha=0.6, label='Preprocessed')
        plt.plot(time_axis, ppg_denoised, 'r-', linewidth=1.5, label='Denoised')
        plt.plot(time_axis, ppg_hr_band, 'm-', linewidth=1, alpha=0.7, label='HR Band')
        plt.title('2. Denoised PPG Signal', fontsize=14, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        noise_reduction = np.std(ppg_hp) - np.std(ppg_denoised)
        noise_text = f'Noise Reduction: {noise_reduction:.2f}'
        ax2.text(0.02, 0.98, noise_text, transform=ax2.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # STFT图
        print("   绘制STFT谱图...")
        ax3 = plt.subplot(3, 2, (5, 6))
        
        # 限制频率范围到10Hz
        freq_mask = frequencies <= 10
        
        im = plt.pcolormesh(
            times,
            frequencies[freq_mask],
            magnitude_db[freq_mask, :],
            shading='gouraud',
            cmap='jet'
        )
        
        plt.title('3. STFT Spectrogram', fontsize=14, fontweight='bold')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Magnitude (dB)', rotation=270, labelpad=20)
        
        # 心率估计
        avg_spectrum = np.mean(magnitude[freq_mask, :], axis=1)
        peak_freq_idx = np.argmax(avg_spectrum)
        peak_freq = frequencies[freq_mask][peak_freq_idx]
        estimated_hr = peak_freq * 60
        
        hr_text = f'Peak Freq: {peak_freq:.2f} Hz\nEst. HR: {estimated_hr:.0f} BPM'
        ax3.text(0.02, 0.98, hr_text, transform=ax3.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图像
        if save_path is None:
            base_name = os.path.splitext(os.path.basename(txt_path))[0]
            save_path = f'{base_name}_{channel}_analysis.png'
        
        print(f"   保存图像到: {save_path}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ 图像保存成功!")
        
    except Exception as e:
        print(f"   ❌ 图像生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 9. 输出最终结果
    print(f"\n{'='*60}")
    print("PPG信号分析完成!")
    print(f"输入文件: {os.path.basename(txt_path)}")
    print(f"PPG通道: {channel.upper()}")
    print(f"信号长度: {len(ppg_raw)} 样本 ({time_axis[-1]:.1f} 秒)")
    print(f"主频率: {peak_freq:.2f} Hz")
    print(f"估计心率: {estimated_hr:.0f} BPM")
    print(f"输出图像: {save_path}")
    print(f"{'='*60}")
    
    return {
        'success': True,
        'save_path': save_path,
        'estimated_hr': estimated_hr,
        'peak_freq': peak_freq
    }

def main():
    """主函数"""
    if len(sys.argv) > 1:
        txt_path = sys.argv[1]
        channel = sys.argv[2] if len(sys.argv) > 2 else 'green'
    else:
        txt_path = './hyx_data/喉咙-吞咽6次间隔10秒.txt'
        channel = 'green'
        print("使用默认参数:")
        print(f"  文件: {txt_path}")
        print(f"  通道: {channel}")
    
    result = analyze_ppg_signal_verbose(txt_path, channel)
    
    if result and result['success']:
        print(f"\n🎉 分析成功完成!")
    else:
        print(f"\n❌ 分析失败!")

if __name__ == "__main__":
    main()
