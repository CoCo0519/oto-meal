# -*- coding: utf-8 -*-
"""
PPG信号高级分析程序
实现多种降噪方法对比和详细的STFT分析
按照Readtxt.py方案，生成三个对比图像：原始、降噪、STFT
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互后端
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import savgol_filter, medfilt, butter, filtfilt, wiener
import os
import sys
from pathlib import Path

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

def apply_multiple_denoising(ppg_signal, fs):
    """
    应用多种降噪方法
    Args:
        ppg_signal: PPG信号
        fs: 采样率
    Returns:
        dict: 不同降噪方法的结果
    """
    results = {}
    
    # 1. Savitzky-Golay滤波
    window_length = min(51, len(ppg_signal) // 4)
    if window_length % 2 == 0:
        window_length += 1
    
    results['savgol'] = savgol_filter(ppg_signal, window_length, polyorder=3)
    
    # 2. 中值滤波
    results['median'] = medfilt(ppg_signal, kernel_size=5)
    
    # 3. 维纳滤波
    # 估计噪声方差
    noise_var = np.var(ppg_signal - results['savgol'])
    results['wiener'] = wiener(ppg_signal, noise=noise_var)
    
    # 4. 小波降噪（用低通滤波近似）
    cutoff = fs / 8  # 截止频率
    b, a = butter(6, cutoff / (fs/2), btype='low')
    results['wavelet_approx'] = filtfilt(b, a, ppg_signal)
    
    # 5. 组合降噪：中值 + SG
    results['combined'] = savgol_filter(results['median'], window_length, polyorder=3)
    
    # 6. 心率带通滤波
    results['heart_rate_band'] = butter_bandpass_simple(ppg_signal, fs, 0.8, 3.5, order=4)
    
    # 7. 自适应滤波（简化版）
    # 使用移动平均的变体
    adaptive_window = 20
    results['adaptive'] = np.convolve(ppg_signal, np.ones(adaptive_window)/adaptive_window, mode='same')
    
    return results

def calculate_snr(original, denoised):
    """计算信噪比提升"""
    noise = original - denoised
    signal_power = np.var(denoised)
    noise_power = np.var(noise)
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')
    return snr_db

def advanced_ppg_analysis(txt_path, channel='green', save_dir=None):
    """
    高级PPG信号分析
    Args:
        txt_path: txt文件路径
        channel: PPG通道
        save_dir: 保存目录
    """
    print(f"🔍 开始高级PPG信号分析...")
    print(f"📁 输入文件: {txt_path}")
    print(f"📊 PPG通道: {channel.upper()}")
    
    # 参数设置
    fs = 100
    mains = 50
    
    # 1. 加载数据
    print("\n1️⃣ 加载数据...")
    try:
        data = np.loadtxt(txt_path, skiprows=1, encoding='utf-8')
        print(f"   ✅ 数据加载成功 (UTF-8): {data.shape}")
    except:
        try:
            data = np.loadtxt(txt_path, skiprows=1, encoding='gbk')
            print(f"   ✅ 数据加载成功 (GBK): {data.shape}")
        except Exception as e:
            print(f"   ❌ 数据加载失败: {e}")
            return None
    
    # 选择PPG通道
    channel_map = {'green': 0, 'ir': 1, 'red': 2}
    ppg_raw = data[:, channel_map[channel]]
    time_axis = np.arange(len(ppg_raw)) / fs
    
    print(f"   📈 信号长度: {len(ppg_raw)} 样本点 ({time_axis[-1]:.1f}秒)")
    print(f"   📊 数据范围: {np.min(ppg_raw):.1f} ~ {np.max(ppg_raw):.1f}")
    
    # 2. 预处理（按Readtxt.py方案）
    print("\n2️⃣ 信号预处理...")
    ppg_notched = notch_filter_simple(ppg_raw, fs, f0=mains, Q=30)
    ppg_preprocessed = butter_highpass_simple(ppg_notched, fs, fc=0.1, order=2)
    print(f"   ✅ 工频陷波 + 高通滤波完成")
    
    # 3. 多种降噪方法
    print("\n3️⃣ 应用多种降噪方法...")
    denoising_results = apply_multiple_denoising(ppg_preprocessed, fs)
    
    # 计算各方法的SNR
    method_performance = {}
    for method, result in denoising_results.items():
        snr = calculate_snr(ppg_preprocessed, result)
        std_reduction = np.std(ppg_preprocessed) - np.std(result)
        method_performance[method] = {
            'signal': result,
            'snr_db': snr,
            'std_reduction': std_reduction
        }
        print(f"   📊 {method}: SNR={snr:.1f}dB, 标准差减少={std_reduction:.2f}")
    
    # 选择最佳方法
    best_method = max(method_performance.keys(), 
                     key=lambda x: method_performance[x]['snr_db'])
    ppg_best_denoised = method_performance[best_method]['signal']
    print(f"   🏆 最佳降噪方法: {best_method}")
    
    # 4. STFT分析
    print("\n4️⃣ STFT时频分析...")
    nperseg = 256
    noverlap = nperseg // 2
    
    # 对原始预处理信号和最佳降噪信号都做STFT
    freq_orig, time_orig, Zxx_orig = signal.stft(
        ppg_preprocessed, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    
    freq_denoised, time_denoised, Zxx_denoised = signal.stft(
        ppg_best_denoised, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    
    mag_orig_db = 20 * np.log10(np.abs(Zxx_orig) + 1e-12)
    mag_denoised_db = 20 * np.log10(np.abs(Zxx_denoised) + 1e-12)
    
    print(f"   ✅ STFT完成: 频率范围 0-{freq_denoised[-1]:.1f} Hz")
    
    # 5. 生成三个对比图像
    print("\n5️⃣ 生成对比图像...")
    
    # 设置保存目录
    if save_dir is None:
        save_dir = Path(txt_path).parent
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
    
    base_name = Path(txt_path).stem
    
    # 设置matplotlib
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # ====== 图像1: 三合一对比图 ======
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'PPG信号完整分析 - {channel.upper()}通道\n文件: {base_name}', 
                fontsize=16, fontweight='bold')
    
    # 子图1: 原始信号
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(time_axis, ppg_raw, 'b-', linewidth=1, alpha=0.8)
    plt.title('🔵 1. 原始PPG信号', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('幅值')
    plt.grid(True, alpha=0.3)
    
    # 统计信息
    stats_text = (f'均值: {np.mean(ppg_raw):.0f}\n'
                 f'标准差: {np.std(ppg_raw):.0f}\n'
                 f'峰峰值: {np.ptp(ppg_raw):.0f}')
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 子图2: 降噪对比
    ax2 = plt.subplot(3, 1, 2)
    plt.plot(time_axis, ppg_preprocessed, 'g-', linewidth=1, alpha=0.7, label='预处理信号')
    plt.plot(time_axis, ppg_best_denoised, 'r-', linewidth=1.5, label=f'最佳降噪 ({best_method})')
    plt.title('🔴 2. 降噪后PPG信号', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('幅值')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 降噪效果
    best_perf = method_performance[best_method]
    noise_text = (f'SNR提升: {best_perf["snr_db"]:.1f} dB\n'
                 f'标准差减少: {best_perf["std_reduction"]:.1f}\n'
                 f'降噪方法: {best_method}')
    ax2.text(0.02, 0.98, noise_text, transform=ax2.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 子图3: STFT时频图
    ax3 = plt.subplot(3, 1, 3)
    freq_mask = freq_denoised <= 10  # 只显示0-10Hz
    
    im = plt.pcolormesh(time_denoised, freq_denoised[freq_mask], 
                       mag_denoised_db[freq_mask, :],
                       shading='gouraud', cmap='jet')
    
    plt.title('🌈 3. STFT时频谱图 (降噪后)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('时间 (秒)')
    plt.ylabel('频率 (Hz)')
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('幅度 (dB)', rotation=270, labelpad=20)
    
    # 心率频带标注
    plt.axhspan(0.8, 3.5, alpha=0.15, color='white', label='心率频带 (0.8-3.5Hz)')
    plt.legend(loc='upper right')
    
    # 频谱分析结果
    avg_spectrum = np.mean(np.abs(Zxx_denoised)[freq_mask, :], axis=1)
    peak_freq_idx = np.argmax(avg_spectrum)
    peak_freq = freq_denoised[freq_mask][peak_freq_idx]
    estimated_hr = peak_freq * 60
    
    freq_text = (f'主频率: {peak_freq:.2f} Hz\n'
                f'估计心率: {estimated_hr:.0f} BPM\n'
                f'频谱峰值: {np.max(mag_denoised_db):.1f} dB')
    ax3.text(0.02, 0.98, freq_text, transform=ax3.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # 保存主要对比图
    main_save_path = save_dir / f'{base_name}_{channel}_三合一分析.png'
    plt.savefig(main_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 主要对比图已保存: {main_save_path}")
    
    # ====== 图像2: 降噪方法对比 ======
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle(f'降噪方法对比 - {channel.upper()}通道', fontsize=16, fontweight='bold')
    
    methods = ['savgol', 'median', 'wiener', 'wavelet_approx', 
               'combined', 'heart_rate_band', 'adaptive']
    method_names = ['Savitzky-Golay', '中值滤波', '维纳滤波', '小波近似',
                   '组合降噪', '心率带通', '自适应滤波']
    
    for i, (method, name) in enumerate(zip(methods, method_names)):
        if i >= 7:  # 只显示前7个方法
            break
            
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        if method in method_performance:
            denoised_signal = method_performance[method]['signal']
            snr = method_performance[method]['snr_db']
            
            # 绘制对比
            ax.plot(time_axis, ppg_preprocessed, 'b-', linewidth=1, alpha=0.5, label='预处理')
            ax.plot(time_axis, denoised_signal, 'r-', linewidth=1.2, label=name)
            
            ax.set_title(f'{name} (SNR: {snr:.1f}dB)', fontweight='bold')
            ax.set_xlabel('时间 (秒)')
            ax.set_ylabel('幅值')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # 标记最佳方法
            if method == best_method:
                ax.set_facecolor('#f0f8ff')  # 浅蓝色背景
                ax.set_title(f'🏆 {name} (SNR: {snr:.1f}dB) - 最佳', 
                           fontweight='bold', color='red')
    
    # 隐藏最后一个空子图
    if len(methods) < 8:
        axes[3, 1].set_visible(False)
    
    plt.tight_layout()
    
    # 保存降噪对比图
    comparison_save_path = save_dir / f'{base_name}_{channel}_降噪方法对比.png'
    plt.savefig(comparison_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 降噪对比图已保存: {comparison_save_path}")
    
    # ====== 图像3: STFT对比图 ======
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'STFT对比：降噪前后 - {channel.upper()}通道', fontsize=16, fontweight='bold')
    
    # 降噪前STFT
    im1 = ax1.pcolormesh(time_orig, freq_orig[freq_orig <= 10], 
                        mag_orig_db[freq_orig <= 10, :],
                        shading='gouraud', cmap='jet')
    ax1.set_title('降噪前 STFT', fontsize=14, fontweight='bold')
    ax1.set_ylabel('频率 (Hz)')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('幅度 (dB)', rotation=270, labelpad=20)
    
    # 降噪后STFT
    im2 = ax2.pcolormesh(time_denoised, freq_denoised[freq_denoised <= 10], 
                        mag_denoised_db[freq_denoised <= 10, :],
                        shading='gouraud', cmap='jet')
    ax2.set_title(f'降噪后 STFT ({best_method})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('时间 (秒)')
    ax2.set_ylabel('频率 (Hz)')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('幅度 (dB)', rotation=270, labelpad=20)
    
    # 标注心率频带
    for ax in [ax1, ax2]:
        ax.axhspan(0.8, 3.5, alpha=0.15, color='white')
    
    plt.tight_layout()
    
    # 保存STFT对比图
    stft_save_path = save_dir / f'{base_name}_{channel}_STFT对比.png'
    plt.savefig(stft_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ STFT对比图已保存: {stft_save_path}")
    
    # 6. 输出详细分析结果
    print("\n" + "="*70)
    print("🎯 PPG信号高级分析结果")
    print("="*70)
    print(f"📁 输入文件: {Path(txt_path).name}")
    print(f"📊 PPG通道: {channel.upper()}")
    print(f"⏱️  信号时长: {time_axis[-1]:.1f} 秒 ({len(ppg_raw)} 样本点)")
    print(f"🔄 采样率: {fs} Hz")
    print(f"⚡ 工频: {mains} Hz")
    print()
    print("📈 原始信号统计:")
    print(f"   均值: {np.mean(ppg_raw):.2f}")
    print(f"   标准差: {np.std(ppg_raw):.2f}")
    print(f"   峰峰值: {np.ptp(ppg_raw):.2f}")
    print()
    print("🔧 降噪方法性能排名:")
    sorted_methods = sorted(method_performance.items(), 
                          key=lambda x: x[1]['snr_db'], reverse=True)
    for i, (method, perf) in enumerate(sorted_methods, 1):
        marker = "🏆" if method == best_method else f"{i}️⃣"
        print(f"   {marker} {method}: SNR={perf['snr_db']:.1f}dB, "
              f"标准差减少={perf['std_reduction']:.2f}")
    print()
    print("🎵 频谱分析:")
    print(f"   主频率: {peak_freq:.2f} Hz")
    print(f"   估计心率: {estimated_hr:.0f} BPM")
    print(f"   频谱峰值: {np.max(mag_denoised_db):.1f} dB")
    print()
    print("💾 输出文件:")
    print(f"   主要分析图: {main_save_path.name}")
    print(f"   降噪对比图: {comparison_save_path.name}")
    print(f"   STFT对比图: {stft_save_path.name}")
    print("="*70)
    
    return {
        'original': ppg_raw,
        'preprocessed': ppg_preprocessed,
        'best_denoised': ppg_best_denoised,
        'best_method': best_method,
        'method_performance': method_performance,
        'estimated_hr': estimated_hr,
        'peak_frequency': peak_freq,
        'save_paths': {
            'main': main_save_path,
            'comparison': comparison_save_path,
            'stft': stft_save_path
        }
    }

def main():
    """主函数"""
    if len(sys.argv) > 1:
        txt_path = sys.argv[1]
        channel = sys.argv[2] if len(sys.argv) > 2 else 'green'
        save_dir = sys.argv[3] if len(sys.argv) > 3 else None
    else:
        # 默认参数
        txt_path = './hyx_data/喉咙-吞咽6次间隔10秒.txt'
        channel = 'green'
        save_dir = './ppg_analysis_results'
        print("🚀 使用默认参数运行高级分析...")
    
    # 检查文件
    if not os.path.exists(txt_path):
        print(f"❌ 文件不存在: {txt_path}")
        return
    
    try:
        result = advanced_ppg_analysis(txt_path, channel, save_dir)
        if result:
            print("\n🎉 PPG信号高级分析完成！")
            print("📊 查看生成的图像文件了解详细结果")
        else:
            print("\n❌ 分析失败")
    except Exception as e:
        print(f"\n💥 分析过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
