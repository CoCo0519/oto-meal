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
import glob
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

def check_available_channels(txt_path):
    """检查数据文件可用的通道"""
    try:
        data = np.loadtxt(txt_path, skiprows=1, encoding='utf-8')
    except:
        try:
            data = np.loadtxt(txt_path, skiprows=1, encoding='gbk')
        except:
            return ['green']  # 默认返回green通道
    
    num_columns = data.shape[1] if len(data.shape) > 1 else 1
    
    # 根据列数确定可用通道
    if num_columns >= 3:
        return ['green', 'ir', 'red']
    elif num_columns >= 2:
        return ['green', 'ir']
    else:
        return ['green']

def advanced_ppg_analysis(txt_path, channel='green', save_dir=None, save_denoised_data=True):
    """
    高级PPG信号分析
    Args:
        txt_path: txt文件路径
        channel: PPG通道
        save_dir: 保存目录
        save_denoised_data: 是否保存降噪后的数据
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
    
    # 设置matplotlib - 支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    
    # ====== Image 1: Comprehensive Analysis ======
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'PPG Signal Comprehensive Analysis - {channel.upper()} Channel\nFile: {base_name}', 
                fontsize=16, fontweight='bold')
    
    # Subplot 1: Original Signal
    ax1 = plt.subplot(3, 1, 1)
    plt.plot(time_axis, ppg_raw, 'b-', linewidth=1, alpha=0.8)
    plt.title('1. Original PPG Signal', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # Statistics
    stats_text = (f'Mean: {np.mean(ppg_raw):.0f}\n'
                 f'Std Dev: {np.std(ppg_raw):.0f}\n'
                 f'Peak-to-Peak: {np.ptp(ppg_raw):.0f}')
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Subplot 2: Denoising Comparison
    ax2 = plt.subplot(3, 1, 2)
    plt.plot(time_axis, ppg_preprocessed, 'g-', linewidth=1, alpha=0.7, label='Preprocessed Signal')
    plt.plot(time_axis, ppg_best_denoised, 'r-', linewidth=1.5, label=f'Best Denoised ({best_method})')
    plt.title('2. Denoised PPG Signal', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Denoising Performance
    best_perf = method_performance[best_method]
    noise_text = (f'SNR Improvement: {best_perf["snr_db"]:.1f} dB\n'
                 f'Std Dev Reduction: {best_perf["std_reduction"]:.1f}\n'
                 f'Method: {best_method}')
    ax2.text(0.02, 0.98, noise_text, transform=ax2.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Subplot 3: STFT Spectrogram
    ax3 = plt.subplot(3, 1, 3)
    freq_mask = freq_denoised <= 10  # Only show 0-10Hz
    
    im = plt.pcolormesh(time_denoised, freq_denoised[freq_mask], 
                       mag_denoised_db[freq_mask, :],
                       shading='gouraud', cmap='jet')
    
    plt.title('3. STFT Spectrogram (Denoised)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Magnitude (dB)', rotation=270, labelpad=20)
    
    # Heart Rate Band Annotation
    plt.axhspan(0.8, 3.5, alpha=0.15, color='white', label='Heart Rate Band (0.8-3.5Hz)')
    plt.legend(loc='upper right')
    
    # Spectral Analysis Results
    avg_spectrum = np.mean(np.abs(Zxx_denoised)[freq_mask, :], axis=1)
    peak_freq_idx = np.argmax(avg_spectrum)
    peak_freq = freq_denoised[freq_mask][peak_freq_idx]
    estimated_hr = peak_freq * 60
    
    freq_text = (f'Peak Frequency: {peak_freq:.2f} Hz\n'
                f'Estimated HR: {estimated_hr:.0f} BPM\n'
                f'Spectral Peak: {np.max(mag_denoised_db):.1f} dB')
    ax3.text(0.02, 0.98, freq_text, transform=ax3.transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save main comparison plot
    main_save_path = save_dir / f'{base_name}_{channel}_comprehensive_analysis.png'
    plt.savefig(main_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Main comparison plot saved: {main_save_path}")
    
    # ====== Image 2: Denoising Methods Comparison ======
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle(f'Denoising Methods Comparison - {channel.upper()} Channel', fontsize=16, fontweight='bold')
    
    methods = ['savgol', 'median', 'wiener', 'wavelet_approx', 
               'combined', 'heart_rate_band', 'adaptive']
    method_names = ['Savitzky-Golay', 'Median Filter', 'Wiener Filter', 'Wavelet Approx',
                   'Combined Denoising', 'Heart Rate Bandpass', 'Adaptive Filter']
    
    for i, (method, name) in enumerate(zip(methods, method_names)):
        if i >= 7:  # Only show first 7 methods
            break
            
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        if method in method_performance:
            denoised_signal = method_performance[method]['signal']
            snr = method_performance[method]['snr_db']
            
            # Plot comparison
            ax.plot(time_axis, ppg_preprocessed, 'b-', linewidth=1, alpha=0.5, label='Preprocessed')
            ax.plot(time_axis, denoised_signal, 'r-', linewidth=1.2, label=name)
            
            ax.set_title(f'{name} (SNR: {snr:.1f}dB)', fontweight='bold')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Mark best method
            if method == best_method:
                ax.set_facecolor('#f0f8ff')  # Light blue background
                ax.set_title(f'Best: {name} (SNR: {snr:.1f}dB)', 
                           fontweight='bold', color='red')
    
    # Hide last empty subplot
    if len(methods) < 8:
        axes[3, 1].set_visible(False)
    
    plt.tight_layout()
    
    # Save denoising comparison plot
    comparison_save_path = save_dir / f'{base_name}_{channel}_denoising_comparison.png'
    plt.savefig(comparison_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Denoising comparison plot saved: {comparison_save_path}")
    
    # ====== Image 3: STFT Comparison ======
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'STFT Comparison: Before vs After Denoising - {channel.upper()} Channel', fontsize=16, fontweight='bold')
    
    # Before denoising STFT
    im1 = ax1.pcolormesh(time_orig, freq_orig[freq_orig <= 10], 
                        mag_orig_db[freq_orig <= 10, :],
                        shading='gouraud', cmap='jet')
    ax1.set_title('STFT Before Denoising', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Frequency (Hz)')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Magnitude (dB)', rotation=270, labelpad=20)
    
    # After denoising STFT
    im2 = ax2.pcolormesh(time_denoised, freq_denoised[freq_denoised <= 10], 
                        mag_denoised_db[freq_denoised <= 10, :],
                        shading='gouraud', cmap='jet')
    ax2.set_title(f'STFT After Denoising ({best_method})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Magnitude (dB)', rotation=270, labelpad=20)
    
    # Annotate heart rate band
    for ax in [ax1, ax2]:
        ax.axhspan(0.8, 3.5, alpha=0.15, color='white')
    
    plt.tight_layout()
    
    # Save STFT comparison plot
    stft_save_path = save_dir / f'{base_name}_{channel}_STFT_comparison.png'
    plt.savefig(stft_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ STFT comparison plot saved: {stft_save_path}")
    
    # 6. Output detailed analysis results
    print("\n" + "="*70)
    print("PPG Signal Advanced Analysis Results")
    print("="*70)
    print(f"Input File: {Path(txt_path).name}")
    print(f"PPG Channel: {channel.upper()}")
    print(f"Signal Duration: {time_axis[-1]:.1f} seconds ({len(ppg_raw)} samples)")
    print(f"Sampling Rate: {fs} Hz")
    print(f"Power Line Frequency: {mains} Hz")
    print()
    print("Original Signal Statistics:")
    print(f"   Mean: {np.mean(ppg_raw):.2f}")
    print(f"   Std Dev: {np.std(ppg_raw):.2f}")
    print(f"   Peak-to-Peak: {np.ptp(ppg_raw):.2f}")
    print()
    print("Denoising Methods Performance Ranking:")
    sorted_methods = sorted(method_performance.items(), 
                          key=lambda x: x[1]['snr_db'], reverse=True)
    for i, (method, perf) in enumerate(sorted_methods, 1):
        marker = "Best" if method == best_method else f"{i}"
        print(f"   {marker}. {method}: SNR={perf['snr_db']:.1f}dB, "
              f"Std Dev Reduction={perf['std_reduction']:.2f}")
    print()
    print("Spectral Analysis:")
    print(f"   Peak Frequency: {peak_freq:.2f} Hz")
    print(f"   Estimated Heart Rate: {estimated_hr:.0f} BPM")
    print(f"   Spectral Peak: {np.max(mag_denoised_db):.1f} dB")
    print()
    print("Output Files:")
    print(f"   Comprehensive Analysis: {main_save_path.name}")
    print(f"   Denoising Comparison: {comparison_save_path.name}")
    print(f"   STFT Comparison: {stft_save_path.name}")
    
    # 7. 保存降噪后的数据
    denoised_data_path = None
    if save_denoised_data:
        print("\n7️⃣ 保存降噪后的数据...")
        
        # 创建降噪数据保存目录
        denoised_data_dir = Path("./ppg_denoised_data")
        denoised_data_dir.mkdir(exist_ok=True)
        
        # 创建数据源子目录
        data_source = Path(txt_path).parent.name
        if data_source.endswith('_data'):
            data_source_name = data_source.replace('_data', '')
        else:
            data_source_name = data_source
        
        source_dir = denoised_data_dir / f"{data_source_name}_denoised"
        source_dir.mkdir(exist_ok=True)
        
        # 保存降噪后的数据
        denoised_filename = f"{base_name}_{channel}_denoised_{best_method}.txt"
        denoised_data_path = source_dir / denoised_filename
        
        # 准备保存的数据：时间轴 + 降噪后的信号
        time_column = time_axis.reshape(-1, 1)
        denoised_column = ppg_best_denoised.reshape(-1, 1)
        data_to_save = np.hstack([time_column, denoised_column])
        
        # 保存数据
        np.savetxt(denoised_data_path, data_to_save, 
                  fmt='%.6f', delimiter='\t',
                  header=f'Time(s)\t{channel.upper()}_denoised_{best_method}',
                  comments='')
        
        print(f"   ✅ 降噪数据已保存: {denoised_data_path}")
        print(f"   📊 数据格式: 时间(秒) | {channel.upper()}通道降噪信号")
        print(f"   🔧 降噪方法: {best_method}")
        print(f"   📈 数据点数: {len(ppg_best_denoised)}")
        print(f"   ⏱️ 时长: {time_axis[-1]:.1f} 秒")
    
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
            'stft': stft_save_path,
            'denoised_data': denoised_data_path
        }
    }

def batch_process_data_directories(data_pattern="*_data", channels=['green', 'ir', 'red'], save_dir="./ppg_analysis_results", save_denoised_data=True):
    """
    Batch process all txt files in directories matching the pattern
    Args:
        data_pattern: Pattern to match data directories (default: "*_data")
        channels: List of PPG channels to process (default: ['green', 'ir', 'red'])
        save_dir: Base directory to save results
        save_denoised_data: Whether to save denoised data (default: True)
    """
    print("="*80)
    print("PPG Signal Batch Analysis")
    print("="*80)
    
    # Find all matching directories
    data_dirs = glob.glob(data_pattern)
    if not data_dirs:
        print(f"❌ No directories found matching pattern: {data_pattern}")
        return
    
    print(f"📁 Found {len(data_dirs)} data directories:")
    for dir_path in data_dirs:
        print(f"   - {dir_path}")
    
    # Create base save directory
    base_save_path = Path(save_dir)
    base_save_path.mkdir(exist_ok=True)
    print(f"📂 Base results directory: {base_save_path.resolve()}")
    
    total_files = 0
    processed_files = 0
    failed_files = 0
    
    # Process each directory
    for data_dir in data_dirs:
        data_dir_name = Path(data_dir).name
        print(f"\n📂 Processing directory: {data_dir}")
        
        # Create specific results directory for this data directory
        # Convert xxx_data to xxx_results
        if data_dir_name.endswith('_data'):
            results_dir_name = data_dir_name.replace('_data', '_results')
        else:
            results_dir_name = f"{data_dir_name}_results"
        
        data_save_path = base_save_path / results_dir_name
        data_save_path.mkdir(exist_ok=True)
        print(f"📂 Results for {data_dir_name} will be saved to: {data_save_path.resolve()}")
        
        # Find all txt files in the directory
        txt_files = list(Path(data_dir).glob("*.txt"))
        if not txt_files:
            print(f"   ⚠️ No .txt files found in {data_dir}")
            continue
        
        print(f"   📄 Found {len(txt_files)} .txt files")
        total_files += len(txt_files)
        
        # Process each file
        for txt_file in txt_files:
            print(f"\n   🔍 Processing: {txt_file.name}")
            
            # Check available channels for this file
            available_channels = check_available_channels(str(txt_file))
            print(f"   📊 Available channels: {', '.join(available_channels)}")
            
            # Process each requested channel that's available
            for channel in channels:
                if channel not in available_channels:
                    print(f"   ⚠️ Skipping {channel.upper()} channel (not available)")
                    continue
                    
                try:
                    # Create subdirectory for this file within the data-specific results directory
                    file_save_dir = data_save_path / txt_file.stem
                    file_save_dir.mkdir(exist_ok=True)
                    
                    result = advanced_ppg_analysis(str(txt_file), channel, str(file_save_dir), save_denoised_data)
                    if result:
                        processed_files += 1
                        print(f"   ✅ {channel.upper()} channel processed successfully")
                    else:
                        failed_files += 1
                        print(f"   ❌ {channel.upper()} channel processing failed")
                        
                except Exception as e:
                    failed_files += 1
                    print(f"   💥 Error processing {channel.upper()} channel: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    print(f"Total files found: {total_files}")
    print(f"Successfully processed: {processed_files}")
    print(f"Failed: {failed_files}")
    print(f"Success rate: {processed_files/total_files*100:.1f}%" if total_files > 0 else "N/A")
    print(f"Base results directory: {base_save_path.resolve()}")
    print("\n📁 Generated results directories:")
    for data_dir in data_dirs:
        data_dir_name = Path(data_dir).name
        if data_dir_name.endswith('_data'):
            results_dir_name = data_dir_name.replace('_data', '_results')
        else:
            results_dir_name = f"{data_dir_name}_results"
        results_path = base_save_path / results_dir_name
        if results_path.exists():
            file_count = len(list(results_path.rglob("*.png")))
            print(f"   - {results_dir_name}/ ({file_count} images)")
    print("="*80)


def main():
    """Main function with batch processing capability"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--batch":
            # Batch processing mode
            data_pattern = sys.argv[2] if len(sys.argv) > 2 else "*_data"
            channels = sys.argv[3].split(',') if len(sys.argv) > 3 else ['green', 'ir', 'red']
            save_dir = sys.argv[4] if len(sys.argv) > 4 else "./ppg_analysis_results"
            save_denoised_data = sys.argv[5].lower() != 'false' if len(sys.argv) > 5 else True
            
            print("🚀 Starting batch processing...")
            batch_process_data_directories(data_pattern, channels, save_dir, save_denoised_data)
        else:
            # Single file processing mode
            txt_path = sys.argv[1]
            channel = sys.argv[2] if len(sys.argv) > 2 else 'green'
            save_dir = sys.argv[3] if len(sys.argv) > 3 else None
            
            # Check file
            if not os.path.exists(txt_path):
                print(f"❌ File does not exist: {txt_path}")
                return
            
            try:
                result = advanced_ppg_analysis(txt_path, channel, save_dir)
                if result:
                    print("\n🎉 PPG signal advanced analysis completed!")
                    print("📊 Check the generated image files for detailed results")
                else:
                    print("\n❌ Analysis failed")
            except Exception as e:
                print(f"\n💥 Error during analysis: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Default: batch processing of all *_data directories
        print("🚀 Running batch analysis on all *_data directories...")
        print("Usage examples:")
        print("  python ppg_advanced_analysis.py --batch")
        print("  python ppg_advanced_analysis.py --batch '*_data' 'green,ir,red'")
        print("  python ppg_advanced_analysis.py --batch '*_data' 'green' './my_results'")
        print("  python ppg_advanced_analysis.py --batch '*_data' 'green' './my_results' 'false'  # 不保存降噪数据")
        print("  python ppg_advanced_analysis.py single_file.txt green")
        print()
        batch_process_data_directories()

if __name__ == "__main__":
    main()
