# -*- coding: utf-8 -*-
"""
PPG信号分析和可视化程序
基于Readtxt.py的方案，实现PPG信号的降噪处理和STFT分析

功能：
1. 读取txt文件并预处理PPG信号
2. 应用多种降噪技术（类似MATLAB）
3. 进行STFT时频分析
4. 生成三个对比图像：原始信号、降噪信号、STFT谱图

作者：基于Project-Swallow项目扩展
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import savgol_filter, medfilt, wiener
import seaborn as sns
from pathlib import Path
import argparse

# 导入现有的信号处理模块
from anc_template_pipeline_100Hz import (
    run_pipeline, notch_filter, butter_highpass, butter_bandpass
)

class PPGSignalAnalyzer:
    """PPG信号分析器"""
    
    def __init__(self, fs=100, mains=50):
        self.fs = fs
        self.mains = mains
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
    def load_txt_data(self, txt_path):
        """
        加载txt数据文件
        Args:
            txt_path: txt文件路径
        Returns:
            dict: 包含PPG和IMU数据的字典
        """
        print(f"正在加载数据文件: {txt_path}")
        
        try:
            # 尝试UTF-8编码
            data = np.loadtxt(txt_path, skiprows=1, encoding='utf-8')
        except:
            try:
                # 尝试GBK编码
                data = np.loadtxt(txt_path, skiprows=1, encoding='gbk')
            except Exception as e:
                raise ValueError(f"无法读取文件 {txt_path}: {e}")
        
        # 解析数据列
        ppg_data = {
            'green': data[:, 0],    # 绿光
            'ir': data[:, 1],       # 红外光
            'red': data[:, 2]       # 红光
        }
        
        imu_data = {
            'acc_x': data[:, 3],    # X轴加速度
            'acc_y': data[:, 4],    # Y轴加速度
            'acc_z': data[:, 5]     # Z轴加速度
        }
        
        print(f"数据加载完成，样本数: {len(data)}, 持续时间: {len(data)/self.fs:.1f}秒")
        
        return {
            'ppg': ppg_data,
            'imu': imu_data,
            'raw_data': data,
            'duration': len(data) / self.fs
        }
    
    def preprocess_ppg(self, ppg_signal):
        """
        PPG信号预处理（基于Readtxt.py方案）
        Args:
            ppg_signal: 原始PPG信号
        Returns:
            dict: 预处理后的信号
        """
        print("正在进行PPG信号预处理...")
        
        # 1) 工频陷波
        ppg_notched = notch_filter(ppg_signal, self.fs, mains=self.mains, Q=30)
        
        # 2) 高通滤波去漂移 (0.1 Hz)
        ppg_hp = butter_highpass(ppg_notched, self.fs, fc=0.1, order=2)
        
        return {
            'original': ppg_signal,
            'notched': ppg_notched,
            'highpass': ppg_hp
        }
    
    def advanced_denoising(self, ppg_signal):
        """
        高级降噪处理（类似MATLAB的多种降噪方法）
        Args:
            ppg_signal: 输入PPG信号
        Returns:
            dict: 不同降噪方法的结果
        """
        print("正在应用高级降噪算法...")
        
        # 1. Savitzky-Golay滤波器（平滑滤波）
        # 窗口长度需要是奇数，选择合适的多项式阶数
        window_length = min(51, len(ppg_signal) // 4)
        if window_length % 2 == 0:
            window_length += 1
        
        sg_filtered = savgol_filter(ppg_signal, window_length, polyorder=3)
        
        # 2. 中值滤波（去除脉冲噪声）
        median_filtered = medfilt(ppg_signal, kernel_size=5)
        
        # 3. 维纳滤波（自适应滤波）
        # 估计噪声方差
        noise_var = np.var(ppg_signal - sg_filtered)
        wiener_filtered = wiener(ppg_signal, noise=noise_var)
        
        # 4. 小波降噪（使用离散小波变换）
        wavelet_filtered = self._wavelet_denoise(ppg_signal)
        
        # 5. 组合降噪（多种方法结合）
        # 先应用中值滤波去除脉冲噪声，再用SG滤波平滑
        combined_filtered = savgol_filter(median_filtered, window_length, polyorder=3)
        
        # 6. 基于心率的带通滤波
        # 心率范围通常在0.8-3.5 Hz (48-210 bpm)
        heart_rate_filtered = butter_bandpass(ppg_signal, self.fs, 0.8, 3.5, order=4)
        
        return {
            'savgol': sg_filtered,
            'median': median_filtered,
            'wiener': wiener_filtered,
            'wavelet': wavelet_filtered,
            'combined': combined_filtered,
            'heart_rate_band': heart_rate_filtered
        }
    
    def _wavelet_denoise(self, signal):
        """
        小波降噪（简化版本）
        Args:
            signal: 输入信号
        Returns:
            denoised_signal: 降噪后的信号
        """
        # 简化的小波降噪：使用低通滤波近似
        # 在没有PyWavelets库的情况下，用低通滤波器近似小波降噪效果
        
        # 设计低通滤波器（截止频率为采样率的1/8）
        nyquist = self.fs / 2
        cutoff = nyquist / 4  # 25 Hz for 100 Hz sampling rate
        
        # 设计Butterworth低通滤波器
        b, a = signal.butter(6, cutoff / nyquist, btype='low')
        
        # 应用零相位滤波
        filtered_signal = signal.filtfilt(b, a, signal)
        
        return filtered_signal
    
    def compute_stft(self, ppg_signal, window='hann', nperseg=256, noverlap=None):
        """
        计算短时傅立叶变换(STFT)
        Args:
            ppg_signal: PPG信号
            window: 窗函数类型
            nperseg: 每段的长度
            noverlap: 重叠长度
        Returns:
            dict: STFT结果
        """
        print("正在计算STFT...")
        
        if noverlap is None:
            noverlap = nperseg // 2
        
        # 计算STFT
        frequencies, times, Zxx = signal.stft(
            ppg_signal, 
            fs=self.fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap
        )
        
        # 计算幅度谱和相位谱
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # 转换为分贝
        magnitude_db = 20 * np.log10(magnitude + 1e-12)
        
        return {
            'frequencies': frequencies,
            'times': times,
            'magnitude': magnitude,
            'magnitude_db': magnitude_db,
            'phase': phase,
            'complex': Zxx
        }
    
    def generate_comparison_plot(self, data, ppg_channel='green', denoising_method='combined', 
                               save_path=None, show_plot=True):
        """
        生成三个对比图像：原始信号、降噪信号、STFT谱图
        Args:
            data: 加载的数据
            ppg_channel: PPG通道选择
            denoising_method: 降噪方法选择
            save_path: 保存路径
            show_plot: 是否显示图像
        """
        print(f"正在生成对比图像...")
        
        # 选择PPG通道
        ppg_raw = data['ppg'][ppg_channel]
        
        # 预处理
        preprocessed = self.preprocess_ppg(ppg_raw)
        ppg_preprocessed = preprocessed['highpass']
        
        # 降噪处理
        denoised_results = self.advanced_denoising(ppg_preprocessed)
        ppg_denoised = denoised_results[denoising_method]
        
        # STFT分析
        stft_results = self.compute_stft(ppg_denoised)
        
        # 创建时间轴
        time_axis = np.arange(len(ppg_raw)) / self.fs
        
        # 创建图像
        fig = plt.figure(figsize=(16, 12))
        
        # 设置整体标题
        fig.suptitle(f'PPG信号分析对比 - {ppg_channel.upper()}通道 ({denoising_method}降噪)', 
                    fontsize=16, fontweight='bold')
        
        # 1. 原始信号
        ax1 = plt.subplot(3, 2, (1, 2))
        plt.plot(time_axis, ppg_raw, 'b-', linewidth=1, alpha=0.8)
        plt.title('1. 原始PPG信号', fontsize=14, fontweight='bold')
        plt.xlabel('时间 (秒)')
        plt.ylabel('幅值')
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        ax1.text(0.02, 0.98, f'均值: {np.mean(ppg_raw):.1f}\n标准差: {np.std(ppg_raw):.1f}\n峰峰值: {np.ptp(ppg_raw):.1f}',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. 降噪后信号
        ax2 = plt.subplot(3, 2, (3, 4))
        plt.plot(time_axis, ppg_preprocessed, 'g-', linewidth=1, alpha=0.6, label='预处理')
        plt.plot(time_axis, ppg_denoised, 'r-', linewidth=1.5, label=f'{denoising_method}降噪')
        plt.title('2. 降噪后PPG信号', fontsize=14, fontweight='bold')
        plt.xlabel('时间 (秒)')
        plt.ylabel('幅值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加统计信息
        noise_reduction = np.std(ppg_preprocessed) - np.std(ppg_denoised)
        ax2.text(0.02, 0.98, f'降噪效果:\n噪声减少: {noise_reduction:.2f}\nSNR提升: {20*np.log10(np.std(ppg_denoised)/np.std(ppg_preprocessed-ppg_denoised)+1e-12):.1f}dB',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. STFT时频图
        ax3 = plt.subplot(3, 2, (5, 6))
        
        # 只显示0-10Hz的频率范围（心率相关）
        freq_mask = stft_results['frequencies'] <= 10
        
        im = plt.pcolormesh(
            stft_results['times'], 
            stft_results['frequencies'][freq_mask], 
            stft_results['magnitude_db'][freq_mask, :],
            shading='gouraud',
            cmap='jet'
        )
        
        plt.title('3. STFT时频谱图', fontsize=14, fontweight='bold')
        plt.xlabel('时间 (秒)')
        plt.ylabel('频率 (Hz)')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('幅度 (dB)', rotation=270, labelpad=20)
        
        # 标注心率频带
        plt.axhspan(0.8, 3.5, alpha=0.2, color='white', label='心率频带')
        plt.legend(loc='upper right')
        
        # 添加频谱峰值信息
        avg_spectrum = np.mean(stft_results['magnitude'][freq_mask, :], axis=1)
        peak_freq_idx = np.argmax(avg_spectrum)
        peak_freq = stft_results['frequencies'][freq_mask][peak_freq_idx]
        estimated_hr = peak_freq * 60  # 转换为BPM
        
        ax3.text(0.02, 0.98, f'主频率: {peak_freq:.2f} Hz\n估计心率: {estimated_hr:.0f} BPM',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图像已保存: {save_path}")
        
        # 显示图像
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return {
            'original_signal': ppg_raw,
            'denoised_signal': ppg_denoised,
            'stft_results': stft_results,
            'estimated_heart_rate': estimated_hr
        }
    
    def compare_denoising_methods(self, data, ppg_channel='green', save_path=None):
        """
        比较不同降噪方法的效果
        Args:
            data: 数据
            ppg_channel: PPG通道
            save_path: 保存路径
        """
        print("正在比较不同降噪方法...")
        
        ppg_raw = data['ppg'][ppg_channel]
        preprocessed = self.preprocess_ppg(ppg_raw)
        ppg_preprocessed = preprocessed['highpass']
        
        # 应用不同降噪方法
        denoised_results = self.advanced_denoising(ppg_preprocessed)
        
        # 创建对比图
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'不同降噪方法对比 - {ppg_channel.upper()}通道', fontsize=16, fontweight='bold')
        
        time_axis = np.arange(len(ppg_raw)) / self.fs
        
        methods = ['savgol', 'median', 'wiener', 'wavelet', 'combined', 'heart_rate_band']
        method_names = ['Savitzky-Golay', '中值滤波', '维纳滤波', '小波降噪', '组合降噪', '心率带通']
        
        for i, (method, name) in enumerate(zip(methods, method_names)):
            ax = axes[i//2, i%2]
            
            # 绘制原始和降噪信号
            ax.plot(time_axis, ppg_preprocessed, 'b-', linewidth=1, alpha=0.5, label='预处理')
            ax.plot(time_axis, denoised_results[method], 'r-', linewidth=1.5, label=name)
            
            ax.set_title(f'{name}降噪', fontweight='bold')
            ax.set_xlabel('时间 (秒)')
            ax.set_ylabel('幅值')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 计算降噪性能指标
            snr_improvement = 20 * np.log10(
                np.std(denoised_results[method]) / 
                (np.std(ppg_preprocessed - denoised_results[method]) + 1e-12)
            )
            
            ax.text(0.02, 0.98, f'SNR提升: {snr_improvement:.1f}dB',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            comparison_path = save_path.replace('.png', '_comparison.png')
            plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
            print(f"对比图已保存: {comparison_path}")
        
        plt.show()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PPG信号分析和可视化程序')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入txt文件路径')
    parser.add_argument('--channel', '-c', choices=['green', 'ir', 'red'], 
                       default='green', help='PPG通道选择')
    parser.add_argument('--method', '-m', 
                       choices=['savgol', 'median', 'wiener', 'wavelet', 'combined', 'heart_rate_band'],
                       default='combined', help='降噪方法选择')
    parser.add_argument('--output', '-o', type=str, help='输出图像路径')
    parser.add_argument('--compare', action='store_true', help='比较不同降噪方法')
    parser.add_argument('--fs', type=int, default=100, help='采样率')
    parser.add_argument('--mains', type=int, default=50, help='工频')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not Path(args.input).exists():
        print(f"错误：输入文件不存在 {args.input}")
        return
    
    # 创建分析器
    analyzer = PPGSignalAnalyzer(fs=args.fs, mains=args.mains)
    
    try:
        # 加载数据
        data = analyzer.load_txt_data(args.input)
        
        # 生成输出路径
        if args.output:
            output_path = args.output
        else:
            input_path = Path(args.input)
            output_path = input_path.parent / f"{input_path.stem}_analysis.png"
        
        print(f"\n{'='*60}")
        print(f"PPG信号分析")
        print(f"输入文件: {args.input}")
        print(f"PPG通道: {args.channel}")
        print(f"降噪方法: {args.method}")
        print(f"采样率: {args.fs} Hz")
        print(f"{'='*60}\n")
        
        # 生成主要分析图
        results = analyzer.generate_comparison_plot(
            data, 
            ppg_channel=args.channel,
            denoising_method=args.method,
            save_path=str(output_path)
        )
        
        # 如果需要，生成降噪方法对比
        if args.compare:
            analyzer.compare_denoising_methods(
                data,
                ppg_channel=args.channel,
                save_path=str(output_path)
            )
        
        # 打印分析结果
        print(f"\n{'='*60}")
        print("分析结果:")
        print(f"估计心率: {results['estimated_heart_rate']:.0f} BPM")
        print(f"信号长度: {data['duration']:.1f} 秒")
        print(f"数据点数: {len(results['original_signal'])}")
        print(f"图像保存: {output_path}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 如果没有命令行参数，使用默认示例
    import sys
    if len(sys.argv) == 1:
        print("PPG信号分析程序")
        print("\n使用示例:")
        print("python ppg_signal_analysis.py -i ./hyx_data/喉咙-吞咽6次间隔10秒.txt")
        print("python ppg_signal_analysis.py -i ./hyx_data/耳道-咳嗽共6次间隔10秒.txt -c green -m combined --compare")
        print("\n使用默认参数运行示例...")
        
        # 使用默认文件路径
        default_file = './hyx_data/喉咙-吞咽6次间隔10秒.txt'
        if Path(default_file).exists():
            sys.argv = ['ppg_signal_analysis.py', '-i', default_file, '-c', 'green', '-m', 'combined']
            main()
        else:
            print(f"默认文件不存在: {default_file}")
            print("请指定有效的输入文件路径")
    else:
        main()
