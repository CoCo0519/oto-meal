# -*- coding: utf-8 -*-
"""
增强版小波降噪GUI界面

功能特点：
- 实时图像预览（原始、降噪后、对比图）
- 数据导出功能
- 处理时间记录
- 多通道支持
"""

import os
import sys
import json
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # 使用TkAgg后端支持GUI显示
# 修复中文字体显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
from pathlib import Path

try:
    import pywt
except ImportError:
    messagebox.showerror("错误", "需要安装 PyWavelets: pip install PyWavelets")
    sys.exit(1)

from ppg_wavelet_denoise import (
    load_ppg_txt, wavelet_denoise, build_time_axis,
    save_single_plot, save_comparison_plot, process_all_channels,
    matlab_bayes_threshold
)


class EnhancedDenoiseGUI:
    """增强版降噪GUI界面"""

    def __init__(self, root):
        self.root = root
        self.config = self.get_default_config()
        self.current_data = None
        self.current_results = None
        self.processing_time = 0
        
        self.setup_gui()
        self.setup_matplotlib()

    def get_default_config(self):
        """获取默认配置"""
        return {
            "fs": 100.0,
            "channel": "green",
            "method": "bayes",
            "wavelet": "sym8",
            "decomposition_level": 7,
            "mode": "symmetric",
            "threshold": {
                "strategy": "bayes",
                "scale": 1.0,
                "manual_value": None,
                "threshold_mode": "soft",
                "q_value": 0.05,
                "noise_estimate": "level_independent"
            },
            "plotting": {
                "figsize": [12, 6]
            },
            "ppg_denoise": True,
            "imu_denoise": True
        }

    def setup_gui(self):
        """设置GUI界面"""
        self.root.title("增强版小波降噪工具")
        self.root.geometry("1400x900")

        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", width=400)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # 右侧显示面板
        display_frame = ttk.LabelFrame(main_frame, text="实时预览", width=900)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_control_panel(control_frame)
        self.setup_display_panel(display_frame)

    def setup_control_panel(self, parent):
        """设置控制面板"""
        # 文件选择
        file_frame = ttk.LabelFrame(parent, text="文件选择")
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(file_frame, text="选择数据文件", command=self.load_file).pack(pady=5)
        
        self.file_label = ttk.Label(file_frame, text="未选择文件", foreground="gray")
        self.file_label.pack(pady=2)

        # 参数设置
        param_frame = ttk.LabelFrame(parent, text="降噪参数")
        param_frame.pack(fill=tk.X, padx=5, pady=5)

        # 采样率
        ttk.Label(param_frame, text="采样率 (Hz):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.fs_var = tk.DoubleVar(value=self.config["fs"])
        ttk.Entry(param_frame, textvariable=self.fs_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # PPG通道
        ttk.Label(param_frame, text="PPG通道:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.channel_var = tk.StringVar(value=self.config["channel"])
        channel_combo = ttk.Combobox(param_frame, textvariable=self.channel_var,
                                   values=["green", "ir", "red"], state="readonly", width=8)
        channel_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # 降噪方法
        ttk.Label(param_frame, text="降噪方法:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.method_var = tk.StringVar(value=self.config["method"])
        method_combo = ttk.Combobox(param_frame, textvariable=self.method_var,
                                   values=["wavelet", "bayes"], state="readonly", width=8)
        method_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        # 小波类型
        ttk.Label(param_frame, text="小波类型:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.wavelet_var = tk.StringVar(value=self.config["wavelet"])
        wavelet_combo = ttk.Combobox(param_frame, textvariable=self.wavelet_var,
                                   values=["db1", "db2", "db3", "db4", "db5", "db6", "db8",
                                          "sym2", "sym3", "sym4", "sym5", "sym6", "sym8",
                                          "coif1", "coif2", "coif3", "coif4", "coif5"], 
                                   state="readonly", width=8)
        wavelet_combo.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        # 分解层数
        ttk.Label(param_frame, text="分解层数:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.level_var = tk.IntVar(value=self.config["decomposition_level"])
        ttk.Spinbox(param_frame, from_=1, to=10, textvariable=self.level_var, width=8).grid(row=4, column=1, sticky=tk.W, padx=5, pady=2)

        # 阈值策略
        ttk.Label(param_frame, text="阈值策略:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.strategy_var = tk.StringVar(value=self.config["threshold"]["strategy"])
        strategy_combo = ttk.Combobox(param_frame, textvariable=self.strategy_var,
                                    values=["universal", "bayes", "manual"], state="readonly", width=8)
        strategy_combo.grid(row=5, column=1, sticky=tk.W, padx=5, pady=2)

        # Q值
        ttk.Label(param_frame, text="Q-Value:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
        self.q_value_var = tk.DoubleVar(value=self.config["threshold"]["q_value"])
        ttk.Entry(param_frame, textvariable=self.q_value_var, width=10).grid(row=6, column=1, sticky=tk.W, padx=5, pady=2)

        # 噪声估计
        ttk.Label(param_frame, text="噪声估计:").grid(row=7, column=0, sticky=tk.W, padx=5, pady=2)
        self.noise_estimate_var = tk.StringVar(value=self.config["threshold"]["noise_estimate"])
        noise_combo = ttk.Combobox(param_frame, textvariable=self.noise_estimate_var,
                                  values=["level_independent", "level_dependent"], state="readonly", width=8)
        noise_combo.grid(row=7, column=1, sticky=tk.W, padx=5, pady=2)

        # 处理选项
        options_frame = ttk.LabelFrame(parent, text="处理选项")
        options_frame.pack(fill=tk.X, padx=5, pady=5)

        self.ppg_denoise_var = tk.BooleanVar(value=self.config["ppg_denoise"])
        ttk.Checkbutton(options_frame, text="PPG降噪", variable=self.ppg_denoise_var).pack(anchor=tk.W, padx=5, pady=2)

        self.imu_denoise_var = tk.BooleanVar(value=self.config["imu_denoise"])
        ttk.Checkbutton(options_frame, text="IMU降噪", variable=self.imu_denoise_var).pack(anchor=tk.W, padx=5, pady=2)

        # 处理按钮
        process_frame = ttk.LabelFrame(parent, text="处理控制")
        process_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(process_frame, text="开始降噪", command=self.start_denoising).pack(pady=5)
        
        # 进度条
        self.progress = ttk.Progressbar(process_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, padx=5, pady=2)

        # 状态显示
        self.status_label = ttk.Label(process_frame, text="就绪")
        self.status_label.pack(pady=2)

        # 处理时间显示
        self.time_label = ttk.Label(process_frame, text="处理时间: 0s")
        self.time_label.pack(pady=2)

        # 导出按钮
        export_frame = ttk.LabelFrame(parent, text="数据导出")
        export_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(export_frame, text="导出降噪数据", command=self.export_data).pack(pady=5)
        ttk.Button(export_frame, text="保存配置", command=self.save_config).pack(pady=2)
        ttk.Button(export_frame, text="加载配置", command=self.load_config).pack(pady=2)

    def setup_display_panel(self, parent):
        """设置显示面板"""
        # 创建notebook用于切换不同的视图
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 原始信号标签页
        self.original_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.original_frame, text="原始信号")

        # 降噪信号标签页
        self.denoised_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.denoised_frame, text="降噪信号")

        # 对比图标签页
        self.comparison_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.comparison_frame, text="对比图")

        # 统计信息标签页
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="统计信息")

    def setup_matplotlib(self):
        """设置matplotlib图形"""
        # 原始信号图
        self.fig_original = Figure(figsize=(8, 6), dpi=100)
        self.ax_original = self.fig_original.add_subplot(111)
        self.canvas_original = FigureCanvasTkAgg(self.fig_original, self.original_frame)
        self.canvas_original.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 降噪信号图
        self.fig_denoised = Figure(figsize=(8, 6), dpi=100)
        self.ax_denoised = self.fig_denoised.add_subplot(111)
        self.canvas_denoised = FigureCanvasTkAgg(self.fig_denoised, self.denoised_frame)
        self.canvas_denoised.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 对比图
        self.fig_comparison = Figure(figsize=(8, 6), dpi=100)
        self.ax_comparison = self.fig_comparison.add_subplot(111)
        self.canvas_comparison = FigureCanvasTkAgg(self.fig_comparison, self.comparison_frame)
        self.canvas_comparison.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 统计信息
        self.setup_stats_display()

    def setup_stats_display(self):
        """设置统计信息显示"""
        # 创建文本显示区域
        self.stats_text = tk.Text(self.stats_frame, wrap=tk.WORD, height=20, width=80)
        scrollbar = ttk.Scrollbar(self.stats_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=scrollbar.set)
        
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def load_file(self):
        """加载数据文件"""
        filename = filedialog.askopenfilename(
            title="选择数据文件",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.current_data = load_ppg_txt(Path(filename), self.fs_var.get())
                self.file_label.config(text=os.path.basename(filename), foreground="green")
                self.status_label.config(text="文件加载成功")
                
                # 显示原始信号
                self.plot_original_signals()
                
            except Exception as e:
                messagebox.showerror("错误", f"无法加载文件: {e}")
                self.file_label.config(text="加载失败", foreground="red")

    def plot_original_signals(self):
        """绘制原始信号"""
        if not self.current_data:
            return

        self.ax_original.clear()
        
        fs = self.fs_var.get()
        channel = self.channel_var.get()
        
        if channel in self.current_data:
            signal = self.current_data[channel]
            time_axis = build_time_axis(len(signal), fs)
            
            self.ax_original.plot(time_axis, signal, linewidth=1, color='blue', label=f'Original {channel.upper()}')
            self.ax_original.set_title(f'原始 {channel.upper()} 信号')
            self.ax_original.set_xlabel('时间 (s)')
            self.ax_original.set_ylabel('幅度')
            self.ax_original.legend()
            self.ax_original.grid(True, alpha=0.3)
            
            self.canvas_original.draw()

    def start_denoising(self):
        """开始降噪处理"""
        if not self.current_data:
            messagebox.showwarning("警告", "请先加载数据文件")
            return

        # 在新线程中执行降噪，避免GUI冻结
        self.progress.start()
        self.status_label.config(text="正在处理...")
        
        thread = threading.Thread(target=self.denoise_worker)
        thread.daemon = True
        thread.start()

    def denoise_worker(self):
        """降噪工作线程"""
        try:
            start_time = time.time()
            
            # 更新配置
            self.update_config_from_gui()
            
            # 执行降噪
            fs = self.fs_var.get()
            channel = self.channel_var.get()
            
            if channel in self.current_data:
                signal = self.current_data[channel]
                
                # 执行小波降噪
                denoised_signal, threshold = wavelet_denoise(signal, self.config)
                
                # 记录处理时间
                self.processing_time = time.time() - start_time
                
                # 保存结果
                self.current_results = {
                    "original": signal,
                    "denoised": denoised_signal,
                    "threshold": threshold,
                    "channel": channel,
                    "processing_time": self.processing_time
                }
                
                # 在主线程中更新GUI
                self.root.after(0, self.update_display_after_denoising)
                
        except Exception as e:
            self.root.after(0, lambda: self.handle_denoising_error(e))

    def update_display_after_denoising(self):
        """降噪完成后更新显示"""
        self.progress.stop()
        self.status_label.config(text="处理完成")
        self.time_label.config(text=f"处理时间: {self.processing_time:.2f}s")
        
        # 更新所有图像
        self.plot_denoised_signals()
        self.plot_comparison()
        self.update_stats()
        
        messagebox.showinfo("完成", f"降噪处理完成！\n处理时间: {self.processing_time:.2f}秒")

    def handle_denoising_error(self, error):
        """处理降噪错误"""
        self.progress.stop()
        self.status_label.config(text="处理失败")
        messagebox.showerror("错误", f"降噪处理失败: {error}")

    def plot_denoised_signals(self):
        """绘制降噪后的信号"""
        if not self.current_results:
            return

        self.ax_denoised.clear()
        
        fs = self.fs_var.get()
        channel = self.current_results["channel"]
        denoised_signal = self.current_results["denoised"]
        time_axis = build_time_axis(len(denoised_signal), fs)
        
        self.ax_denoised.plot(time_axis, denoised_signal, linewidth=1, color='red', label=f'Denoised {channel.upper()}')
        self.ax_denoised.set_title(f'降噪后 {channel.upper()} 信号')
        self.ax_denoised.set_xlabel('时间 (s)')
        self.ax_denoised.set_ylabel('幅度')
        self.ax_denoised.legend()
        self.ax_denoised.grid(True, alpha=0.3)
        
        self.canvas_denoised.draw()

    def plot_comparison(self):
        """绘制对比图"""
        if not self.current_results:
            return

        self.ax_comparison.clear()
        
        fs = self.fs_var.get()
        channel = self.current_results["channel"]
        original = self.current_results["original"]
        denoised = self.current_results["denoised"]
        time_axis = build_time_axis(len(original), fs)
        
        # 绘制对比
        self.ax_comparison.plot(time_axis, original, linewidth=1, alpha=0.6, color='blue', label='Original')
        self.ax_comparison.plot(time_axis, denoised, linewidth=1.5, color='red', label='Denoised')
        
        # 绘制残差
        residual = original - denoised
        self.ax_comparison_twin = self.ax_comparison.twinx()
        self.ax_comparison_twin.plot(time_axis, residual, linewidth=1, color='green', alpha=0.7, label='Residual')
        
        self.ax_comparison.set_title(f'{channel.upper()} 信号对比')
        self.ax_comparison.set_xlabel('时间 (s)')
        self.ax_comparison.set_ylabel('幅度', color='blue')
        self.ax_comparison_twin.set_ylabel('残差', color='green')
        self.ax_comparison.legend(loc='upper left')
        self.ax_comparison_twin.legend(loc='upper right')
        self.ax_comparison.grid(True, alpha=0.3)
        
        self.canvas_comparison.draw()

    def update_stats(self):
        """更新统计信息"""
        if not self.current_results:
            return

        original = self.current_results["original"]
        denoised = self.current_results["denoised"]
        residual = original - denoised
        
        # 计算统计指标
        snr_original = 20 * np.log10(np.std(original) / (np.std(residual) + 1e-12))
        snr_denoised = 20 * np.log10(np.std(denoised) / (np.std(residual) + 1e-12))
        snr_improvement = snr_denoised - snr_original
        
        # 更新统计文本
        stats_text = f"""
降噪处理统计报告
{'='*50}

基本信息:
- 处理通道: {self.current_results['channel'].upper()}
- 采样率: {self.fs_var.get()} Hz
- 信号长度: {len(original)} 样本
- 信号时长: {len(original) / self.fs_var.get():.2f} 秒
- 处理时间: {self.processing_time:.2f} 秒

降噪参数:
- 小波类型: {self.config['wavelet']}
- 降噪方法: {self.config['method']}
- 分解层数: {self.config['decomposition_level']}
- 阈值策略: {self.config['threshold']['strategy']}
- Q值: {self.config['threshold']['q_value']}
- 阈值: {self.current_results['threshold']:.6f}

信号统计:
原始信号:
- 均值: {np.mean(original):.4f}
- 标准差: {np.std(original):.4f}
- 峰值: {np.max(original):.4f}
- 最小值: {np.min(original):.4f}
- 峰峰值: {np.ptp(original):.4f}

降噪信号:
- 均值: {np.mean(denoised):.4f}
- 标准差: {np.std(denoised):.4f}
- 峰值: {np.max(denoised):.4f}
- 最小值: {np.min(denoised):.4f}
- 峰峰值: {np.ptp(denoised):.4f}

残差信号:
- 均值: {np.mean(residual):.4f}
- 标准差: {np.std(residual):.4f}
- 峰值: {np.max(residual):.4f}
- 最小值: {np.min(residual):.4f}

降噪效果:
- 原始信噪比: {snr_original:.2f} dB
- 降噪后信噪比: {snr_denoised:.2f} dB
- 信噪比改善: {snr_improvement:.2f} dB
- 噪声抑制率: {(1 - np.std(residual)/np.std(original))*100:.1f}%
        """
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)

    def export_data(self):
        """导出降噪数据"""
        if not self.current_results:
            messagebox.showwarning("警告", "没有可导出的数据")
            return

        # 选择导出目录
        export_dir = filedialog.askdirectory(title="选择导出目录")
        if not export_dir:
            return

        try:
            fs = self.fs_var.get()
            channel = self.current_results["channel"]
            original = self.current_results["original"]
            denoised = self.current_results["denoised"]
            residual = original - denoised
            
            # 创建时间轴
            time_axis = np.arange(len(original)) / fs
            
            # 导出CSV格式数据
            csv_filename = os.path.join(export_dir, f"denoised_data_{channel}.csv")
            data_to_export = np.column_stack([
                time_axis,
                original,
                denoised,
                residual
            ])
            
            header = "Time(s),Original,Denoised,Residual"
            np.savetxt(csv_filename, data_to_export, delimiter=',', 
                      header=header, comments='', fmt='%.6f')
            
            # 导出配置文件
            config_filename = os.path.join(export_dir, "denoising_config.json")
            with open(config_filename, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            # 导出统计报告
            stats_filename = os.path.join(export_dir, "denoising_report.txt")
            with open(stats_filename, 'w', encoding='utf-8') as f:
                f.write(f"降噪处理报告\n")
                f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"处理时间: {self.processing_time:.2f}秒\n")
                f.write(f"信号时长: {len(original) / fs:.2f}秒\n")
                f.write(f"通道: {channel.upper()}\n")
                f.write(f"阈值: {self.current_results['threshold']:.6f}\n")
            
            # 保存图像
            self.save_export_images(export_dir)
            
            messagebox.showinfo("成功", f"数据已导出到:\n{export_dir}")
            
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {e}")

    def save_export_images(self, export_dir):
        """保存导出图像"""
        if not self.current_results:
            return

        fs = self.fs_var.get()
        channel = self.current_results["channel"]
        original = self.current_results["original"]
        denoised = self.current_results["denoised"]
        time_axis = build_time_axis(len(original), fs)
        
        # 保存原始信号图
        original_fig = Figure(figsize=(10, 6), dpi=150)
        ax = original_fig.add_subplot(111)
        ax.plot(time_axis, original, linewidth=1, color='blue')
        ax.set_title(f'Original {channel.upper()} Signal')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        original_fig.savefig(os.path.join(export_dir, f"original_{channel}.png"), 
                           dpi=150, bbox_inches='tight')
        
        # 保存降噪信号图
        denoised_fig = Figure(figsize=(10, 6), dpi=150)
        ax = denoised_fig.add_subplot(111)
        ax.plot(time_axis, denoised, linewidth=1, color='red')
        ax.set_title(f'Denoised {channel.upper()} Signal')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, alpha=0.3)
        denoised_fig.savefig(os.path.join(export_dir, f"denoised_{channel}.png"), 
                           dpi=150, bbox_inches='tight')
        
        # 保存对比图
        comparison_fig = Figure(figsize=(10, 6), dpi=150)
        ax = comparison_fig.add_subplot(111)
        ax.plot(time_axis, original, linewidth=1, alpha=0.6, color='blue', label='Original')
        ax.plot(time_axis, denoised, linewidth=1.5, color='red', label='Denoised')
        ax.set_title(f'{channel.upper()} Signal Comparison')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        comparison_fig.savefig(os.path.join(export_dir, f"comparison_{channel}.png"), 
                             dpi=150, bbox_inches='tight')

    def update_config_from_gui(self):
        """从GUI更新配置"""
        self.config.update({
            "fs": self.fs_var.get(),
            "channel": self.channel_var.get(),
            "method": self.method_var.get(),
            "wavelet": self.wavelet_var.get(),
            "decomposition_level": self.level_var.get(),
            "threshold": {
                "strategy": self.strategy_var.get(),
                "scale": 1.0,
                "manual_value": None,
                "threshold_mode": "soft",
                "q_value": self.q_value_var.get(),
                "noise_estimate": self.noise_estimate_var.get()
            },
            "ppg_denoise": self.ppg_denoise_var.get(),
            "imu_denoise": self.imu_denoise_var.get()
        })

    def save_config(self):
        """保存配置"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.update_config_from_gui()
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("成功", f"配置已保存到: {filename}")

    def load_config(self):
        """加载配置"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.config = config
                self.update_gui_from_config()
                messagebox.showinfo("成功", f"配置已从 {filename} 加载")
            except Exception as e:
                messagebox.showerror("错误", f"无法加载配置文件: {e}")

    def update_gui_from_config(self):
        """从配置更新GUI"""
        self.fs_var.set(self.config.get("fs", 100.0))
        self.channel_var.set(self.config.get("channel", "green"))
        self.method_var.set(self.config.get("method", "bayes"))
        self.wavelet_var.set(self.config.get("wavelet", "sym8"))
        self.level_var.set(self.config.get("decomposition_level", 7))
        self.strategy_var.set(self.config.get("threshold", {}).get("strategy", "bayes"))
        self.q_value_var.set(self.config.get("threshold", {}).get("q_value", 0.05))
        self.noise_estimate_var.set(self.config.get("threshold", {}).get("noise_estimate", "level_independent"))
        self.ppg_denoise_var.set(self.config.get("ppg_denoise", True))
        self.imu_denoise_var.set(self.config.get("imu_denoise", True))


def main():
    """主函数"""
    root = tk.Tk()
    app = EnhancedDenoiseGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
