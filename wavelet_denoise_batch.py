# -*- coding: utf-8 -*-
"""
基于 ReadDirectory.py 逻辑的小波降噪批处理脚本：
- 对所有符合要求的路径中的 PPG 与 IMU 文件分别进行小波降噪
- 生成结果命名为 denoised_xxx_data/xxx
- 使用图形化配置界面选择小波参数
- 生成降噪后的对比图像

用法示例：
    python wavelet_denoise_batch.py --dir ./hyx_data
"""

import os
import sys
import glob
import json
import argparse
from datetime import datetime
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 后端设为非交互，直接保存图片
import matplotlib.pyplot as plt

try:
    import pywt
except ImportError:
    print("需要安装 PyWavelets: pip install PyWavelets")
    sys.exit(1)

from ppg_wavelet_denoise import (
    load_ppg_txt, wavelet_denoise, build_time_axis,
    save_single_plot, save_comparison_plot
)


class WaveletConfigGUI:
    """图形化配置界面"""

    def __init__(self, parent):
        self.parent = parent
        self.config = self.get_default_config()
        self.result = None

        self.setup_gui()

    def get_default_config(self):
        """获取默认配置"""
        return {
            "fs": 100.0,
            "channel": "green",
            "wavelet": "db6",
            "decomposition_level": 4,
            "mode": "symmetric",
            "threshold": {
                "strategy": "universal",
                "scale": 1.0,
                "manual_value": None,
                "threshold_mode": "soft"
            },
            "plotting": {
                "figsize": [12, 6]
            },
            "imu_denoise": True,
            "ppg_denoise": True
        }

    def setup_gui(self):
        """设置GUI界面"""
        self.parent.title("小波降噪配置")
        self.parent.geometry("600x500")

        # 创建主框架
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 基本参数
        ttk.Label(main_frame, text="基本参数", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        # 采样率
        ttk.Label(main_frame, text="采样率 (Hz):").grid(row=1, column=0, sticky=tk.W)
        self.fs_var = tk.DoubleVar(value=self.config["fs"])
        ttk.Entry(main_frame, textvariable=self.fs_var, width=10).grid(row=1, column=1, sticky=tk.W)

        # PPG通道
        ttk.Label(main_frame, text="PPG通道:").grid(row=2, column=0, sticky=tk.W)
        self.channel_var = tk.StringVar(value=self.config["channel"])
        channel_combo = ttk.Combobox(main_frame, textvariable=self.channel_var,
                                   values=["green", "ir", "red"], state="readonly")
        channel_combo.grid(row=2, column=1, sticky=tk.W)

        # 小波参数
        ttk.Label(main_frame, text="小波参数", font=("Arial", 12, "bold")).grid(row=3, column=0, sticky=tk.W, pady=(20, 10))

        # 小波类型
        ttk.Label(main_frame, text="小波类型:").grid(row=4, column=0, sticky=tk.W)
        self.wavelet_var = tk.StringVar(value=self.config["wavelet"])
        wavelet_combo = ttk.Combobox(main_frame, textvariable=self.wavelet_var,
                                   values=["db1", "db2", "db3", "db4", "db5", "db6", "db8",
                                          "sym2", "sym3", "sym4", "sym5", "sym6", "sym8",
                                          "coif1", "coif2", "coif3", "coif4", "coif5"], state="readonly")
        wavelet_combo.grid(row=4, column=1, sticky=tk.W)

        # 分解层数
        ttk.Label(main_frame, text="分解层数:").grid(row=5, column=0, sticky=tk.W)
        self.level_var = tk.IntVar(value=self.config["decomposition_level"])
        ttk.Spinbox(main_frame, from_=1, to=10, textvariable=self.level_var, width=8).grid(row=5, column=1, sticky=tk.W)

        # 边界模式
        ttk.Label(main_frame, text="边界模式:").grid(row=6, column=0, sticky=tk.W)
        self.mode_var = tk.StringVar(value=self.config["mode"])
        mode_combo = ttk.Combobox(main_frame, textvariable=self.mode_var,
                                values=["zero", "constant", "symmetric", "periodic", "smooth", "periodization"], state="readonly")
        mode_combo.grid(row=6, column=1, sticky=tk.W)

        # 阈值参数
        ttk.Label(main_frame, text="阈值参数", font=("Arial", 12, "bold")).grid(row=7, column=0, sticky=tk.W, pady=(20, 10))

        # 阈值策略
        ttk.Label(main_frame, text="阈值策略:").grid(row=8, column=0, sticky=tk.W)
        self.strategy_var = tk.StringVar(value=self.config["threshold"]["strategy"])
        strategy_combo = ttk.Combobox(main_frame, textvariable=self.strategy_var,
                                    values=["universal", "manual"], state="readonly")
        strategy_combo.grid(row=8, column=1, sticky=tk.W)

        # 阈值缩放
        ttk.Label(main_frame, text="阈值缩放:").grid(row=9, column=0, sticky=tk.W)
        self.scale_var = tk.DoubleVar(value=self.config["threshold"]["scale"])
        ttk.Entry(main_frame, textvariable=self.scale_var, width=10).grid(row=9, column=1, sticky=tk.W)

        # 手动阈值
        ttk.Label(main_frame, text="手动阈值:").grid(row=10, column=0, sticky=tk.W)
        self.manual_var = tk.DoubleVar(value=self.config["threshold"]["manual_value"] or 0.0)
        self.manual_entry = ttk.Entry(main_frame, textvariable=self.manual_var, width=10, state="disabled")
        self.manual_entry.grid(row=10, column=1, sticky=tk.W)

        # 阈值模式
        ttk.Label(main_frame, text="阈值模式:").grid(row=11, column=0, sticky=tk.W)
        self.threshold_mode_var = tk.StringVar(value=self.config["threshold"]["threshold_mode"])
        threshold_combo = ttk.Combobox(main_frame, textvariable=self.threshold_mode_var,
                                     values=["soft", "hard"], state="readonly")
        threshold_combo.grid(row=11, column=1, sticky=tk.W)

        # 降噪选项
        ttk.Label(main_frame, text="降噪选项", font=("Arial", 12, "bold")).grid(row=12, column=0, sticky=tk.W, pady=(20, 10))

        self.ppg_denoise_var = tk.BooleanVar(value=self.config["ppg_denoise"])
        ttk.Checkbutton(main_frame, text="对PPG信号进行小波降噪", variable=self.ppg_denoise_var).grid(row=13, column=0, columnspan=2, sticky=tk.W)

        self.imu_denoise_var = tk.BooleanVar(value=self.config["imu_denoise"])
        ttk.Checkbutton(main_frame, text="对IMU信号进行小波降噪", variable=self.imu_denoise_var).grid(row=14, column=0, columnspan=2, sticky=tk.W)

        # 按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=15, column=0, columnspan=2, pady=(20, 0))

        ttk.Button(button_frame, text="保存配置", command=self.save_config).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(button_frame, text="加载配置", command=self.load_config).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(button_frame, text="确定", command=self.confirm).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(button_frame, text="取消", command=self.cancel).grid(row=0, column=3)

        # 绑定事件
        self.strategy_var.trace("w", self.update_manual_threshold_state)

    def update_manual_threshold_state(self, *args):
        """更新手动阈值输入框状态"""
        if self.strategy_var.get() == "manual":
            self.manual_entry.config(state="normal")
        else:
            self.manual_entry.config(state="disabled")

    def save_config(self):
        """保存配置到文件"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            config = self.get_config_from_gui()
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("保存成功", f"配置已保存到：{filename}")

    def load_config(self):
        """从文件加载配置"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                self.set_config_to_gui(config)
                messagebox.showinfo("加载成功", f"配置已从 {filename} 加载")
            except Exception as e:
                messagebox.showerror("加载失败", f"无法加载配置文件：{e}")

    def get_config_from_gui(self):
        """从GUI获取配置"""
        config = {
            "fs": self.fs_var.get(),
            "channel": self.channel_var.get(),
            "wavelet": self.wavelet_var.get(),
            "decomposition_level": self.level_var.get(),
            "mode": self.mode_var.get(),
            "threshold": {
                "strategy": self.strategy_var.get(),
                "scale": self.scale_var.get(),
                "manual_value": self.manual_var.get() if self.strategy_var.get() == "manual" else None,
                "threshold_mode": self.threshold_mode_var.get()
            },
            "plotting": {
                "figsize": [12, 6]
            },
            "ppg_denoise": self.ppg_denoise_var.get(),
            "imu_denoise": self.imu_denoise_var.get()
        }
        return config

    def set_config_to_gui(self, config):
        """将配置设置到GUI"""
        self.fs_var.set(config.get("fs", 100.0))
        self.channel_var.set(config.get("channel", "green"))
        self.wavelet_var.set(config.get("wavelet", "db6"))
        self.level_var.set(config.get("decomposition_level", 4))
        self.mode_var.set(config.get("mode", "symmetric"))
        self.strategy_var.set(config.get("threshold", {}).get("strategy", "universal"))
        self.scale_var.set(config.get("threshold", {}).get("scale", 1.0))
        manual_value = config.get("threshold", {}).get("manual_value")
        if manual_value is not None:
            self.manual_var.set(manual_value)
        self.threshold_mode_var.set(config.get("threshold", {}).get("threshold_mode", "soft"))
        self.ppg_denoise_var.set(config.get("ppg_denoise", True))
        self.imu_denoise_var.set(config.get("imu_denoise", True))

    def confirm(self):
        """确认配置"""
        self.config = self.get_config_from_gui()
        self.result = "confirm"
        self.parent.quit()

    def cancel(self):
        """取消配置"""
        self.result = "cancel"
        self.parent.quit()


def get_config_from_gui():
    """启动GUI并获取配置"""
    root = tk.Tk()
    app = WaveletConfigGUI(root)
    root.mainloop()

    if app.result == "confirm":
        return app.config
    else:
        return None


def safe_loadtxt(path):
    """优先按 utf-8 读取；若失败则回退 gbk。首行表头需在调用方 skiprows=1。"""
    try:
        return np.loadtxt(path, skiprows=1, encoding="utf-8")
    except TypeError:
        # 老版本 numpy 没 encoding 参数
        with open(path, encoding="utf-8") as f:
            return np.loadtxt(f, skiprows=1)
    except UnicodeDecodeError:
        # 尝试 GBK
        try:
            return np.loadtxt(path, skiprows=1, encoding="gbk")
        except TypeError:
            with open(path, encoding="gbk") as f:
                return np.loadtxt(f, skiprows=1)


def imu_wavelet_denoise(acc_data, gyro_data, config):
    """对IMU数据进行小波降噪"""
    denoised_acc = []
    denoised_gyro = []

    for i in range(3):  # X, Y, Z轴
        if config["imu_denoise"]:
            acc_denoised, _ = wavelet_denoise(acc_data[:, i], config)
            denoised_acc.append(acc_denoised)
        else:
            denoised_acc.append(acc_data[:, i])

    denoised_acc = np.column_stack(denoised_acc)

    if gyro_data is not None and config["imu_denoise"]:
        for i in range(3):  # X, Y, Z轴
            gyro_denoised, _ = wavelet_denoise(gyro_data[:, i], config)
            denoised_gyro.append(gyro_denoised)
        denoised_gyro = np.column_stack(denoised_gyro)
    else:
        denoised_gyro = gyro_data

    return denoised_acc, denoised_gyro


def save_denoised_data_comparison(ppg_raw, ppg_denoised, acc_raw, acc_denoised,
                                gyro_raw, gyro_denoised, fs, save_path, config):
    """保存降噪对比图像"""
    t = np.arange(len(ppg_raw)) / fs
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))

    # PPG对比
    axes[0, 0].plot(t, ppg_raw, label='Raw PPG')
    axes[0, 0].set_title('PPG Raw')
    axes[0, 0].legend(loc='upper right')

    if config["ppg_denoise"]:
        axes[0, 1].plot(t, ppg_denoised, label='Denoised PPG', color='red')
        axes[0, 1].set_title('PPG Denoised')
        axes[0, 1].legend(loc='upper right')

        # PPG残差
        ppg_residual = ppg_raw - ppg_denoised
        axes[1, 0].plot(t, ppg_residual, label='PPG Residual', color='green')
        axes[1, 0].set_title('PPG Residual')
        axes[1, 0].legend(loc='upper right')
    else:
        axes[0, 1].plot(t, ppg_raw, label='PPG (No Denoise)', color='gray')
        axes[0, 1].set_title('PPG (No Denoise)')
        axes[0, 1].legend(loc='upper right')

        axes[1, 0].text(0.5, 0.5, 'No PPG Denoising Applied',
                       ha='center', va='center', transform=axes[1, 0].transAxes)

    # ACC对比
    if acc_raw is not None:
        axes[1, 1].plot(t, acc_raw[:, 0], label='ACC X Raw', alpha=0.7)
        axes[1, 1].plot(t, acc_raw[:, 1], label='ACC Y Raw', alpha=0.7)
        axes[1, 1].plot(t, acc_raw[:, 2], label='ACC Z Raw', alpha=0.7)
        axes[1, 1].set_title('ACC Raw')
        axes[1, 1].legend(loc='upper right')

        axes[2, 0].plot(t, acc_denoised[:, 0], label='ACC X Denoised', alpha=0.7)
        axes[2, 0].plot(t, acc_denoised[:, 1], label='ACC Y Denoised', alpha=0.7)
        axes[2, 0].plot(t, acc_denoised[:, 2], label='ACC Z Denoised', alpha=0.7)
        axes[2, 0].set_title('ACC Denoised')
        axes[2, 0].legend(loc='upper right')

        # ACC残差
        if config["imu_denoise"]:
            acc_residual = acc_raw - acc_denoised
            axes[2, 1].plot(t, acc_residual[:, 0], label='ACC X Residual', alpha=0.7)
            axes[2, 1].plot(t, acc_residual[:, 1], label='ACC Y Residual', alpha=0.7)
            axes[2, 1].plot(t, acc_residual[:, 2], label='ACC Z Residual', alpha=0.7)
            axes[2, 1].set_title('ACC Residual')
            axes[2, 1].legend(loc='upper right')
        else:
            axes[2, 1].text(0.5, 0.5, 'No ACC Denoising Applied',
                           ha='center', va='center', transform=axes[2, 1].transAxes)

    # Gyro对比
    if gyro_raw is not None:
        axes[3, 0].plot(t, gyro_raw[:, 0], label='GYRO X Raw', alpha=0.7)
        axes[3, 0].plot(t, gyro_raw[:, 1], label='GYRO Y Raw', alpha=0.7)
        axes[3, 0].plot(t, gyro_raw[:, 2], label='GYRO Z Raw', alpha=0.7)
        axes[3, 0].set_title('GYRO Raw')
        axes[3, 0].legend(loc='upper right')
        axes[3, 0].set_xlabel('Time (s)')

        axes[3, 1].plot(t, gyro_denoised[:, 0], label='GYRO X Denoised', alpha=0.7)
        axes[3, 1].plot(t, gyro_denoised[:, 1], label='GYRO Y Denoised', alpha=0.7)
        axes[3, 1].plot(t, gyro_denoised[:, 2], label='GYRO Z Denoised', alpha=0.7)
        axes[3, 1].set_title('GYRO Denoised')
        axes[3, 1].legend(loc='upper right')
        axes[3, 1].set_xlabel('Time (s)')
    else:
        axes[3, 0].text(0.5, 0.5, 'No GYRO Data', ha='center', va='center', transform=axes[3, 0].transAxes)
        axes[3, 1].text(0.5, 0.5, 'No GYRO Data', ha='center', va='center', transform=axes[3, 1].transAxes)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def extract_name_from_dir(dir_path):
    """从形如 .../xxx_data 中提取 name=xxx；若不满足，退回使用目录名。"""
    base = os.path.basename(os.path.normpath(dir_path))
    if base.endswith("_data") and len(base) > 5:
        return base[:-5]
    return base


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dir", type=str, required=True, help="包含 .txt 的目录（建议形如 xxx_data）")
    ap.add_argument("--config", type=str, help="配置文件路径（可选）")
    args = ap.parse_args()

    # 获取配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        print("启动图形化配置界面...")
        config = get_config_from_gui()
        if config is None:
            print("用户取消操作")
            return

    in_dir = os.path.abspath(args.dir)
    if not os.path.isdir(in_dir):
        print(f"[ERROR] 输入目录不存在：{in_dir}")
        sys.exit(1)

    # 构造结果目录名：denoised_<name>_<YYYYMMDD-HHMM>，位于脚本所在目录
    name = extract_name_from_dir(in_dir)
    now_str = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"denoised_{name}_{now_str}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] 输出目录：{out_dir}")

    # 保存配置
    config_path = os.path.join(out_dir, "wavelet_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 收集所有 txt 文件
    txt_list = sorted(glob.glob(os.path.join(in_dir, "*.txt")))
    if not txt_list:
        print(f"[WARN] 目录下未发现 .txt 文件：{in_dir}")
        sys.exit(0)

    # 汇总 CSV
    csv_path = os.path.join(out_dir, "denoising_summary.csv")
    fieldnames = [
        "file", "channel", "N_samples", "fs", "duration_sec",
        "wavelet", "decomposition_level", "mode",
        "threshold_strategy", "threshold_scale", "threshold_value", "threshold_mode",
        "ppg_denoised", "imu_denoised",
        "ppg_snr_before", "ppg_snr_after", "ppg_snr_improvement",
        "acc_energy_before", "acc_energy_after", "acc_snr_improvement",
        "gyro_present", "status", "error_message",
        "png_path",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()

        ok, fail = 0, 0
        for txt_path in txt_list:
            base = os.path.splitext(os.path.basename(txt_path))[0]
            save_path = os.path.join(out_dir, f"{base}_denoising_comparison.png")
            row = {
                "file": base,
                "channel": config["channel"],
                "fs": float(config["fs"]),
                "wavelet": config["wavelet"],
                "decomposition_level": int(config["decomposition_level"]),
                "mode": config["mode"],
                "threshold_strategy": config["threshold"]["strategy"],
                "threshold_scale": float(config["threshold"]["scale"]),
                "threshold_mode": config["threshold"]["threshold_mode"],
                "ppg_denoised": bool(config["ppg_denoise"]),
                "imu_denoised": bool(config["imu_denoise"]),
                "png_path": save_path,
            }
            try:
                # 读取数据（跳过中文表头）
                data = safe_loadtxt(txt_path)
                N = data.shape[0]
                row["N_samples"] = int(N)
                row["duration_sec"] = float(N / config["fs"])

                # 列映射
                ppg_green = data[:, 0]
                ppg_ir    = data[:, 1]
                ppg_red   = data[:, 2]
                acc_raw   = data[:, 3:6]    # X, Y, Z
                gyro_raw  = data[:, 6:9] if data.shape[1] > 6 else None

                # 选择 PPG 通道
                if config["channel"] == "green":
                    ppg_raw = ppg_green
                elif config["channel"] == "ir":
                    ppg_raw = ppg_ir
                else:
                    ppg_raw = ppg_red

                # 计算原始SNR（如果需要降噪）
                if config["ppg_denoise"]:
                    ppg_noise_std = np.std(ppg_raw)
                    ppg_snr_before = 20 * np.log10(np.std(ppg_raw) / ppg_noise_std) if ppg_noise_std > 0 else 0
                else:
                    ppg_snr_before = 0

                # PPG小波降噪
                if config["ppg_denoise"]:
                    ppg_denoised, threshold = wavelet_denoise(ppg_raw, config)
                    ppg_noise_std_after = np.std(ppg_raw - ppg_denoised)
                    ppg_snr_after = 20 * np.log10(np.std(ppg_denoised) / ppg_noise_std_after) if ppg_noise_std_after > 0 else 0
                    ppg_snr_improvement = ppg_snr_after - ppg_snr_before
                else:
                    ppg_denoised = ppg_raw
                    ppg_snr_after = ppg_snr_before
                    ppg_snr_improvement = 0
                    threshold = 0

                row["threshold_value"] = float(threshold)
                row["ppg_snr_before"] = float(ppg_snr_before)
                row["ppg_snr_after"] = float(ppg_snr_after)
                row["ppg_snr_improvement"] = float(ppg_snr_improvement)

                # IMU小波降噪
                if config["imu_denoise"] and acc_raw is not None:
                    acc_denoised, gyro_denoised = imu_wavelet_denoise(acc_raw, gyro_raw, config)

                    # 计算ACC能量改善
                    acc_energy_before = np.mean(np.sqrt(np.sum(acc_raw**2, axis=1)))
                    acc_energy_after = np.mean(np.sqrt(np.sum(acc_denoised**2, axis=1)))
                    acc_snr_improvement = acc_energy_after - acc_energy_before if acc_energy_before > 0 else 0
                else:
                    acc_denoised = acc_raw
                    gyro_denoised = gyro_raw
                    acc_energy_before = np.mean(np.sqrt(np.sum(acc_raw**2, axis=1))) if acc_raw is not None else 0
                    acc_energy_after = acc_energy_before
                    acc_snr_improvement = 0

                row["acc_energy_before"] = float(acc_energy_before)
                row["acc_energy_after"] = float(acc_energy_after)
                row["acc_snr_improvement"] = float(acc_snr_improvement)
                row["gyro_present"] = gyro_raw is not None

                # 生成对比图像
                save_denoised_data_comparison(
                    ppg_raw, ppg_denoised, acc_raw, acc_denoised,
                    gyro_raw, gyro_denoised, config["fs"], save_path, config
                )

                row.update({
                    "status": "OK",
                    "error_message": "",
                })
                writer.writerow(row)
                ok += 1
                print(f"[OK] {base} -> {save_path}")

            except Exception as e:
                row.update({
                    "N_samples": None,
                    "duration_sec": None,
                    "threshold_value": None,
                    "ppg_snr_before": None,
                    "ppg_snr_after": None,
                    "ppg_snr_improvement": None,
                    "acc_energy_before": None,
                    "acc_energy_after": None,
                    "acc_snr_improvement": None,
                    "gyro_present": None,
                    "status": "FAIL",
                    "error_message": str(e),
                })
                writer.writerow(row)
                fail += 1
                print(f"[FAIL] {base}: {e}")

    print(f"\n[SUMMARY] 输出目录: {out_dir}")
    print(f"[SUMMARY] 汇总表: {csv_path}")
    print(f"[SUMMARY] 成功: {ok}, 失败: {fail}")
    print(f"[SUMMARY] 配置文件: {config_path}")


if __name__ == "__main__":
    main()

