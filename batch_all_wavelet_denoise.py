# -*- coding: utf-8 -*-
"""
批量小波降噪处理脚本 - 一次性处理所有数据目录

这个脚本会自动发现项目中的所有 xxx_data 目录，
并对每个目录执行小波降噪处理。

用法示例：
    python batch_all_wavelet_denoise.py
    python batch_all_wavelet_denoise.py --config my_config.json
    python batch_all_wavelet_denoise.py --dirs hyx_data lhr_data lj_data
"""

import os
import sys
import glob
import json
import argparse
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import subprocess
import threading
from pathlib import Path

class BatchDenoiseGUI:
    """批量降噪处理的图形化界面"""

    def __init__(self, parent):
        self.parent = parent
        self.config = self.get_default_config()
        self.selected_dirs = []
        self.result = None

        self.setup_gui()

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
        self.parent.title("批量小波降噪处理")
        self.parent.geometry("700x600")

        # 创建主框架
        main_frame = tk.Frame(self.parent, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 标题
        title_label = tk.Label(main_frame, text="批量小波降噪处理", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))

        # 数据目录选择
        tk.Label(main_frame, text="选择要处理的数据目录:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(0, 10))

        # 自动发现的目录
        self.auto_dirs_frame = tk.Frame(main_frame)
        self.auto_dirs_frame.pack(fill=tk.X, pady=(0, 10))

        self.dir_vars = {}
        self.scan_directories()

        # 手动选择按钮
        tk.Button(main_frame, text="重新扫描目录", command=self.scan_directories).pack(pady=(5, 10))

        # 基本参数
        tk.Label(main_frame, text="降噪参数:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(10, 10))

        # 采样率
        param_frame = tk.Frame(main_frame)
        param_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(param_frame, text="采样率 (Hz):").grid(row=0, column=0, sticky=tk.W)
        self.fs_var = tk.DoubleVar(value=self.config["fs"])
        tk.Entry(param_frame, textvariable=self.fs_var, width=10).grid(row=0, column=1, sticky=tk.W)

        # PPG通道
        tk.Label(param_frame, text="PPG通道:").grid(row=0, column=2, sticky=tk.W, padx=(20, 0))
        self.channel_var = tk.StringVar(value=self.config["channel"])
        channel_combo = ttk.Combobox(param_frame, textvariable=self.channel_var,
                                   values=["green", "ir", "red"], state="readonly", width=8)
        channel_combo.grid(row=0, column=3, sticky=tk.W)

        # 降噪方法
        tk.Label(param_frame, text="降噪方法:").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.method_var = tk.StringVar(value=self.config.get("method", "wavelet"))
        method_combo = ttk.Combobox(param_frame, textvariable=self.method_var,
                                   values=["wavelet", "bayes"], state="readonly", width=8)
        method_combo.grid(row=1, column=1, sticky=tk.W, pady=(5, 0))

        # 小波类型
        tk.Label(param_frame, text="小波类型:").grid(row=1, column=2, sticky=tk.W, pady=(5, 0))
        self.wavelet_var = tk.StringVar(value=self.config["wavelet"])
        wavelet_combo = ttk.Combobox(param_frame, textvariable=self.wavelet_var,
                                   values=["db1", "db2", "db3", "db4", "db5", "db6", "db8",
                                          "sym2", "sym3", "sym4", "sym5", "sym6", "sym8",
                                          "coif1", "coif2", "coif3", "coif4", "coif5"], state="readonly", width=8)
        wavelet_combo.grid(row=1, column=3, sticky=tk.W, pady=(5, 0))

        # 分解层数
        tk.Label(param_frame, text="分解层数:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.level_var = tk.IntVar(value=self.config["decomposition_level"])
        tk.Spinbox(param_frame, from_=1, to=10, textvariable=self.level_var, width=8).grid(row=2, column=1, sticky=tk.W, pady=(5, 0))

        # 阈值策略
        tk.Label(param_frame, text="阈值策略:").grid(row=2, column=2, sticky=tk.W, pady=(5, 0))
        self.strategy_var = tk.StringVar(value=self.config["threshold"]["strategy"])
        strategy_combo = ttk.Combobox(param_frame, textvariable=self.strategy_var,
                                    values=["universal", "bayes", "manual"], state="readonly", width=8)
        strategy_combo.grid(row=2, column=3, sticky=tk.W, pady=(5, 0))

        # Q值参数
        tk.Label(param_frame, text="Q-Value:").grid(row=3, column=0, sticky=tk.W, pady=(5, 0))
        self.q_value_var = tk.DoubleVar(value=self.config["threshold"].get("q_value", 0.05))
        tk.Entry(param_frame, textvariable=self.q_value_var, width=10).grid(row=3, column=1, sticky=tk.W, pady=(5, 0))

        # 噪声估计方法
        tk.Label(param_frame, text="噪声估计:").grid(row=3, column=2, sticky=tk.W, pady=(5, 0))
        self.noise_estimate_var = tk.StringVar(value=self.config["threshold"].get("noise_estimate", "level_independent"))
        noise_combo = ttk.Combobox(param_frame, textvariable=self.noise_estimate_var,
                                  values=["level_independent", "level_dependent"], state="readonly", width=8)
        noise_combo.grid(row=3, column=3, sticky=tk.W, pady=(5, 0))

        # 降噪选项
        options_frame = tk.Frame(main_frame)
        options_frame.pack(fill=tk.X, pady=(10, 0))

        self.ppg_denoise_var = tk.BooleanVar(value=self.config["ppg_denoise"])
        tk.Checkbutton(options_frame, text="对PPG信号进行小波降噪", variable=self.ppg_denoise_var).pack(anchor=tk.W)

        self.imu_denoise_var = tk.BooleanVar(value=self.config["imu_denoise"])
        tk.Checkbutton(options_frame, text="对IMU信号进行小波降噪", variable=self.imu_denoise_var).pack(anchor=tk.W)

        # 按钮
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=(20, 0))

        tk.Button(button_frame, text="保存配置", command=self.save_config).pack(side=tk.LEFT, padx=(0, 10))
        tk.Button(button_frame, text="加载配置", command=self.load_config).pack(side=tk.LEFT, padx=(0, 10))
        tk.Button(button_frame, text="开始处理", command=self.confirm).pack(side=tk.LEFT, padx=(0, 10))
        tk.Button(button_frame, text="取消", command=self.cancel).pack(side=tk.LEFT)

        # 进度显示
        self.progress_var = tk.StringVar(value="就绪")
        progress_label = tk.Label(main_frame, textvariable=self.progress_var, font=("Arial", 10))
        progress_label.pack(pady=(10, 0))

    def scan_directories(self):
        """扫描数据目录"""
        # 清除现有的复选框
        for widget in self.auto_dirs_frame.winfo_children():
            widget.destroy()

        self.dir_vars.clear()

        # 查找所有 xxx_data 目录
        data_dirs = []
        for item in os.listdir('.'):
            if os.path.isdir(item) and item.endswith('_data'):
                data_dirs.append(item)

        if not data_dirs:
            tk.Label(self.auto_dirs_frame, text="未发现 xxx_data 格式的目录").pack(anchor=tk.W)
            return

        tk.Label(self.auto_dirs_frame, text="发现的数据目录:").pack(anchor=tk.W)

        for dir_name in sorted(data_dirs):
            var = tk.BooleanVar(value=True)  # 默认选中
            self.dir_vars[dir_name] = var
            cb = tk.Checkbutton(self.auto_dirs_frame, text=dir_name, variable=var)
            cb.pack(anchor=tk.W)

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
        selected_dirs = [name for name, var in self.dir_vars.items() if var.get()]

        config = {
            "fs": self.fs_var.get(),
            "channel": self.channel_var.get(),
            "method": self.method_var.get(),
            "wavelet": self.wavelet_var.get(),
            "decomposition_level": self.level_var.get(),
            "mode": "symmetric",
            "threshold": {
                "strategy": self.strategy_var.get(),
                "scale": 1.0,
                "manual_value": None,
                "threshold_mode": "soft",
                "q_value": self.q_value_var.get(),
                "noise_estimate": self.noise_estimate_var.get()
            },
            "plotting": {
                "figsize": [12, 6]
            },
            "ppg_denoise": self.ppg_denoise_var.get(),
            "imu_denoise": self.imu_denoise_var.get(),
            "selected_directories": selected_dirs
        }
        return config

    def set_config_to_gui(self, config):
        """将配置设置到GUI"""
        self.fs_var.set(config.get("fs", 100.0))
        self.channel_var.set(config.get("channel", "green"))
        self.method_var.set(config.get("method", "wavelet"))
        self.wavelet_var.set(config.get("wavelet", "db6"))
        self.level_var.set(config.get("decomposition_level", 4))
        self.strategy_var.set(config.get("threshold", {}).get("strategy", "universal"))
        self.q_value_var.set(config.get("threshold", {}).get("q_value", 0.05))
        self.noise_estimate_var.set(config.get("threshold", {}).get("noise_estimate", "level_independent"))
        self.ppg_denoise_var.set(config.get("ppg_denoise", True))
        self.imu_denoise_var.set(config.get("imu_denoise", True))

        # 设置目录选择
        selected_dirs = config.get("selected_directories", [])
        for dir_name, var in self.dir_vars.items():
            var.set(dir_name in selected_dirs)

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
    app = BatchDenoiseGUI(root)
    root.mainloop()

    if app.result == "confirm":
        return app.config
    else:
        return None


def find_data_directories():
    """自动发现所有数据目录"""
    data_dirs = []
    for item in os.listdir('.'):
        if os.path.isdir(item) and item.endswith('_data'):
            # 检查目录是否包含txt文件
            txt_files = glob.glob(os.path.join(item, "*.txt"))
            if txt_files:
                data_dirs.append(item)

    return sorted(data_dirs)


def run_denoise_on_directory(data_dir, config, output_base_dir):
    """对单个目录执行降噪处理"""
    print(f"开始处理目录: {data_dir}")

    # 构建命令
    cmd = [
        sys.executable,
        "wavelet_denoise_batch.py",
        "--dir", data_dir,
        "--config", "-"  # 从stdin读取配置
    ]

    # 准备配置数据
    process_config = {
        "fs": config["fs"],
        "channel": config["channel"],
        "method": config.get("method", "wavelet"),
        "wavelet": config["wavelet"],
        "decomposition_level": config["decomposition_level"],
        "mode": config["mode"],
        "threshold": config["threshold"],
        "plotting": config["plotting"],
        "ppg_denoise": config["ppg_denoise"],
        "imu_denoise": config["imu_denoise"]
    }

    try:
        # 启动子进程
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8'
        )

        # 通过stdin传递配置
        config_json = json.dumps(process_config, ensure_ascii=False)
        stdout, stderr = process.communicate(input=config_json)

        if process.returncode == 0:
            print(f"✓ 成功处理目录: {data_dir}")
            return True, stdout, None
        else:
            print(f"✗ 处理失败: {data_dir}")
            print(f"错误信息: {stderr}")
            return False, stdout, stderr

    except Exception as e:
        return False, None, str(e)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量小波降噪处理 - 一次性处理所有数据目录",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--config", type=str, help="配置文件路径（可选）")
    parser.add_argument("--dirs", nargs="*", help="指定要处理的数据目录")
    parser.add_argument("--output-dir", type=str, help="输出基础目录")

    args = parser.parse_args()

    print("=== 批量小波降噪处理工具 ===")

    # 获取配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"使用配置文件: {args.config}")
    else:
        print("启动图形化配置界面...")
        config = get_config_from_gui()
        if config is None:
            print("用户取消操作")
            return

    # 确定要处理的数据目录
    if args.dirs:
        data_dirs = args.dirs
        print(f"指定处理目录: {', '.join(data_dirs)}")
    else:
        data_dirs = find_data_directories()
        if not data_dirs:
            print("未发现任何数据目录 (xxx_data)")
            return
        print(f"自动发现数据目录: {', '.join(data_dirs)}")

    # 验证目录存在且包含文件
    valid_dirs = []
    for data_dir in data_dirs:
        if os.path.isdir(data_dir):
            txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
            if txt_files:
                valid_dirs.append(data_dir)
            else:
                print(f"警告: 目录 {data_dir} 不包含 .txt 文件，跳过")
        else:
            print(f"警告: 目录 {data_dir} 不存在，跳过")

    if not valid_dirs:
        print("没有有效的目录可处理")
        return

    # 确定输出目录
    output_base_dir = args.output_dir or "batch_denoised_results"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    main_output_dir = os.path.join(output_base_dir, f"batch_all_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)

    print(f"输出目录: {main_output_dir}")
    print(f"需要处理 {len(valid_dirs)} 个目录")

    # 保存批量配置
    batch_config_path = os.path.join(main_output_dir, "batch_config.json")
    with open(batch_config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # 处理每个目录
    results = []
    successful = 0
    failed = 0

    for i, data_dir in enumerate(valid_dirs, 1):
        print(f"\n[{i}/{len(valid_dirs)}] 处理: {data_dir}")

        success, stdout, error = run_denoise_on_directory(data_dir, config, main_output_dir)

        results.append({
            "directory": data_dir,
            "success": success,
            "stdout": stdout,
            "error": error
        })

        if success:
            successful += 1
        else:
            failed += 1

    # 生成汇总报告
    summary_path = os.path.join(main_output_dir, "batch_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("批量小波降噪处理报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"配置文件: {batch_config_path}\n")
        f.write(f"处理目录数: {len(valid_dirs)}\n")
        f.write(f"成功: {successful}, 失败: {failed}\n\n")

        f.write("详细结果:\n")
        f.write("-" * 30 + "\n")
        for result in results:
            status = "✓" if result["success"] else "✗"
            f.write(f"{status} {result['directory']}\n")
            if result["error"]:
                f.write(f"   错误: {result['error']}\n")

        f.write(f"\n输出目录: {main_output_dir}\n")

    print("\n=== 处理完成 ===")
    print(f"输出目录: {main_output_dir}")
    print(f"汇总报告: {summary_path}")
    print(f"成功: {successful}, 失败: {failed}")

    if failed == 0:
        print("🎉 所有目录处理成功！")
    else:
        print(f"⚠️  有 {failed} 个目录处理失败，请查看汇总报告了解详情")


if __name__ == "__main__":
    main()
