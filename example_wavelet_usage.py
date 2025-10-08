#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
小波降噪使用示例

这个脚本展示了如何使用小波降噪批处理工具处理实际数据。
"""

import os
import sys
import json

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")

    # 使用默认配置（会启动GUI）
    print("python wavelet_denoise_batch.py --dir ./hyx_data")

    # 或使用便捷脚本
    print("python run_wavelet_denoise.py --dir ./hyx_data")

def example_with_config():
    """使用配置文件示例"""
    print("\n=== 使用配置文件示例 ===")

    # 创建一个示例配置
    config = {
        "fs": 100,
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
        "ppg_denoise": True,
        "imu_denoise": True
    }

    # 保存配置
    config_path = "example_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"python wavelet_denoise_batch.py --dir ./hyx_data --config {config_path}")

    # 清理示例文件
    if os.path.exists(config_path):
        os.remove(config_path)

def example_advanced_config():
    """高级配置示例"""
    print("\n=== 高级配置示例 ===")

    # 自定义阈值配置
    config = {
        "fs": 100,
        "channel": "green",
        "wavelet": "sym8",
        "decomposition_level": 5,
        "mode": "symmetric",
        "threshold": {
            "strategy": "manual",
            "scale": 1.0,
            "manual_value": 0.1,
            "threshold_mode": "soft"
        },
        "plotting": {
            "figsize": [16, 8]
        },
        "ppg_denoise": True,
        "imu_denoise": False  # 只对PPG降噪
    }

    # 保存配置
    config_path = "advanced_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"python wavelet_denoise_batch.py --dir ./lhr_data --config {config_path}")

    # 清理示例文件
    if os.path.exists(config_path):
        os.remove(config_path)

def show_output_structure():
    """显示输出结构"""
    print("\n=== 输出目录结构示例 ===")
    print("denoised_hyx_20250926-1430/")
    print("├── denoising_summary.csv          # 详细统计报告")
    print("├── wavelet_config.json            # 使用的配置参数")
    print("├── 喉咙-吞咽6次间隔10秒_denoising_comparison.png")
    print("├── 喉咙-咀嚼5下共6次间隔10秒_denoising_comparison.png")
    print("└── ...                            # 其他文件的对比图像")

def show_csv_content_example():
    """显示CSV内容示例"""
    print("\n=== CSV报告内容示例 ===")
    print("file,channel,N_samples,fs,duration_sec,wavelength,decomposition_level,mode,...")
    print("喉咙-吞咽6次间隔10秒,green,1000,100,10.0,db6,4,symmetric,...")
    print("SNR_before,SNR_after,SNR_improvement,acc_energy_before,acc_energy_after,...")
    print("15.2,18.7,3.5,2.34,2.12,...")
    print("status,error_message,png_path")
    print("OK,,denoised_hyx_20250926-1430/喉咙-吞咽6次间隔10秒_denoising_comparison.png")

def main():
    """主函数"""
    print("小波降噪批处理工具使用示例")
    print("=" * 50)

    example_basic_usage()
    example_with_config()
    example_advanced_config()
    show_output_structure()
    show_csv_content_example()

    print("\n" + "=" * 50)
    print("提示：")
    print("1. 确保输入目录包含 .txt 文件")
    print("2. 建议目录命名为 xxx_data 格式")
    print("3. 首次使用建议用默认配置（GUI界面）")
    print("4. 调整参数后可以保存为配置文件")
    print("5. 查看 WAVELET_DENOISE_README.md 获取详细说明")

if __name__ == "__main__":
    main()

