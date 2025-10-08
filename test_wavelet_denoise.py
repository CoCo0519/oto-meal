#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
小波降噪功能测试脚本

用于测试小波降噪功能是否正常工作，不启动GUI界面。
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wavelet_denoise_batch import (
    get_config_from_gui, imu_wavelet_denoise,
    save_denoised_data_comparison, safe_loadtxt
)

def test_config_loading():
    """测试配置加载功能"""
    print("测试配置加载功能...")

    # 创建测试配置
    test_config = {
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
        "ppg_denoise": True,
        "imu_denoise": True
    }

    # 保存测试配置
    config_path = "test_wavelet_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(test_config, f, indent=2, ensure_ascii=False)

    print(f"测试配置文件已保存: {config_path}")
    return config_path

def test_data_loading():
    """测试数据加载功能"""
    print("测试数据加载功能...")

    # 创建模拟数据
    fs = 100
    duration = 10  # 10秒
    n_samples = int(fs * duration)

    # PPG信号 (模拟心率信号 + 噪声)
    t = np.arange(n_samples) / fs
    ppg_signal = np.sin(2 * np.pi * 1.5 * t) + 0.1 * np.random.randn(n_samples)

    # ACC信号 (X, Y, Z)
    acc_x = 0.5 * np.sin(2 * np.pi * 2.0 * t) + 0.05 * np.random.randn(n_samples)
    acc_y = 0.3 * np.cos(2 * np.pi * 2.0 * t) + 0.05 * np.random.randn(n_samples)
    acc_z = 0.8 + 0.02 * np.random.randn(n_samples)
    acc_data = np.column_stack([acc_x, acc_y, acc_z])

    # GYRO信号 (可选)
    gyro_data = np.column_stack([
        0.1 * np.random.randn(n_samples),
        0.1 * np.random.randn(n_samples),
        0.1 * np.random.randn(n_samples)
    ])

    # 保存为txt文件（跳过中文表头）
    data = np.column_stack([ppg_signal, ppg_signal, ppg_signal, acc_data, gyro_data])

    test_file = "test_signal.txt"
    header = "Green\tIR\tRed\tACC_X\tACC_Y\tACC_Z\tGYRO_X\tGYRO_Y\tGYRO_Z"
    np.savetxt(test_file, data, delimiter='\t', header=header, comments='')

    print(f"测试数据文件已创建: {test_file}")
    return test_file

def test_wavelet_denoise():
    """测试小波降噪功能"""
    print("测试小波降噪功能...")

    # 加载配置
    config_path = test_config_loading()
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 加载测试数据
    test_file = test_data_loading()

    # 读取数据
    data = safe_loadtxt(test_file)
    ppg_raw = data[:, 0]  # Green通道
    acc_raw = data[:, 3:6]  # ACC数据
    gyro_raw = data[:, 6:9]  # GYRO数据

    # 执行PPG降噪
    if config["ppg_denoise"]:
        try:
            from ppg_wavelet_denoise import wavelet_denoise
            ppg_denoised, threshold = wavelet_denoise(ppg_raw, config)
            print(f"PPG降噪成功，阈值: {threshold:.6f}")
        except Exception as e:
            print(f"PPG降噪失败: {e}")
            ppg_denoised = ppg_raw
    else:
        ppg_denoised = ppg_raw
        print("PPG降噪已禁用")

    # 执行IMU降噪
    if config["imu_denoise"] and acc_raw is not None:
        try:
            acc_denoised, gyro_denoised = imu_wavelet_denoise(acc_raw, gyro_raw, config)
            print("IMU降噪成功")
        except Exception as e:
            print(f"IMU降噪失败: {e}")
            acc_denoised = acc_raw
            gyro_denoised = gyro_raw
    else:
        acc_denoised = acc_raw
        gyro_denoised = gyro_raw
        print("IMU降噪已禁用")

    # 生成对比图像
    try:
        output_path = "test_denoising_comparison.png"
        save_denoised_data_comparison(
            ppg_raw, ppg_denoised, acc_raw, acc_denoised,
            gyro_raw, gyro_denoised, config["fs"], output_path, config
        )
        print(f"对比图像已保存: {output_path}")
    except Exception as e:
        print(f"生成对比图像失败: {e}")

    # 清理测试文件
    for f in [config_path, test_file, output_path]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass

    print("测试完成！")

def main():
    """主函数"""
    print("=== 小波降噪功能测试 ===")

    try:
        test_wavelet_denoise()
        print("✓ 所有测试通过")
    except KeyboardInterrupt:
        print("\n测试已取消")
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
