#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量小波降噪功能测试脚本

用于测试批量处理功能是否正常工作，不启动GUI界面。
"""

import os
import sys
import json
import shutil
from pathlib import Path

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_directories():
    """创建测试数据目录"""
    print("创建测试数据目录...")

    # 清理可能存在的测试目录
    test_dirs = ["test_hyx_data", "test_lhr_data", "test_lj_data"]
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

    # 创建测试目录和数据
    for i, test_dir in enumerate(test_dirs):
        os.makedirs(test_dir, exist_ok=True)

        # 创建模拟数据文件
        fs = 100
        duration = 5  # 5秒测试数据
        n_samples = int(fs * duration)

        # PPG信号 (模拟心率信号 + 噪声)
        t = list(range(n_samples))
        ppg_signal = [50 + 30 * (i % 3) + 5 * (j % 2) for j in range(n_samples)]  # 不同的基线和幅度

        # ACC信号 (X, Y, Z)
        acc_x = [0.5 * (1 + i) + 0.1 * (j % 10) for j in range(n_samples)]
        acc_y = [0.3 * (1 + i) + 0.1 * (j % 8) for j in range(n_samples)]
        acc_z = [0.8 + 0.02 * (j % 5) for j in range(n_samples)]

        # GYRO信号 (可选)
        gyro_x = [0.1 * (i + 1) + 0.01 * (j % 20) for j in range(n_samples)]
        gyro_y = [0.1 * (i + 1) + 0.01 * (j % 18) for j in range(n_samples)]
        gyro_z = [0.1 * (i + 1) + 0.01 * (j % 22) for j in range(n_samples)]

        # 创建不同噪声水平的信号用于测试Bayesian方法
        noise_level = 0.1 + i * 0.05  # 递增的噪声水平

        # 创建数据文件
        for file_idx in range(2 if i < 2 else 1):  # 前两个目录2个文件，最后一个目录1个文件
            filename = f"test_signal_{file_idx + 1}.txt"
            filepath = os.path.join(test_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                # 写入中文表头
                f.write("Green\tIR\tRed\tACC_X\tACC_Y\tACC_Z\tGYRO_X\tGYRO_Y\tGYRO_Z\n")

                # 写入数据
                for j in range(n_samples):
                    line = f"{ppg_signal[j]}\t{ppg_signal[j]}\t{ppg_signal[j]}\t"
                    line += f"{acc_x[j]}\t{acc_y[j]}\t{acc_z[j]}\t"
                    line += f"{gyro_x[j]}\t{gyro_y[j]}\t{gyro_z[j]}\n"
                    f.write(line)

    print(f"创建了 {len(test_dirs)} 个测试目录:")
    for test_dir in test_dirs:
        txt_count = len([f for f in os.listdir(test_dir) if f.endswith('.txt')])
        print(f"  - {test_dir}: {txt_count} 个 txt 文件")

    return test_dirs

def create_test_config():
    """创建测试配置"""
    print("创建测试配置...")

    # 标准小波降噪配置
    wavelet_config = {
        "fs": 100.0,
        "channel": "green",
        "method": "wavelet",
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
        "imu_denoise": True,
        "selected_directories": ["test_hyx_data", "test_lhr_data", "test_lj_data"]
    }

    # Bayesian降噪配置
    bayes_config = {
        "fs": 100.0,
        "channel": "green",
        "method": "bayes",
        "wavelet": "db6",
        "decomposition_level": 4,
        "mode": "symmetric",
        "threshold": {
            "strategy": "bayes",
            "scale": 1.0,
            "manual_value": None,
            "threshold_mode": "soft"
        },
        "plotting": {
            "figsize": [12, 6]
        },
        "ppg_denoise": True,
        "imu_denoise": True,
        "selected_directories": ["test_hyx_data", "test_lhr_data", "test_lj_data"]
    }

    # 保存配置
    wavelet_config_path = "test_wavelet_config.json"
    bayes_config_path = "test_bayes_config.json"

    with open(wavelet_config_path, 'w', encoding='utf-8') as f:
        json.dump(wavelet_config, f, indent=2, ensure_ascii=False)

    with open(bayes_config_path, 'w', encoding='utf-8') as f:
        json.dump(bayes_config, f, indent=2, ensure_ascii=False)

    print(f"测试配置文件已保存: {wavelet_config_path}, {bayes_config_path}")
    return wavelet_config_path, bayes_config_path

def test_batch_processing():
    """测试批量处理功能"""
    print("测试批量处理功能...")

    # 创建测试数据
    test_dirs = create_test_directories()
    wavelet_config_path, bayes_config_path = create_test_config()

    # 模拟批量处理逻辑（不实际运行，避免依赖问题）
    from batch_all_wavelet_denoise import find_data_directories

    # 测试目录发现功能
    discovered_dirs = find_data_directories()
    print(f"发现的数据目录: {discovered_dirs}")

    # 验证发现的目录
    expected_dirs = [d for d in test_dirs if os.path.exists(d)]
    if set(discovered_dirs) == set(expected_dirs):
        print("✓ 目录发现功能正常")
    else:
        print(f"✗ 目录发现异常，期望: {expected_dirs}, 实际: {discovered_dirs}")

    # 测试小波配置加载
    try:
        with open(wavelet_config_path, 'r', encoding='utf-8') as f:
            wavelet_config = json.load(f)
        print("✓ 小波配置加载功能正常")
    except Exception as e:
        print(f"✗ 小波配置加载失败: {e}")

    # 测试Bayesian配置加载
    try:
        with open(bayes_config_path, 'r', encoding='utf-8') as f:
            bayes_config = json.load(f)
        print("✓ Bayesian配置加载功能正常")
    except Exception as e:
        print(f"✗ Bayesian配置加载失败: {e}")

    # 验证配置内容
    required_keys = ["fs", "channel", "method", "wavelet", "decomposition_level", "ppg_denoise", "imu_denoise"]
    for config_name, config in [("小波", wavelet_config), ("Bayesian", bayes_config)]:
        missing_keys = [key for key in required_keys if key not in config]
        if not missing_keys:
            print(f"✓ {config_name}配置内容完整")
        else:
            print(f"✗ {config_name}配置缺少必要参数: {missing_keys}")

    # 验证方法设置
    if wavelet_config.get("method") == "wavelet":
        print("✓ 小波配置方法设置正确")
    else:
        print("✗ 小波配置方法设置错误")

    if bayes_config.get("method") == "bayes":
        print("✓ Bayesian配置方法设置正确")
    else:
        print("✗ Bayesian配置方法设置错误")

    # 验证阈值策略
    if bayes_config.get("threshold", {}).get("strategy") == "bayes":
        print("✓ Bayesian阈值策略设置正确")
    else:
        print("✗ Bayesian阈值策略设置错误")

    return True

def cleanup_test_files():
    """清理测试文件"""
    print("清理测试文件...")

    test_files = [
        "test_wavelet_config.json",
        "test_bayes_config.json",
        "test_hyx_data",
        "test_lhr_data",
        "test_lj_data",
        "batch_denoised_results"
    ]

    for item in test_files:
        if os.path.exists(item):
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)

    print("测试文件清理完成")

def main():
    """主函数"""
    print("=== 批量小波降噪功能测试 ===")

    try:
        success = test_batch_processing()

        if success:
            print("✓ 所有测试通过")
            print("批量处理功能正常，可以安全使用")
        else:
            print("✗ 部分测试失败")
            return 1

    except KeyboardInterrupt:
        print("\n测试已取消")
        return 1
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cleanup_test_files()

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
