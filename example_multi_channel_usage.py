#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多通道小波降噪使用示例

这个脚本展示了如何使用多通道PPG和IMU降噪功能，
包括小波降噪和Bayesian降噪两种方法。
"""

import os
import sys
import json

def show_ppg_channels():
    """显示PPG通道处理"""
    print("=== PPG多通道处理 ===")
    print()
    print("支持的所有PPG通道:")
    print("  🟢 Green通道 - 主要用于心率检测")
    print("  🔴 IR通道 - 红外光通道")
    print("  🔵 Red通道 - 红色光通道")
    print()
    print("每个通道独立降噪，生成独立的对比图像")
    print()

def show_imu_processing():
    """显示IMU数据处理"""
    print("=== IMU数据处理 ===")
    print()
    print("支持的IMU数据:")
    print("  📱 ACC (加速度计) - X/Y/Z轴")
    print("  🔄 GYRO (陀螺仪) - X/Y/Z轴")
    print()
    print("每个轴独立降噪，生成三轴对比图")
    print()

def show_methods_comparison():
    """显示两种方法的对比"""
    print("=== 降噪方法对比 ===")
    print()
    print("1. 小波降噪 (Wavelet Denoising):")
    print("   • 原理: 小波变换 + 阈值收缩")
    print("   • 阈值策略: universal, bayes, manual")
    print("   • 优势: 计算高效，适合实时处理")
    print("   • 推荐: 标准应用场景")
    print()
    print("2. Bayesian降噪 (Bayesian Denoising):")
    print("   • 原理: Bayesian统计理论")
    print("   • 特点: 自适应阈值计算")
    print("   • 优势: 低信噪比下表现更好")
    print("   • 推荐: 噪声水平较高的情况")
    print()

def show_config_examples():
    """显示配置示例"""
    print("=== 配置示例 ===")
    print()
    print("1. 标准小波降噪配置:")
    config1 = {
        "method": "wavelet",
        "wavelet": "db6",
        "decomposition_level": 4,
        "threshold": {"strategy": "universal"}
    }
    print(f"   {json.dumps(config1, indent=2)}")
    print()
    print("2. Bayesian降噪配置:")
    config2 = {
        "method": "bayes",
        "wavelet": "db6",
        "decomposition_level": 4,
        "threshold": {"strategy": "bayes"}
    }
    print(f"   {json.dumps(config2, indent=2)}")
    print()

def show_output_structure():
    """显示输出结构"""
    print("=== 输出结构示例 ===")
    print()
    print("multi_denoise_results/")
    print("└── input_file_name/")
    print("    ├── input_file_green_original.png      # 绿色通道原始")
    print("    ├── input_file_green_denoised.png      # 绿色通道降噪")
    print("    ├── input_file_green_comparison.png    # 绿色通道对比")
    print("    ├── input_file_ir_original.png         # IR通道原始")
    print("    ├── input_file_ir_denoised.png         # IR通道降噪")
    print("    ├── input_file_ir_comparison.png       # IR通道对比")
    print("    ├── input_file_red_original.png        # 红色通道原始")
    print("    ├── input_file_red_denoised.png        # 红色通道降噪")
    print("    ├── input_file_red_comparison.png      # 红色通道对比")
    print("    ├── input_file_acc_comparison.png      # ACC三轴对比")
    print("    ├── input_file_gyro_comparison.png     # GYRO三轴对比")
    print("    └── input_file_summary.txt              # 处理报告")
    print()

def show_usage_commands():
    """显示使用命令"""
    print("=== 使用命令 ===")
    print()
    print("1. 标准小波降噪:")
    print("   python ppg_wavelet_denoise.py --input data.txt --method wavelet")
    print()
    print("2. Bayesian降噪:")
    print("   python ppg_wavelet_denoise.py --input data.txt --method bayes")
    print()
    print("3. 指定输出目录:")
    print("   python ppg_wavelet_denoise.py --input data.txt --output-dir results/")
    print()
    print("4. 批量处理:")
    print("   python batch_all_denoise.py --config multi_channel_config.json")
    print()

def show_ai_training_compatibility():
    """显示AI训练兼容性"""
    print("=== AI训练兼容性 ===")
    print()
    print("降噪后的数据适合用于多模态AI训练:")
    print()
    print("✅ PPG通道:")
    print("   • 绿色通道: 主要心率特征")
    print("   • IR通道: 辅助心率特征")
    print("   • 红色通道: 额外生理信息")
    print()
    print("✅ IMU数据:")
    print("   • ACC X/Y/Z: 运动和姿态信息")
    print("   • GYRO X/Y/Z: 旋转和角速度")
    print()
    print("✅ 信号质量:")
    print("   • 统一的降噪参数")
    print("   • 标准化的输出格式")
    print("   • 完整的信噪比报告")
    print()

def main():
    """主函数"""
    print("多通道小波降噪工具使用指南")
    print("=" * 50)
    print()

    show_ppg_channels()
    show_imu_processing()
    show_methods_comparison()
    show_config_examples()
    show_output_structure()
    show_usage_commands()
    show_ai_training_compatibility()

    print("=" * 50)
    print("🎯 关键优势:")
    print("1. 同时处理所有PPG通道和IMU数据")
    print("2. 支持小波和Bayesian两种降噪方法")
    print("3. 生成丰富的对比可视化图像")
    print("4. 提供详细的降噪效果报告")
    print("5. 完美适合多模态AI训练")
    print()
    print("🚀 立即开始使用多通道降噪功能！")

if __name__ == "__main__":
    main()

