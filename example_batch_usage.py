#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量小波降噪使用示例

这个脚本展示了如何使用批量处理工具一次性处理所有数据目录。
"""

import os
import sys

def show_basic_usage():
    """显示基本使用方法"""
    print("=== 批量小波降噪基本使用 ===")
    print()
    print("1. 自动处理所有数据目录（推荐）:")
    print("   python batch_all_wavelet_denoise.py")
    print("   python batch_all_denoise.py --auto")
    print()
    print("2. 使用配置文件:")
    print("   python batch_all_denoise.py --config batch_denoise_config.json")
    print()
    print("3. 指定特定目录:")
    print("   python batch_all_denoise.py --dirs hyx_data lhr_data")
    print()

def show_current_data():
    """显示当前可用的数据目录"""
    print("=== 当前项目中的数据目录 ===")
    print()

    data_dirs = []
    for item in os.listdir('.'):
        if os.path.isdir(item) and item.endswith('_data'):
            # 检查目录是否包含txt文件
            txt_files = [f for f in os.listdir(item) if f.endswith('.txt')]
            if txt_files:
                data_dirs.append((item, len(txt_files)))

    if not data_dirs:
        print("未发现任何数据目录")
        return

    print("发现以下数据目录:")
    for dir_name, file_count in sorted(data_dirs):
        print(f"  📁 {dir_name}: {file_count} 个数据文件")

    print()
    print(f"总共 {len(data_dirs)} 个数据目录，可一次性处理 {sum(count for _, count in data_dirs)} 个文件")
    print()

def show_expected_output():
    """显示预期输出结构"""
    print("=== 预期输出结构 ===")
    print()
    print("处理后会在以下目录中生成结果:")
    print("batch_denoised_results/")
    print("└── batch_all_YYYYMMDD-HHMM/")
    print("    ├── batch_config.json          # 批量配置参数")
    print("    ├── batch_summary.txt          # 处理汇总报告")
    print("    ├── denoised_hyx_YYYYMMDD-HHMM/    # hyx_data 结果")
    print("    ├── denoised_lhr_YYYYMMDD-HHMM/    # lhr_data 结果")
    print("    └── denoised_lj_YYYYMMDD-HHMM/     # lj_data 结果")
    print()

def show_comparison():
    """显示与单目录处理的对比"""
    print("=== 与单目录处理的对比 ===")
    print()
    print("单目录处理:")
    print("  📁 python wavelet_denoise_batch.py --dir hyx_data")
    print("  📁 python wavelet_denoise_batch.py --dir lhr_data")
    print("  📁 python wavelet_denoise_batch.py --dir lj_data")
    print("  ⚠️  需要手动配置3次，每次单独处理")
    print()
    print("批量处理:")
    print("  🚀 python batch_all_denoise.py --auto")
    print("  ✅ 自动发现所有目录，一次性处理")
    print("  ✅ 统一配置，统一报告")
    print("  ✅ 进度跟踪，结果汇总")
    print()

def show_config_examples():
    """显示配置示例"""
    print("=== 配置示例 ===")
    print()
    print("1. 标准配置 (batch_denoise_config.json):")
    print("   - 采样率: 100 Hz")
    print("   - PPG通道: 绿色通道")
    print("   - 小波: db6")
    print("   - 分解层数: 4")
    print("   - 阈值策略: 通用阈值")
    print()
    print("2. 自定义配置:")
    print("   - 修改 batch_denoise_config.json")
    print("   - 调整小波类型、分解层数等参数")
    print("   - 选择要处理的目录")
    print()

def show_gui_features():
    """显示GUI功能特点"""
    print("=== 图形化界面功能 ===")
    print()
    print("🎛️ 目录管理:")
    print("  - 自动扫描所有 xxx_data 目录")
    print("  - 勾选要处理的目录")
    print("  - 实时显示文件数量")
    print()
    print("⚙️ 参数配置:")
    print("  - 采样率和通道选择")
    print("  - 小波类型和分解层数")
    print("  - 阈值策略和模式")
    print("  - PPG/IMU降噪选项")
    print()
    print("📊 进度监控:")
    print("  - 实时显示处理进度")
    print("  - 每个目录的状态反馈")
    print("  - 处理完成后的汇总")
    print()

def main():
    """主函数"""
    print("批量小波降噪工具使用指南")
    print("=" * 50)

    show_current_data()
    show_basic_usage()
    show_expected_output()
    show_comparison()
    show_config_examples()
    show_gui_features()

    print("=" * 50)
    print("快速开始:")
    print("1. 运行: python batch_all_denoise.py")
    print("2. 在GUI中选择参数和目录")
    print("3. 点击开始处理")
    print("4. 查看 batch_denoised_results 目录中的结果")
    print()
    print("🎉 享受批量处理的高效体验！")

if __name__ == "__main__":
    main()

