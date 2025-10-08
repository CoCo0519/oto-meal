#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量小波降噪便捷运行脚本

这个脚本提供了最简单的命令行接口来批量处理所有数据目录。

使用示例:
    python batch_all_denoise.py
    python batch_all_denoise.py --config my_config.json
    python batch_all_denoise.py --auto  # 自动发现并处理所有数据目录
"""

import os
import sys
import argparse
from batch_all_wavelet_denoise import main as run_batch_denoise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量小波降噪处理 - 便捷版本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        help="配置文件路径（可选，不指定则启动GUI）"
    )

    parser.add_argument(
        "--auto", "-a",
        action="store_true",
        help="自动发现并处理所有 xxx_data 目录"
    )

    parser.add_argument(
        "--dirs", "-d",
        nargs="*",
        help="指定要处理的数据目录"
    )

    args = parser.parse_args()

    print("=== 批量小波降噪处理工具 (便捷版) ===")

    if args.auto:
        print("🔍 自动发现数据目录...")
        # 传递参数给主函数
        sys.argv = [sys.argv[0]]  # 清除其他参数
    elif args.dirs:
        print(f"📁 指定处理目录: {', '.join(args.dirs)}")
        # 传递参数给主函数
        sys.argv = [sys.argv[0], "--dirs"] + args.dirs
    elif args.config:
        print(f"⚙️ 使用配置文件: {args.config}")
        # 传递参数给主函数
        sys.argv = [sys.argv[0], "--config", args.config]
    else:
        print("🎛️ 启动图形化配置界面...")
        # 清除参数，让主函数启动GUI
        sys.argv = [sys.argv[0]]

    try:
        # 调用主函数
        run_batch_denoise()
    except KeyboardInterrupt:
        print("\n⏹️  操作已取消")
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
