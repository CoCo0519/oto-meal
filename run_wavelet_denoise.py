#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
小波降噪批处理运行脚本

这个脚本提供了便捷的命令行接口来运行小波降噪批处理，
基于 ReadDirectory.py 的逻辑，但使用小波方法进行降噪。

使用示例:
    python run_wavelet_denoise.py --dir ./hyx_data
    python run_wavelet_denoise.py --dir ./lhr_data --config my_config.json
"""

import os
import sys
import argparse
from wavelet_denoise_batch import main as run_wavelet_denoise_batch

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="小波降噪批处理工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dir", "-d",
        type=str,
        required=True,
        help="包含 .txt 文件的目录（建议形如 xxx_data）"
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        help="配置文件路径（可选，不指定则启动图形界面）"
    )

    args = parser.parse_args()

    print("=== 小波降噪批处理工具 ===")
    print(f"输入目录: {args.dir}")
    if args.config:
        print(f"配置文件: {args.config}")

    try:
        # 调用主函数
        run_wavelet_denoise_batch()
    except KeyboardInterrupt:
        print("\n操作已取消")
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

