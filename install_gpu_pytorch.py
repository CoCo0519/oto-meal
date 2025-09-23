#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU PyTorch 安装脚本
用于解决Python 3.13与PyTorch CUDA版本不兼容的问题
"""

import os
import sys
import subprocess
import urllib.request
import platform

def download_file(url, filename):
    """下载文件"""
    print(f"正在下载: {filename}")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✅ 下载完成: {filename}")
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def install_wheel(wheel_file):
    """安装wheel文件"""
    print(f"正在安装: {wheel_file}")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", wheel_file, "--force-reinstall"], check=True)
        print(f"✅ 安装完成: {wheel_file}")
        return True
    except Exception as e:
        print(f"❌ 安装失败: {e}")
        return False

def main():
    print("="*60)
    print("GPU PyTorch 安装脚本")
    print("="*60)
    
    # 检查Python版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python版本: {python_version}")
    
    if python_version == "3.13":
        print("⚠️ 检测到Python 3.13，PyTorch CUDA版本可能不兼容")
        print("建议使用Python 3.11或3.12")
        
        # 尝试安装最新可用的CUDA版本
        print("\n尝试安装PyTorch CUDA版本...")
        
        # PyTorch 2.7.0 CUDA 12.1 (可能兼容)
        torch_url = "https://download.pytorch.org/whl/cu121/torch-2.7.0%2Bcu121-cp313-cp313-win_amd64.whl"
        torchvision_url = "https://download.pytorch.org/whl/cu121/torchvision-0.22.0%2Bcu121-cp313-cp313-win_amd64.whl"
        torchaudio_url = "https://download.pytorch.org/whl/cu121/torchaudio-2.7.0%2Bcu121-cp313-cp313-win_amd64.whl"
        
        files_to_download = [
            (torch_url, "torch-2.7.0+cu121-cp313-cp313-win_amd64.whl"),
            (torchvision_url, "torchvision-0.22.0+cu121-cp313-cp313-win_amd64.whl"),
            (torchaudio_url, "torchaudio-2.7.0+cu121-cp313-cp313-win_amd64.whl")
        ]
        
        # 下载并安装
        for url, filename in files_to_download:
            if download_file(url, filename):
                install_wheel(filename)
                # 清理下载的文件
                try:
                    os.remove(filename)
                except:
                    pass
    
    else:
        print(f"✅ Python {python_version} 与PyTorch CUDA版本兼容")
        print("使用标准安装命令...")
        
        # 标准安装命令
        install_cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
        
        try:
            subprocess.run(install_cmd, check=True)
            print("✅ PyTorch CUDA版本安装完成")
        except Exception as e:
            print(f"❌ 安装失败: {e}")
    
    # 验证安装
    print("\n验证安装...")
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"GPU名称: {torch.cuda.get_device_name(0)}")
            print("🎉 GPU训练环境配置成功！")
        else:
            print("⚠️ CUDA不可用，请检查GPU驱动")
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")

if __name__ == "__main__":
    main()

