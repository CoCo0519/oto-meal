#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为Python 3.13编译PyTorch GPU版本
从源码编译PyTorch以支持Python 3.13和CUDA
"""

import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import shutil

def check_requirements():
    """检查编译要求"""
    print("🔍 检查编译要求...")
    
    # 检查CUDA
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA编译器可用")
            return True
        else:
            print("❌ CUDA编译器不可用")
            return False
    except FileNotFoundError:
        print("❌ nvcc未找到，请安装CUDA Toolkit")
        return False

def install_build_dependencies():
    """安装编译依赖"""
    print("📦 安装编译依赖...")
    
    dependencies = [
        "ninja",
        "cmake", 
        "setuptools",
        "wheel",
        "numpy",
        "pyyaml",
        "requests",
        "typing-extensions",
        "sympy",
        "networkx",
        "jinja2",
        "fsspec",
        "filelock"
    ]
    
    for dep in dependencies:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
            print(f"✅ {dep} 安装完成")
        except subprocess.CalledProcessError:
            print(f"⚠️ {dep} 安装失败，继续...")

def download_pytorch_source():
    """下载PyTorch源码"""
    print("📥 下载PyTorch源码...")
    
    pytorch_url = "https://github.com/pytorch/pytorch/archive/refs/heads/main.zip"
    pytorch_zip = "pytorch-main.zip"
    
    try:
        urllib.request.urlretrieve(pytorch_url, pytorch_zip)
        print("✅ PyTorch源码下载完成")
        
        # 解压
        with zipfile.ZipFile(pytorch_zip, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # 重命名目录
        if os.path.exists("pytorch-main"):
            if os.path.exists("pytorch"):
                shutil.rmtree("pytorch")
            os.rename("pytorch-main", "pytorch")
        
        # 清理zip文件
        os.remove(pytorch_zip)
        
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return False

def build_pytorch():
    """编译PyTorch"""
    print("🔨 开始编译PyTorch...")
    
    os.chdir("pytorch")
    
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_HOME"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1"
    env["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6"
    env["USE_CUDA"] = "1"
    env["USE_CUDNN"] = "1"
    env["USE_NCCL"] = "0"
    env["USE_DISTRIBUTED"] = "0"
    
    # 编译命令
    build_cmd = [
        sys.executable, "setup.py", "build_ext", "--inplace"
    ]
    
    try:
        print("开始编译（这可能需要1-2小时）...")
        subprocess.run(build_cmd, env=env, check=True)
        print("✅ PyTorch编译完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 编译失败: {e}")
        return False

def install_pytorch():
    """安装编译好的PyTorch"""
    print("📦 安装编译好的PyTorch...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
        print("✅ PyTorch安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        return False

def test_pytorch():
    """测试PyTorch GPU功能"""
    print("🧪 测试PyTorch GPU功能...")
    
    test_code = """
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'GPU名称: {torch.cuda.get_device_name(0)}')
    
    # 测试GPU计算
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print('✅ GPU计算测试成功')
else:
    print('❌ CUDA不可用')
"""
    
    try:
        exec(test_code)
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    print("="*60)
    print("Python 3.13 PyTorch GPU编译脚本")
    print("="*60)
    
    # 检查Python版本
    if sys.version_info[:2] != (3, 13):
        print(f"⚠️ 当前Python版本: {sys.version_info.major}.{sys.version_info.minor}")
        print("此脚本专为Python 3.13设计")
    
    # 检查要求
    if not check_requirements():
        print("\n❌ 编译要求不满足，请安装CUDA Toolkit")
        print("下载地址: https://developer.nvidia.com/cuda-downloads")
        return False
    
    # 安装依赖
    install_build_dependencies()
    
    # 下载源码
    if not download_pytorch_source():
        return False
    
    # 编译
    if not build_pytorch():
        return False
    
    # 安装
    if not install_pytorch():
        return False
    
    # 测试
    if test_pytorch():
        print("\n🎉 PyTorch GPU版本编译和安装成功！")
        print("现在可以使用Python 3.13进行GPU训练了")
        return True
    else:
        print("\n❌ 编译完成但测试失败")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💡 替代方案:")
        print("1. 使用Docker环境")
        print("2. 降级到Python 3.11/3.12")
        print("3. 等待官方支持Python 3.13")
