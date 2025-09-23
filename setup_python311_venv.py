#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建Python 3.11虚拟环境并安装GPU训练所需模块
"""

import os
import sys
import subprocess
import json
import urllib.request
import zipfile

def download_python311():
    """下载Python 3.11"""
    print("📥 下载Python 3.11...")
    
    # Python 3.11下载URL
    python_url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
    python_installer = "python-3.11.9-amd64.exe"
    
    try:
        print("正在下载Python 3.11.9...")
        urllib.request.urlretrieve(python_url, python_installer)
        print("✅ Python 3.11.9下载完成")
        return python_installer
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

def install_python311(installer_path):
    """安装Python 3.11"""
    print("🔧 安装Python 3.11...")
    
    try:
        # 静默安装Python 3.11
        cmd = [installer_path, "/quiet", "InstallAllUsers=0", "PrependPath=1"]
        result = subprocess.run(cmd, check=True)
        print("✅ Python 3.11安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        return False

def create_venv():
    """创建虚拟环境"""
    print("🐍 创建Python 3.11虚拟环境...")
    
    try:
        # 使用Python 3.11创建虚拟环境
        cmd = ["py", "-3.11", "-m", "venv", "python311_gpu_env"]
        result = subprocess.run(cmd, check=True)
        print("✅ 虚拟环境创建成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 虚拟环境创建失败: {e}")
        return False

def install_pytorch_gpu():
    """安装PyTorch GPU版本"""
    print("📦 安装PyTorch GPU版本...")
    
    # 激活虚拟环境并安装PyTorch
    if os.name == 'nt':  # Windows
        activate_script = "python311_gpu_env\\Scripts\\activate.bat"
        pip_path = "python311_gpu_env\\Scripts\\pip.exe"
    else:  # Linux/Mac
        activate_script = "python311_gpu_env/bin/activate"
        pip_path = "python311_gpu_env/bin/pip"
    
    # 安装PyTorch GPU版本
    pytorch_commands = [
        [pip_path, "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"],
        [pip_path, "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"],
        [pip_path, "install", "torch", "torchvision", "torchaudio"]
    ]
    
    for cmd in pytorch_commands:
        try:
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("✅ PyTorch安装成功")
                return True
            else:
                print(f"❌ 失败: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("⏰ 安装超时")
        except Exception as e:
            print(f"❌ 异常: {e}")
    
    return False

def install_dependencies():
    """安装其他依赖"""
    print("📦 安装其他依赖...")
    
    if os.name == 'nt':  # Windows
        pip_path = "python311_gpu_env\\Scripts\\pip.exe"
    else:  # Linux/Mac
        pip_path = "python311_gpu_env/bin/pip"
    
    dependencies = [
        "numpy", "pandas", "matplotlib", "scipy", "scikit-learn", 
        "seaborn", "tqdm", "jupyter", "ipykernel"
    ]
    
    for dep in dependencies:
        try:
            cmd = [pip_path, "install", dep]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {dep} 安装成功")
            else:
                print(f"⚠️ {dep} 安装失败: {result.stderr}")
        except Exception as e:
            print(f"⚠️ {dep} 安装异常: {e}")

def test_gpu_functionality():
    """测试GPU功能"""
    print("🧪 测试GPU功能...")
    
    if os.name == 'nt':  # Windows
        python_path = "python311_gpu_env\\Scripts\\python.exe"
    else:  # Linux/Mac
        python_path = "python311_gpu_env/bin/python"
    
    test_script = """
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA版本: {torch.version.cuda}')
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'GPU名称: {torch.cuda.get_device_name(0)}')
    
    # 测试GPU计算
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print('✅ GPU计算测试成功')
    except Exception as e:
        print(f'❌ GPU计算测试失败: {e}')
else:
    print('❌ CUDA不可用')
"""
    
    try:
        cmd = [python_path, "-c", test_script]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"❌ 测试失败: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        return False

def create_gpu_config():
    """创建GPU配置文件"""
    config = {
        "data_dir": "./hyx_data",
        "batch_size": 32,
        "learning_rate": 1e-4,
        "num_epochs": 100,
        "patience": 15,
        "k_folds": 5,
        "test_size": 0.2,
        "model_types": ["cnn", "transformer", "fusion"],
        "use_class_weights": True,
        "augmentation": True,
        "feature_types": ["stft", "time_series"],
        "use_gpu": True,
        "device_id": 0,
        "use_amp": True,
        "compile_model": True,
        "multi_gpu": False,
        "device_count": 1,
        "dataloader_params": {
            "num_workers": 8,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "drop_last": True
        }
    }
    
    with open("python311_gpu_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("✅ Python 3.11 GPU配置文件创建完成")

def create_run_script():
    """创建运行脚本"""
    if os.name == 'nt':  # Windows
        python_path = "python311_gpu_env\\Scripts\\python.exe"
        script_content = f'''@echo off
echo 🚀 Python 3.11 GPU训练
echo ============================================================

call python311_gpu_env\\Scripts\\activate.bat

echo 开始训练...
{python_path} run_behavior_classification.py --full-pipeline --config python311_gpu_config.json --mixed-precision --batch-size 32

echo 训练完成！
pause
'''
        script_file = "run_python311_gpu_training.bat"
    else:  # Linux/Mac
        python_path = "python311_gpu_env/bin/python"
        script_content = f'''#!/bin/bash
echo "🚀 Python 3.11 GPU训练"
echo "============================================================"

source python311_gpu_env/bin/activate

echo "开始训练..."
{python_path} run_behavior_classification.py --full-pipeline --config python311_gpu_config.json --mixed-precision --batch-size 32

echo "训练完成！"
'''
        script_file = "run_python311_gpu_training.sh"
    
    with open(script_file, "w", encoding="utf-8") as f:
        f.write(script_content)
    
    if os.name != 'nt':  # Linux/Mac
        os.chmod(script_file, 0o755)
    
    print(f"✅ Python 3.11训练脚本创建完成: {script_file}")

def main():
    print("="*60)
    print("Python 3.11 GPU训练环境设置")
    print("="*60)
    
    # 检查是否已有Python 3.11
    try:
        result = subprocess.run(["py", "-3.11", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ 检测到Python 3.11: {result.stdout.strip()}")
            python311_available = True
        else:
            python311_available = False
    except:
        python311_available = False
    
    if not python311_available:
        print("❌ 未检测到Python 3.11")
        print("请先安装Python 3.11:")
        print("1. 访问 https://www.python.org/downloads/release/python-3119/")
        print("2. 下载并安装Python 3.11.9")
        print("3. 重新运行此脚本")
        return False
    
    # 创建虚拟环境
    if not create_venv():
        return False
    
    # 安装PyTorch
    if not install_pytorch_gpu():
        print("⚠️ PyTorch安装失败，继续安装其他依赖")
    
    # 安装其他依赖
    install_dependencies()
    
    # 测试GPU功能
    if test_gpu_functionality():
        print("\n🎉 Python 3.11 GPU训练环境配置成功！")
        
        # 创建配置文件
        create_gpu_config()
        
        # 创建运行脚本
        create_run_script()
        
        print("\n📋 使用方法:")
        if os.name == 'nt':  # Windows
            print("1. 运行训练: run_python311_gpu_training.bat")
            print("2. 或手动激活环境: python311_gpu_env\\Scripts\\activate.bat")
            print("3. 然后运行: python run_behavior_classification.py --full-pipeline --config python311_gpu_config.json --mixed-precision")
        else:  # Linux/Mac
            print("1. 运行训练: ./run_python311_gpu_training.sh")
            print("2. 或手动激活环境: source python311_gpu_env/bin/activate")
            print("3. 然后运行: python run_behavior_classification.py --full-pipeline --config python311_gpu_config.json --mixed-precision")
        
        return True
    else:
        print("\n⚠️ GPU功能测试失败")
        print("可能的原因:")
        print("1. CUDA驱动未安装")
        print("2. GPU不支持CUDA")
        print("3. PyTorch版本与CUDA版本不匹配")
        return False

if __name__ == "__main__":
    main()
