#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python 3.11 GPU训练环境设置
使用conda或直接安装PyTorch GPU版本
"""

import os
import sys
import subprocess
import json

def create_conda_environment():
    """创建conda环境"""
    print("🐍 创建Python 3.11 conda环境...")
    
    # 尝试不同的conda路径
    conda_paths = [
        "conda",
        "G:\\anaconda3\\Scripts\\conda.exe",
        "C:\\Users\\tsawke\\anaconda3\\Scripts\\conda.exe",
        "C:\\ProgramData\\Anaconda3\\Scripts\\conda.exe"
    ]
    
    for conda_path in conda_paths:
        try:
            print(f"尝试使用: {conda_path}")
            result = subprocess.run([
                conda_path, "create", "-n", "swallow-gpu", "python=3.11", "-y"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ conda环境创建成功")
                return conda_path
            else:
                print(f"❌ 失败: {result.stderr}")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"❌ 异常: {e}")
    
    return None

def install_pytorch_gpu(conda_path):
    """安装PyTorch GPU版本"""
    print("\n📦 安装PyTorch GPU版本...")
    
    # 激活环境并安装PyTorch
    install_commands = [
        [conda_path, "install", "-n", "swallow-gpu", "-c", "pytorch", "-c", "nvidia", "pytorch", "torchvision", "torchaudio", "pytorch-cuda=12.1", "-y"],
        [conda_path, "install", "-n", "swallow-gpu", "-c", "pytorch", "-c", "nvidia", "pytorch", "torchvision", "torchaudio", "pytorch-cuda=11.8", "-y"],
        [conda_path, "run", "-n", "swallow-gpu", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"]
    ]
    
    for cmd in install_commands:
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

def install_other_dependencies(conda_path):
    """安装其他依赖"""
    print("\n📦 安装其他依赖...")
    
    dependencies = [
        "numpy", "pandas", "matplotlib", "scipy", "scikit-learn", 
        "seaborn", "tqdm", "jupyter", "ipykernel"
    ]
    
    for dep in dependencies:
        try:
            cmd = [conda_path, "install", "-n", "swallow-gpu", dep, "-y"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {dep} 安装成功")
            else:
                print(f"⚠️ {dep} 安装失败，尝试pip安装")
                # 尝试pip安装
                pip_cmd = [conda_path, "run", "-n", "swallow-gpu", "pip", "install", dep]
                subprocess.run(pip_cmd, capture_output=True, text=True, timeout=300)
        except Exception as e:
            print(f"⚠️ {dep} 安装异常: {e}")

def test_gpu_functionality(conda_path):
    """测试GPU功能"""
    print("\n🧪 测试GPU功能...")
    
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
        cmd = [conda_path, "run", "-n", "swallow-gpu", "python", "-c", test_script]
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

def create_run_script(conda_path):
    """创建运行脚本"""
    script_content = f'''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python 3.11 GPU训练运行脚本
"""

import subprocess
import sys
import os

def main():
    print("🚀 Python 3.11 GPU训练")
    print("="*50)
    
    # 使用conda环境运行训练
    cmd = [
        "{conda_path}", "run", "-n", "swallow-gpu", "python",
        "run_behavior_classification.py",
        "--full-pipeline",
        "--config", "python311_gpu_config.json",
        "--mixed-precision",
        "--batch-size", "32"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ 训练完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
    except Exception as e:
        print(f"❌ 异常: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open("run_python311_training.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✅ Python 3.11训练脚本创建完成")

def main():
    print("="*60)
    print("Python 3.11 GPU训练环境设置")
    print("="*60)
    
    # 创建conda环境
    conda_path = create_conda_environment()
    
    if conda_path:
        # 安装PyTorch
        if install_pytorch_gpu(conda_path):
            # 安装其他依赖
            install_other_dependencies(conda_path)
            
            # 测试GPU功能
            if test_gpu_functionality(conda_path):
                print("\n🎉 Python 3.11 GPU训练环境配置成功！")
                
                # 创建配置文件
                create_gpu_config()
                
                # 创建运行脚本
                create_run_script(conda_path)
                
                print("\n📋 使用方法:")
                print("1. 激活环境: conda activate swallow-gpu")
                print("2. 运行训练: python run_python311_training.py")
                print("3. 或直接运行: conda run -n swallow-gpu python run_behavior_classification.py --full-pipeline --config python311_gpu_config.json --mixed-precision")
                
                return True
            else:
                print("\n⚠️ PyTorch安装成功但GPU不可用")
        else:
            print("\n❌ PyTorch安装失败")
    else:
        print("\n❌ conda环境创建失败")
    
    print("\n💡 替代方案:")
    print("1. 手动安装Python 3.11")
    print("2. 使用Docker环境")
    print("3. 使用云端GPU服务")
    
    return False

if __name__ == "__main__":
    main()
