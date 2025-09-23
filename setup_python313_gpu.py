#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python 3.13 GPU训练环境设置
使用多种方法尝试安装PyTorch GPU版本
"""

import os
import sys
import subprocess
import urllib.request
import platform
import json

def create_gpu_config():
    """创建GPU训练配置"""
    config = {
        "data_dir": "./hyx_data",
        "batch_size": 16,
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
        "compile_model": False,  # Python 3.13可能不支持
        "multi_gpu": False,
        "device_count": 1,
        "dataloader_params": {
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "drop_last": True
        }
    }
    
    with open("python313_gpu_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("✅ Python 3.13 GPU配置文件创建完成")

def install_pytorch_methods():
    """尝试多种方法安装PyTorch GPU版本"""
    methods = [
        {
            "name": "方法1: 标准CUDA 12.1安装",
            "cmd": [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"]
        },
        {
            "name": "方法2: CUDA 11.8安装",
            "cmd": [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"]
        },
        {
            "name": "方法3: CPU版本+手动CUDA",
            "cmd": [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
        },
        {
            "name": "方法4: 预发布版本",
            "cmd": [sys.executable, "-m", "pip", "install", "--pre", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/nightly/cu121"]
        }
    ]
    
    for method in methods:
        print(f"\n🔄 尝试{method['name']}...")
        try:
            result = subprocess.run(method['cmd'], capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ {method['name']}成功")
                return True
            else:
                print(f"❌ {method['name']}失败: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"⏰ {method['name']}超时")
        except Exception as e:
            print(f"❌ {method['name']}异常: {e}")
    
    return False

def test_gpu_functionality():
    """测试GPU功能"""
    print("\n🧪 测试GPU功能...")
    
    test_code = """
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
        return True
    except Exception as e:
        print(f'❌ GPU计算测试失败: {e}')
        return False
else:
    print('❌ CUDA不可用')
    return False
"""
    
    try:
        exec(test_code)
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def create_fallback_script():
    """创建回退脚本"""
    script_content = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python 3.13 GPU训练回退脚本
如果PyTorch GPU不可用，使用CPU模式但保持GPU优化配置
"""

import os
import sys
import torch

def setup_environment():
    """设置环境"""
    print("🔧 设置Python 3.13训练环境...")
    
    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print("✅ GPU训练环境就绪")
        return True
    else:
        print("⚠️ GPU不可用，使用CPU模式")
        # 设置CPU优化
        torch.set_num_threads(min(8, os.cpu_count() or 4))
        return False

def run_training():
    """运行训练"""
    use_gpu = setup_environment()
    
    # 导入训练脚本
    try:
        from run_behavior_classification import main as training_main
        
        # 设置命令行参数
        if use_gpu:
            sys.argv = [
                'run_behavior_classification.py',
                '--full-pipeline',
                '--config', 'python313_gpu_config.json',
                '--mixed-precision',
                '--batch-size', '16'
            ]
        else:
            sys.argv = [
                'run_behavior_classification.py',
                '--full-pipeline',
                '--config', 'python313_gpu_config.json',
                '--no-gpu',
                '--batch-size', '8'
            ]
        
        print("🚀 开始训练...")
        training_main()
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")

if __name__ == "__main__":
    run_training()
'''
    
    with open("run_python313_training.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✅ Python 3.13训练回退脚本创建完成")

def main():
    print("="*60)
    print("Python 3.13 GPU训练环境设置")
    print("="*60)
    
    # 检查Python版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python版本: {python_version}")
    
    if python_version != "3.13":
        print("⚠️ 此脚本专为Python 3.13设计")
    
    # 创建配置文件
    create_gpu_config()
    
    # 尝试安装PyTorch
    print("\n📦 尝试安装PyTorch GPU版本...")
    if install_pytorch_methods():
        print("✅ PyTorch安装成功")
        
        # 测试GPU功能
        if test_gpu_functionality():
            print("\n🎉 Python 3.13 GPU训练环境配置成功！")
            print("可以使用以下命令开始训练:")
            print("python run_behavior_classification.py --full-pipeline --config python313_gpu_config.json --mixed-precision")
        else:
            print("\n⚠️ PyTorch安装成功但GPU不可用")
            print("可能的原因:")
            print("1. CUDA驱动未安装")
            print("2. GPU不支持CUDA")
            print("3. PyTorch版本与CUDA版本不匹配")
    else:
        print("\n❌ 所有PyTorch安装方法都失败了")
        print("可能的原因:")
        print("1. 网络连接问题")
        print("2. Python 3.13与PyTorch不兼容")
        print("3. 防火墙阻止下载")
    
    # 创建回退脚本
    create_fallback_script()
    
    print("\n💡 替代方案:")
    print("1. 使用回退脚本: python run_python313_training.py")
    print("2. 使用Docker环境")
    print("3. 降级到Python 3.11/3.12")
    print("4. 等待官方支持Python 3.13")

if __name__ == "__main__":
    main()
