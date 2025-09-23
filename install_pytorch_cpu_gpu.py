#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为Python 3.13安装PyTorch CPU版本，然后尝试GPU支持
"""

import os
import sys
import subprocess
import json

def install_pytorch_cpu():
    """安装PyTorch CPU版本"""
    print("📦 安装PyTorch CPU版本...")
    
    try:
        cmd = [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ PyTorch CPU版本安装成功")
            return True
        else:
            print(f"❌ 安装失败: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ 安装超时")
        return False
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False

def install_other_dependencies():
    """安装其他依赖"""
    print("📦 安装其他依赖...")
    
    dependencies = [
        "numpy", "pandas", "matplotlib", "scipy", "scikit-learn", 
        "seaborn", "tqdm", "jupyter", "ipykernel"
    ]
    
    for dep in dependencies:
        try:
            cmd = [sys.executable, "-m", "pip", "install", dep]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ {dep} 安装成功")
            else:
                print(f"⚠️ {dep} 安装失败: {result.stderr}")
        except Exception as e:
            print(f"⚠️ {dep} 安装异常: {e}")

def test_pytorch():
    """测试PyTorch功能"""
    print("🧪 测试PyTorch功能...")
    
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
    print('ℹ️ CUDA不可用，使用CPU模式')
    
    # 测试CPU计算
    try:
        x = torch.randn(100, 100)
        y = torch.randn(100, 100)
        z = torch.matmul(x, y)
        print('✅ CPU计算测试成功')
    except Exception as e:
        print(f'❌ CPU计算测试失败: {e}')
"""
    
    try:
        exec(test_script)
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def create_config():
    """创建配置文件"""
    print("📝 创建配置文件...")
    
    # 检测GPU可用性
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if use_gpu else 1
    except:
        use_gpu = False
        device_count = 1
    
    config = {
        "data_dir": "./hyx_data",
        "batch_size": 16 if use_gpu else 8,
        "learning_rate": 1e-4,
        "num_epochs": 100,
        "patience": 15,
        "k_folds": 5,
        "test_size": 0.2,
        "model_types": ["cnn", "transformer", "fusion"],
        "use_class_weights": True,
        "augmentation": True,
        "feature_types": ["stft", "time_series"],
        "use_gpu": use_gpu,
        "device_id": 0,
        "use_amp": use_gpu,
        "compile_model": False,  # Python 3.13可能不支持
        "multi_gpu": False,
        "device_count": device_count,
        "dataloader_params": {
            "num_workers": 4 if use_gpu else 2,
            "pin_memory": use_gpu,
            "persistent_workers": use_gpu,
            "prefetch_factor": 2,
            "drop_last": True
        }
    }
    
    with open("python313_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("✅ 配置文件创建完成")

def create_run_script():
    """创建运行脚本"""
    print("📝 创建运行脚本...")
    
    script_content = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python 3.13训练运行脚本
"""

import subprocess
import sys
import os

def main():
    print("🚀 Python 3.13训练")
    print("="*50)
    
    # 检查PyTorch
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU数量: {torch.cuda.device_count()}")
            print(f"GPU名称: {torch.cuda.get_device_name(0)}")
            config_file = "python313_config.json"
            batch_size = "16"
            use_mixed_precision = "--mixed-precision"
        else:
            print("使用CPU模式")
            config_file = "python313_config.json"
            batch_size = "8"
            use_mixed_precision = "--no-gpu"
    except ImportError:
        print("❌ PyTorch未安装")
        return
    
    # 运行训练
    cmd = [
        sys.executable,
        "run_behavior_classification.py",
        "--full-pipeline",
        "--config", config_file,
        use_mixed_precision,
        "--batch-size", batch_size
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
    
    with open("run_python313_training.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✅ 运行脚本创建完成")

def main():
    print("="*60)
    print("Python 3.13 PyTorch安装和配置")
    print("="*60)
    
    # 检查Python版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python版本: {python_version}")
    
    if python_version != "3.13":
        print("⚠️ 此脚本专为Python 3.13设计")
    
    # 安装PyTorch CPU版本
    if install_pytorch_cpu():
        # 安装其他依赖
        install_other_dependencies()
        
        # 测试PyTorch
        if test_pytorch():
            print("\n🎉 PyTorch安装和测试成功！")
            
            # 创建配置文件
            create_config()
            
            # 创建运行脚本
            create_run_script()
            
            print("\n📋 使用方法:")
            print("1. 运行训练: python run_python313_training.py")
            print("2. 或直接运行: python run_behavior_classification.py --full-pipeline --config python313_config.json")
            
            return True
        else:
            print("\n❌ PyTorch测试失败")
            return False
    else:
        print("\n❌ PyTorch安装失败")
        return False

if __name__ == "__main__":
    main()
