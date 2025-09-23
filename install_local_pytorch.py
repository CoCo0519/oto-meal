#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本地安装PyTorch GPU版本
下载并安装兼容Python 3.13的PyTorch wheel文件
"""

import os
import sys
import subprocess
import urllib.request
import platform

def download_pytorch_wheels():
    """下载PyTorch wheel文件"""
    print("📥 下载PyTorch wheel文件...")
    
    # Python 3.13兼容的wheel文件URL
    wheels = [
        {
            "name": "torch-2.7.0+cu121-cp313-cp313-win_amd64.whl",
            "url": "https://download.pytorch.org/whl/cu121/torch-2.7.0%2Bcu121-cp313-cp313-win_amd64.whl"
        },
        {
            "name": "torchvision-0.22.0+cu121-cp313-cp313-win_amd64.whl", 
            "url": "https://download.pytorch.org/whl/cu121/torchvision-0.22.0%2Bcu121-cp313-cp313-win_amd64.whl"
        },
        {
            "name": "torchaudio-2.7.0+cu121-cp313-cp313-win_amd64.whl",
            "url": "https://download.pytorch.org/whl/cu121/torchaudio-2.7.0%2Bcu121-cp313-cp313-win_amd64.whl"
        }
    ]
    
    downloaded_files = []
    
    for wheel in wheels:
        try:
            print(f"正在下载: {wheel['name']}")
            urllib.request.urlretrieve(wheel['url'], wheel['name'])
            downloaded_files.append(wheel['name'])
            print(f"✅ {wheel['name']} 下载完成")
        except Exception as e:
            print(f"❌ {wheel['name']} 下载失败: {e}")
    
    return downloaded_files

def install_wheels(wheel_files):
    """安装wheel文件"""
    print("\n📦 安装wheel文件...")
    
    for wheel_file in wheel_files:
        try:
            print(f"正在安装: {wheel_file}")
            subprocess.run([sys.executable, "-m", "pip", "install", wheel_file, "--force-reinstall"], check=True)
            print(f"✅ {wheel_file} 安装完成")
        except subprocess.CalledProcessError as e:
            print(f"❌ {wheel_file} 安装失败: {e}")

def test_installation():
    """测试安装"""
    print("\n🧪 测试安装...")
    
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

def create_gpu_training_script():
    """创建GPU训练脚本"""
    script_content = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python 3.13 GPU训练脚本
"""

import torch
import sys
import os

def main():
    print("🚀 Python 3.13 GPU训练")
    print("="*50)
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
        
        # 设置GPU优化
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # 运行训练
        os.system("python run_behavior_classification.py --full-pipeline --config python313_gpu_config.json --mixed-precision --batch-size 16")
    else:
        print("❌ GPU不可用，使用CPU模式")
        os.system("python run_behavior_classification.py --full-pipeline --config python313_gpu_config.json --no-gpu --batch-size 8")

if __name__ == "__main__":
    main()
'''
    
    with open("run_gpu_training.py", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✅ GPU训练脚本创建完成")

def main():
    print("="*60)
    print("Python 3.13 PyTorch GPU本地安装")
    print("="*60)
    
    # 检查Python版本
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Python版本: {python_version}")
    
    if python_version != "3.13":
        print("⚠️ 此脚本专为Python 3.13设计")
    
    # 下载wheel文件
    wheel_files = download_pytorch_wheels()
    
    if wheel_files:
        # 安装wheel文件
        install_wheels(wheel_files)
        
        # 测试安装
        if test_installation():
            print("\n🎉 PyTorch GPU版本安装成功！")
            create_gpu_training_script()
            print("\n📋 使用方法:")
            print("python run_gpu_training.py")
        else:
            print("\n❌ 安装完成但GPU不可用")
    else:
        print("\n❌ 无法下载wheel文件")
        print("可能的原因:")
        print("1. 网络连接问题")
        print("2. 防火墙阻止")
        print("3. PyTorch官方暂不支持Python 3.13")
    
    # 清理下载的文件
    for wheel_file in wheel_files:
        try:
            os.remove(wheel_file)
        except:
            pass

if __name__ == "__main__":
    main()
