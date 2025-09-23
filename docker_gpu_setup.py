#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Docker GPU训练环境设置脚本
创建一个包含GPU支持的Docker环境来运行训练
"""

import os
import subprocess
import sys

def create_dockerfile():
    """创建Dockerfile"""
    dockerfile_content = """
# 使用官方PyTorch镜像，包含CUDA支持
FROM pytorch/pytorch:2.7.0-cuda12.1-cudnn8-devel

# 设置工作目录
WORKDIR /app

# 安装必要的Python包
RUN pip install --no-cache-dir \\
    numpy pandas matplotlib seaborn \\
    scikit-learn tqdm \\
    jupyter notebook

# 复制项目文件
COPY . /app/

# 设置环境变量
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# 暴露端口（如果需要Jupyter）
EXPOSE 8888

# 默认命令
CMD ["python", "run_behavior_classification.py", "--full-pipeline"]
"""
    
    with open("Dockerfile", "w", encoding="utf-8") as f:
        f.write(dockerfile_content)
    print("✅ Dockerfile 创建完成")

def create_docker_compose():
    """创建docker-compose.yml"""
    compose_content = """
version: '3.8'

services:
  gpu-training:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    volumes:
      - .:/app
      - ./results:/app/results
    command: python run_behavior_classification.py --full-pipeline --mixed-precision --batch-size 32
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""
    
    with open("docker-compose.yml", "w", encoding="utf-8") as f:
        f.write(compose_content)
    print("✅ docker-compose.yml 创建完成")

def create_gpu_config():
    """创建GPU训练配置"""
    config_content = """{
  "data_dir": "./hyx_data",
  "batch_size": 32,
  "learning_rate": 1e-4,
  "num_epochs": 100,
  "patience": 15,
  "k_folds": 5,
  "test_size": 0.2,
  "model_types": ["cnn", "transformer", "fusion"],
  "use_class_weights": true,
  "augmentation": true,
  "feature_types": ["stft", "time_series"],
  "use_gpu": true,
  "device_id": 0,
  "use_amp": true,
  "compile_model": true,
  "multi_gpu": false,
  "device_count": 1,
  "dataloader_params": {
    "num_workers": 8,
    "pin_memory": true,
    "persistent_workers": true,
    "prefetch_factor": 4,
    "drop_last": true
  }
}"""
    
    with open("gpu_config.json", "w", encoding="utf-8") as f:
        f.write(config_content)
    print("✅ GPU配置文件创建完成")

def create_run_script():
    """创建运行脚本"""
    run_script_content = """#!/bin/bash
# GPU训练运行脚本

echo "🚀 启动GPU训练环境..."

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker Desktop"
    exit 1
fi

# 检查NVIDIA Docker支持
if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Docker支持未配置，请安装nvidia-docker2"
    exit 1
fi

# 构建Docker镜像
echo "📦 构建Docker镜像..."
docker build -t swallow-gpu-training .

# 运行GPU训练
echo "🎯 开始GPU训练..."
docker run --rm --gpus all \\
    -v $(pwd):/app \\
    -v $(pwd)/results:/app/results \\
    swallow-gpu-training \\
    python run_behavior_classification.py \\
    --full-pipeline \\
    --config gpu_config.json \\
    --mixed-precision \\
    --batch-size 32

echo "✅ GPU训练完成！"
"""
    
    with open("run_gpu_training.sh", "w", encoding="utf-8") as f:
        f.write(run_script_content)
    
    # 在Windows上创建批处理文件
    bat_content = """@echo off
echo 🚀 启动GPU训练环境...

REM 检查Docker是否安装
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker未安装，请先安装Docker Desktop
    pause
    exit /b 1
)

REM 构建Docker镜像
echo 📦 构建Docker镜像...
docker build -t swallow-gpu-training .

REM 运行GPU训练
echo 🎯 开始GPU训练...
docker run --rm --gpus all ^
    -v %cd%:/app ^
    -v %cd%/results:/app/results ^
    swallow-gpu-training ^
    python run_behavior_classification.py ^
    --full-pipeline ^
    --config gpu_config.json ^
    --mixed-precision ^
    --batch-size 32

echo ✅ GPU训练完成！
pause
"""
    
    with open("run_gpu_training.bat", "w", encoding="utf-8") as f:
        f.write(bat_content)
    
    print("✅ 运行脚本创建完成")

def main():
    print("="*60)
    print("Docker GPU训练环境设置")
    print("="*60)
    
    print("正在创建Docker配置文件...")
    create_dockerfile()
    create_docker_compose()
    create_gpu_config()
    create_run_script()
    
    print("\n" + "="*60)
    print("🎉 Docker GPU环境设置完成！")
    print("="*60)
    
    print("\n📋 使用步骤：")
    print("1. 安装Docker Desktop")
    print("2. 安装NVIDIA Container Toolkit")
    print("3. 运行: run_gpu_training.bat")
    
    print("\n🔧 手动运行命令：")
    print("docker build -t swallow-gpu-training .")
    print("docker run --rm --gpus all -v %cd%:/app swallow-gpu-training python run_behavior_classification.py --full-pipeline --mixed-precision")
    
    print("\n📁 创建的文件：")
    print("- Dockerfile")
    print("- docker-compose.yml") 
    print("- gpu_config.json")
    print("- run_gpu_training.bat")
    print("- run_gpu_training.sh")

if __name__ == "__main__":
    main()

