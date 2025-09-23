#!/bin/bash
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
docker run --rm --gpus all \
    -v $(pwd):/app \
    -v $(pwd)/results:/app/results \
    swallow-gpu-training \
    python run_behavior_classification.py \
    --full-pipeline \
    --config gpu_config.json \
    --mixed-precision \
    --batch-size 32

echo "✅ GPU训练完成！"
