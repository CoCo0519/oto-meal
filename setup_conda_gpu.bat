@echo off
echo ============================================================
echo Python 3.11 GPU训练环境设置
echo ============================================================

echo 🐍 创建Python 3.11 conda环境...
call conda create -n swallow-gpu python=3.11 -y

echo 📦 安装PyTorch GPU版本...
call conda activate swallow-gpu
call conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1 -y

echo 📦 安装其他依赖...
call conda install numpy pandas matplotlib scipy scikit-learn seaborn tqdm jupyter ipykernel -y

echo 🧪 测试GPU功能...
call python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"

echo ✅ 环境设置完成！
echo.
echo 📋 使用方法:
echo 1. 激活环境: conda activate swallow-gpu
echo 2. 运行训练: python run_behavior_classification.py --full-pipeline --mixed-precision
echo.
pause
