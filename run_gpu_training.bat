@echo off
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
