
# 使用官方PyTorch镜像，包含CUDA支持
FROM pytorch/pytorch:2.7.0-cuda12.1-cudnn8-devel

# 设置工作目录
WORKDIR /app

# 安装必要的Python包
RUN pip install --no-cache-dir \
    numpy pandas matplotlib seaborn \
    scikit-learn tqdm \
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
