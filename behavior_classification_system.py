# -*- coding: utf-8 -*-
"""
基于耳道PPG&IMU信号的行为分类系统 (静息/咀嚼/咳嗽/吞咽)
使用喉咙PPG&IMU作为标注数据，训练耳道信号的深度学习分类器

技术方案：
1. 特征提取：STFT + MFCC (每个通道独立处理)
2. 深度学习模型：CNN ResNet + Transformer（本文件演示 CNN）
3. 四分类：0-静息, 1-咀嚼, 2-咳嗽, 3-吞咽
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import stft
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm

# ---- PyTorch 导入（修复：显式导入 optim，并在无 Torch 环境下给出兜底提示） ----
TORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim                    # ✅ 修复点：显式导入 optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch 未安装，本模块的训练功能不可用（仅数据预处理/特征提取可运行）。")
    # 仅为避免解释期 NameError 放置的占位符；训练将无法进行
    class Dataset: ...
    class DataLoader: ...
    class nn:
        class Module: ...
        class Linear: ...
        class Conv1d: ...
        class Conv2d: ...
        class BatchNorm1d: ...
        class BatchNorm2d: ...
        class Dropout: ...
        class ReLU: ...
        class MaxPool1d: ...
        class MaxPool2d: ...
        class AdaptiveAvgPool1d: ...
        class AdaptiveAvgPool2d: ...
    class F:
        @staticmethod
        def relu(x): return x
        @staticmethod
        def max_pool1d(x, kernel_size): return x
        @staticmethod
        def adaptive_avg_pool1d(x, output_size): return x

# 导入现有的预处理模块
from anc_template_pipeline_100Hz import (
    notch_filter, butter_highpass, butter_bandpass
)

# --------------------------- 配置 ---------------------------

class BehaviorClassificationConfig:
    """配置参数类"""
    # 数据参数
    FS = 100  # 采样率
    WINDOW_SIZE = 5.0  # 窗口大小(秒)
    OVERLAP = 0.5  # 重叠比例

    # 特征提取参数
    STFT_NPERSEG = 256  # STFT窗口长度
    STFT_NOVERLAP = 128  # STFT重叠长度
    MFCC_N_MFCC = 13  # MFCC系数个数
    MFCC_N_FFT = 512  # MFCC FFT长度
    MFCC_N_MELS = 26  # Mel滤波器组数量

    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100

    # 设备/AMP
    if TORCH_AVAILABLE:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        USE_AMP = bool(torch.cuda.is_available())
    else:
        DEVICE = 'cpu'
        USE_AMP = False

    # 分类标签
    BEHAVIOR_LABELS = {
        'rest': 0,      # 静息
        'chew': 1,      # 咀嚼
        'cough': 2,     # 咳嗽
        'swallow': 3    # 吞咽
    }

    # 数据路径映射
    BEHAVIOR_MAPPING = {
        'cough': '咳嗽',
        'swallow': '吞咽',
        'chew': '咀嚼',
        'drink': '喝水'  # 将喝水归类为吞咽
    }

# --------------------------- 特征提取 ---------------------------

class FeatureExtractor:
    """特征提取器：STFT + MFCC"""

    def __init__(self, fs=100):
        self.fs = fs
        self.config = BehaviorClassificationConfig()

    def extract_stft_features(self, signal_data: np.ndarray) -> np.ndarray:
        """
        提取STFT特征
        Args:
            signal_data: (N, 6) - PPG(3) + IMU(3)
        Returns:
            (6, freq_bins, time_bins)
        """
        stft_features = []
        for ch in range(signal_data.shape[1]):
            f, t, Zxx = stft(
                signal_data[:, ch],
                fs=self.fs,
                nperseg=self.config.STFT_NPERSEG,
                noverlap=self.config.STFT_NOVERLAP,
                padded=False, boundary=None
            )
            mag = np.abs(Zxx)
            log_mag = np.log(mag + 1e-8)
            stft_features.append(log_mag)
        return np.asarray(stft_features, dtype=np.float32)

    def extract_mfcc_features(self, signal_data: np.ndarray) -> np.ndarray:
        """
        简化版 MFCC（基于 PSD + Mel 滤波 + DCT）
        Returns:
            (6, n_mfcc, time_frames)
        """
        mfcc_features = []
        for ch in range(signal_data.shape[1]):
            f, psd = signal.welch(
                signal_data[:, ch],
                fs=self.fs,
                nperseg=self.config.MFCC_N_FFT // 2
            )
            mel_filters = self._mel_filter_bank(f, self.config.MFCC_N_MELS)
            mel_spec = np.dot(mel_filters, psd)
            log_mel_spec = np.log(mel_spec + 1e-8)
            mfcc = self._dct(log_mel_spec)[: self.config.MFCC_N_MFCC]
            time_frames = max(1, signal_data.shape[0] // (self.fs // 10))  # 10 Hz 帧率
            mfcc_time = np.tile(mfcc[:, None], (1, time_frames))
            mfcc_features.append(mfcc_time.astype(np.float32))
        return np.asarray(mfcc_features, dtype=np.float32)

    def _mel_filter_bank(self, freqs: np.ndarray, n_mels: int) -> np.ndarray:
        """生成 Mel 滤波器组"""
        def hz_to_mel(hz): return 2595 * np.log10(1 + hz / 700)
        def mel_to_hz(mel): return 700 * (10**(mel / 2595) - 1)

        mel_min = hz_to_mel(freqs[0])
        mel_max = hz_to_mel(freqs[-1])
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        filters = np.zeros((n_mels, len(freqs)), dtype=np.float32)
        for i in range(n_mels):
            left, center, right = hz_points[i], hz_points[i+1], hz_points[i+2]
            for j, f in enumerate(freqs):
                if left <= f <= center:
                    filters[i, j] = (f - left) / max(1e-8, (center - left))
                elif center <= f <= right:
                    filters[i, j] = (right - f) / max(1e-8, (right - center))
        return filters

    def _dct(self, x: np.ndarray) -> np.ndarray:
        """简化 DCT-II"""
        N = len(x); result = np.zeros(N, dtype=np.float32)
        for k in range(N):
            ck = 0.0
            for n in range(N):
                ck += x[n] * np.cos(np.pi * k * (2*n + 1) / (2*N))
            result[k] = ck * (np.sqrt(1/(4*N)) if k == 0 else np.sqrt(1/(2*N)))
        return result

# --------------------------- 数据集 ---------------------------

class BehaviorDataset(Dataset):
    """行为分类数据集"""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        assert TORCH_AVAILABLE, "需要安装 PyTorch 才能使用数据集与训练功能。"
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# --------------------------- 模型 ---------------------------

class ResNetBlock(nn.Module):
    """ResNet基本块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.short = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.short = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.short(x)
        return F.relu(out)

class BehaviorCNN(nn.Module):
    """基于ResNet的行为分类CNN"""
    def __init__(self, input_shape, num_classes=4):
        super().__init__()
        # input_shape: (C, H, W)
        self.input_conv = nn.Conv2d(input_shape[0], 64, 7, 2, 3, bias=False)
        self.input_bn   = nn.BatchNorm2d(64)
        self.maxpool    = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(64,  64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool   = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = [ResNetBlock(in_ch, out_ch, stride)]
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# （可选）Transformer 结构留存
class TransformerEncoder(nn.Module):
    """Transformer编码器用于时间序列分类（示例保留）"""
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=6, num_classes=4):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048, dropout=0.1, batch_first=False
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(d_model, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, x):  # x: (B, T, F)
        T = x.size(1)
        x = self.input_projection(x) + self.pos_encoding[:T, :].unsqueeze(0)
        x = x.transpose(0, 1)              # (T, B, D)
        x = self.transformer(x).mean(0)    # (B, D)
        return self.classifier(x)

# --------------------------- 主流程 ---------------------------

class BehaviorClassifier:
    """行为分类主类"""

    def __init__(self, data_dir='./hyx_data'):
        self.data_dir = data_dir
        self.config = BehaviorClassificationConfig()
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    # ---------- 数据处理 ----------

    def load_and_preprocess_data(self):
        """加载并预处理数据"""
        print("正在加载和预处理数据...")

        all_windows = []
        all_labels  = []

        for filename in os.listdir(self.data_dir):
            if not filename.endswith('.txt'):
                continue

            label = self._parse_behavior_from_filename(filename)
            if label is None:
                continue

            filepath = os.path.join(self.data_dir, filename)
            data = self._safe_load_txt(filepath)
            if data is None or data.shape[1] < 6:
                print(f"⚠️ 数据维度不足，跳过：{filename}")
                continue

            processed = self._preprocess_signals(data)
            windows, labels = self._create_windows(processed, label)
            all_windows.extend(windows)
            all_labels.extend(labels)

        print(f"总共加载了 {len(all_windows)} 个样本")
        return np.asarray(all_windows), np.asarray(all_labels, dtype=np.int64)

    def _safe_load_txt(self, path: str):
        """兼容不同编码读取 txt"""
        try:
            # 部分 numpy 版本不支持 encoding 参数，改为手动打开文件对象
            with open(path, 'r', encoding='utf-8') as f:
                return np.loadtxt(f, skiprows=1)
        except Exception:
            try:
                with open(path, 'r', encoding='gbk') as f:
                    return np.loadtxt(f, skiprows=1)
            except Exception as e:
                print(f"无法读取文件：{os.path.basename(path)} ({e})")
                return None

    def _parse_behavior_from_filename(self, filename):
        """从文件名解析行为标签（仅耳道数据）"""
        if '耳道' not in filename:
            return None
        for behavior_en, behavior_cn in self.config.BEHAVIOR_MAPPING.items():
            if behavior_cn in filename:
                if behavior_en == 'drink':
                    return self.config.BEHAVIOR_LABELS['swallow']
                return self.config.BEHAVIOR_LABELS[behavior_en]
        return self.config.BEHAVIOR_LABELS['rest']

    def _preprocess_signals(self, data: np.ndarray) -> np.ndarray:
        """信号预处理：PPG 陷波+高通；IMU 高通+带通"""
        ppg_green, ppg_ir, ppg_red = data[:, 0], data[:, 1], data[:, 2]
        acc_raw = data[:, 3:6]

        ppg_processed = []
        for ppg in [ppg_green, ppg_ir, ppg_red]:
            ppg_nf = notch_filter(ppg, self.config.FS, mains=50, Q=30)
            ppg_hp = butter_highpass(ppg_nf, self.config.FS, fc=0.1, order=2)
            ppg_processed.append(ppg_hp)

        acc_hp = butter_highpass(acc_raw, self.config.FS, fc=0.3, order=2)
        acc_bp = butter_bandpass(acc_hp, self.config.FS, 0.5, 15.0, order=3)

        processed = np.column_stack([
            ppg_processed[0], ppg_processed[1], ppg_processed[2],
            acc_bp[:, 0], acc_bp[:, 1], acc_bp[:, 2]
        ]).astype(np.float32)
        return processed

    def _create_windows(self, data: np.ndarray, label: int):
        """创建滑动窗口"""
        win = int(self.config.WINDOW_SIZE * self.config.FS)
        stride = int(win * (1 - self.config.OVERLAP))
        windows, labels = [], []
        for start in range(0, len(data) - win + 1, stride):
            seg = data[start:start+win]
            windows.append(seg)
            labels.append(label)
        return windows, labels

    # ---------- 特征 ----------

    def extract_features(self, data_windows: np.ndarray):
        """提取 STFT 与 MFCC 特征"""
        print("正在提取STFT和MFCC特征...")
        stft_features, mfcc_features = [], []
        for win in tqdm(data_windows):
            stft_features.append(self.feature_extractor.extract_stft_features(win))
            mfcc_features.append(self.feature_extractor.extract_mfcc_features(win))
        return np.asarray(stft_features), np.asarray(mfcc_features)

    # ---------- 训练 ----------

    def _dataloader_kwargs(self):
        """根据设备给 DataLoader 合理的默认参数"""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            num_workers = min(os.cpu_count() or 2, 8)
            return dict(batch_size=self.config.BATCH_SIZE, shuffle=True,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=num_workers > 0,
                        prefetch_factor=4 if num_workers > 0 else None)
        else:
            return dict(batch_size=self.config.BATCH_SIZE, shuffle=True,
                        num_workers=max(1, (os.cpu_count() or 2)//2),
                        pin_memory=False)

    def train_cnn_model(self, features: np.ndarray, labels: np.ndarray):
        """训练 CNN 模型（支持 AMP）"""
        assert TORCH_AVAILABLE, "需要安装 PyTorch 才能训练。"
        print("正在训练CNN ResNet模型...")

        # 输入形状：features -> (N, C, H, W)
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )

        train_loader = DataLoader(BehaviorDataset(X_train, y_train), **self._dataloader_kwargs())
        test_loader  = DataLoader(BehaviorDataset(X_test,  y_test),
                                  batch_size=self.config.BATCH_SIZE, shuffle=False,
                                  num_workers=self._dataloader_kwargs().get('num_workers', 0),
                                  pin_memory=self._dataloader_kwargs().get('pin_memory', False),
                                  persistent_workers=self._dataloader_kwargs().get('persistent_workers', False))

        input_shape = features.shape[1:]  # (C,H,W)
        model = BehaviorCNN(input_shape, num_classes=4).to(self.config.DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)   # ✅ 修复点：optim 已显式导入
        criterion = nn.CrossEntropyLoss()

        scaler = torch.cuda.amp.GradScaler(enabled=self.config.USE_AMP)  # 2.8 仍兼容 GradScaler

        train_losses, train_accuracies = [], []
        for epoch in range(self.config.NUM_EPOCHS):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.config.DEVICE, non_blocking=True)
                batch_y = batch_y.to(self.config.DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                # AMP 前向
                if hasattr(torch.amp, 'autocast'):
                    with torch.amp.autocast('cuda', enabled=self.config.USE_AMP):
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

                # AMP 反向
                if self.config.USE_AMP:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += float(loss)
                pred = outputs.argmax(dim=1)
                total += batch_y.size(0)
                correct += (pred == batch_y).sum().item()

            train_loss = total_loss / max(1, len(train_loader))
            train_acc = 100.0 * correct / max(1, total)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}]  Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%")

        # 测试评估
        model.eval()
        test_correct = 0
        test_total = 0
        all_pred, all_true = [], []

        with torch.inference_mode():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.config.DEVICE, non_blocking=True)
                batch_y = batch_y.to(self.config.DEVICE, non_blocking=True)
                logits = model(batch_x)
                pred = logits.argmax(dim=1)
                test_total += batch_y.size(0)
                test_correct += (pred == batch_y).sum().item()
                all_pred.extend(pred.cpu().numpy().tolist())
                all_true.extend(batch_y.cpu().numpy().tolist())

        test_acc = 100.0 * test_correct / max(1, test_total)
        print(f'测试准确率: {test_acc:.2f}%')

        return model, train_losses, train_accuracies, all_pred, all_true

    # ---------- 可视化 ----------

    def visualize_results(self, train_losses, train_accuracies, predictions, true_labels):
        """可视化训练与评估结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        axes[0, 0].plot(train_losses)
        axes[0, 0].set_title('训练损失'); axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')

        axes[0, 1].plot(train_accuracies)
        axes[0, 1].set_title('训练准确率'); axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy (%)')

        cm = confusion_matrix(true_labels, predictions, labels=[0, 1, 2, 3])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0],
                    xticklabels=['静息', '咀嚼', '咳嗽', '吞咽'],
                    yticklabels=['静息', '咀嚼', '咳嗽', '吞咽'])
        axes[1, 0].set_title('混淆矩阵')

        behaviors = ['静息', '咀嚼', '咳嗽', '吞咽']
        report = classification_report(true_labels, predictions,
                                       labels=[0, 1, 2, 3],
                                       target_names=behaviors,
                                       output_dict=True, zero_division=0)
        f1_scores = [report.get(name, {}).get('f1-score', 0.0) for name in behaviors]
        axes[1, 1].bar(behaviors, f1_scores)
        axes[1, 1].set_title('各行为F1分数'); axes[1, 1].set_ylabel('F1 Score'); axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('behavior_classification_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n详细分类报告:")
        print(classification_report(true_labels, predictions,
                                    labels=[0, 1, 2, 3],
                                    target_names=behaviors,
                                    zero_division=0))

# --------------------------- 程序入口 ---------------------------

def main():
    """主函数"""
    print("=== 基于耳道PPG&IMU的行为分类系统 ===")
    print("目标：静息/咀嚼/咳嗽/吞咽 四分类")

    classifier = BehaviorClassifier()

    # 加载 & 预处理
    data_windows, labels = classifier.load_and_preprocess_data()

    # 特征
    stft_features, mfcc_features = classifier.extract_features(data_windows)

    # 使用 STFT 特征训练 CNN
    print("\n使用STFT特征训练CNN ResNet模型...")
    model, train_losses, train_accs, preds, trues = classifier.train_cnn_model(stft_features, labels)

    # 可视化
    classifier.visualize_results(train_losses, train_accs, preds, trues)

    # 保存模型
    if TORCH_AVAILABLE:
        torch.save(model.state_dict(), 'behavior_cnn_model.pth')
        print("\n模型已保存为 behavior_cnn_model.pth")

    print("\n=== 训练完成 ===")

if __name__ == "__main__":
    main()
