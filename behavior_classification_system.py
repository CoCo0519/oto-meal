# -*- coding: utf-8 -*-
"""
基于耳道PPG&IMU信号的行为分类系统 (静息/咀嚼/咳嗽/吞咽)
使用喉咙PPG&IMU作为标注数据，训练耳道信号的深度学习分类器

技术方案：
1. 特征提取：STFT + MFCC (每个通道独立处理)
2. 深度学习模型：CNN ResNet + Transformer
3. 四分类：0-静息, 1-咀嚼, 2-咳嗽, 3-吞咽

作者：基于Project-Swallow项目扩展
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import stft
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm

# 导入现有的预处理模块
from anc_template_pipeline_100Hz import (
    notch_filter, butter_highpass, butter_bandpass
)

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
    
    # 模型参数
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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

class FeatureExtractor:
    """特征提取器：STFT + MFCC"""
    
    def __init__(self, fs=100):
        self.fs = fs
        self.config = BehaviorClassificationConfig()
        
    def extract_stft_features(self, signal_data):
        """
        提取STFT特征
        Args:
            signal_data: (N, 6) - PPG(3) + IMU(3)
        Returns:
            stft_features: (6, freq_bins, time_bins) 
        """
        stft_features = []
        
        for channel in range(signal_data.shape[1]):
            # 对每个通道计算STFT
            f, t, Zxx = stft(
                signal_data[:, channel], 
                fs=self.fs,
                nperseg=self.config.STFT_NPERSEG,
                noverlap=self.config.STFT_NOVERLAP
            )
            
            # 取幅度谱的对数
            magnitude = np.abs(Zxx)
            log_magnitude = np.log(magnitude + 1e-8)
            stft_features.append(log_magnitude)
            
        return np.array(stft_features)  # (6, freq_bins, time_bins)
    
    def extract_mfcc_features(self, signal_data):
        """
        提取MFCC特征 (简化版本，适用于生理信号)
        Args:
            signal_data: (N, 6)
        Returns:
            mfcc_features: (6, n_mfcc, time_frames)
        """
        mfcc_features = []
        
        for channel in range(signal_data.shape[1]):
            # 计算功率谱密度
            f, psd = signal.welch(
                signal_data[:, channel], 
                fs=self.fs, 
                nperseg=self.config.MFCC_N_FFT//2
            )
            
            # Mel滤波器组
            mel_filters = self._mel_filter_bank(f, self.config.MFCC_N_MELS)
            
            # 应用Mel滤波器
            mel_spec = np.dot(mel_filters, psd)
            log_mel_spec = np.log(mel_spec + 1e-8)
            
            # DCT变换得到MFCC
            mfcc = self._dct(log_mel_spec)[:self.config.MFCC_N_MFCC]
            
            # 扩展为时间序列（简化处理）
            time_frames = signal_data.shape[0] // (self.fs // 10)  # 10Hz帧率
            mfcc_time = np.tile(mfcc[:, np.newaxis], (1, time_frames))
            
            mfcc_features.append(mfcc_time)
            
        return np.array(mfcc_features)  # (6, n_mfcc, time_frames)
    
    def _mel_filter_bank(self, freqs, n_mels):
        """生成Mel滤波器组"""
        # Mel尺度转换
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)
        
        def mel_to_hz(mel):
            return 700 * (10**(mel / 2595) - 1)
        
        # Mel频率点
        mel_min = hz_to_mel(freqs[0])
        mel_max = hz_to_mel(freqs[-1])
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        
        # 构建滤波器
        filters = np.zeros((n_mels, len(freqs)))
        for i in range(n_mels):
            left = hz_points[i]
            center = hz_points[i + 1] 
            right = hz_points[i + 2]
            
            for j, f in enumerate(freqs):
                if left <= f <= center:
                    filters[i, j] = (f - left) / (center - left)
                elif center <= f <= right:
                    filters[i, j] = (right - f) / (right - center)
                    
        return filters
    
    def _dct(self, x):
        """离散余弦变换"""
        N = len(x)
        result = np.zeros(N)
        for k in range(N):
            for n in range(N):
                result[k] += x[n] * np.cos(np.pi * k * (2*n + 1) / (2*N))
            if k == 0:
                result[k] *= np.sqrt(1/(4*N))
            else:
                result[k] *= np.sqrt(1/(2*N))
        return result

class BehaviorDataset(Dataset):
    """行为分类数据集"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class ResNetBlock(nn.Module):
    """ResNet基本块"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BehaviorCNN(nn.Module):
    """基于ResNet的行为分类CNN"""
    
    def __init__(self, input_shape, num_classes=4):
        super(BehaviorCNN, self).__init__()
        self.input_shape = input_shape  # (channels, height, width)
        
        # 输入适配层
        self.input_conv = nn.Conv2d(input_shape[0], 64, 7, 2, 3, bias=False)
        self.input_bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # ResNet层
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x) 
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

class TransformerEncoder(nn.Module):
    """Transformer编码器用于时间序列分类"""
    
    def __init__(self, input_dim, d_model=512, nhead=8, num_layers=6, num_classes=4):
        super(TransformerEncoder, self).__init__()
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=2048,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        seq_len = x.size(1)
        
        # 输入投影 + 位置编码
        x = self.input_projection(x)
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer编码 (需要转置为 seq_len, batch, d_model)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        
        # 全局平均池化 + 分类
        x = x.mean(dim=0)  # (batch, d_model)
        x = self.classifier(x)
        
        return x

class BehaviorClassifier:
    """行为分类主类"""
    
    def __init__(self, data_dir='./hyx_data'):
        self.data_dir = data_dir
        self.config = BehaviorClassificationConfig()
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self):
        """加载并预处理数据"""
        print("正在加载和预处理数据...")
        
        all_features = []
        all_labels = []
        
        # 遍历所有数据文件
        for filename in os.listdir(self.data_dir):
            if not filename.endswith('.txt'):
                continue
                
            # 解析文件名获取行为标签
            label = self._parse_behavior_from_filename(filename)
            if label is None:
                continue
                
            # 加载数据
            filepath = os.path.join(self.data_dir, filename)
            try:
                data = np.loadtxt(filepath, skiprows=1, encoding='utf-8')
            except:
                try:
                    data = np.loadtxt(filepath, skiprows=1, encoding='gbk')
                except:
                    print(f"无法读取文件: {filename}")
                    continue
            
            # 数据预处理（使用现有的预处理流程）
            processed_data = self._preprocess_signals(data)
            
            # 滑动窗口分割
            windows, window_labels = self._create_windows(processed_data, label)
            
            all_features.extend(windows)
            all_labels.extend(window_labels)
            
        print(f"总共加载了 {len(all_features)} 个样本")
        return np.array(all_features), np.array(all_labels)
    
    def _parse_behavior_from_filename(self, filename):
        """从文件名解析行为标签"""
        # 只处理耳道数据
        if '耳道' not in filename:
            return None
            
        for behavior_en, behavior_cn in self.config.BEHAVIOR_MAPPING.items():
            if behavior_cn in filename:
                if behavior_en == 'drink':  # 将喝水归类为吞咽
                    return self.config.BEHAVIOR_LABELS['swallow']
                return self.config.BEHAVIOR_LABELS[behavior_en]
                
        # 默认为静息状态（如果没有明确的行为标识）
        return self.config.BEHAVIOR_LABELS['rest']
    
    def _preprocess_signals(self, data):
        """信号预处理（基于现有流程）"""
        # 提取PPG和IMU数据
        ppg_green = data[:, 0]
        ppg_ir = data[:, 1] 
        ppg_red = data[:, 2]
        acc_raw = data[:, 3:6]
        
        # PPG预处理：工频陷波 + 高通
        ppg_processed = []
        for ppg_channel in [ppg_green, ppg_ir, ppg_red]:
            ppg_nf = notch_filter(ppg_channel, self.config.FS, mains=50, Q=30)
            ppg_hp = butter_highpass(ppg_nf, self.config.FS, fc=0.1, order=2)
            ppg_processed.append(ppg_hp)
        
        # IMU预处理：高通去重力 + 带通滤波
        acc_hp = butter_highpass(acc_raw, self.config.FS, fc=0.3, order=2)
        acc_bp = butter_bandpass(acc_hp, self.config.FS, 0.5, 15.0, order=3)
        
        # 合并为6通道数据
        processed_data = np.column_stack([
            ppg_processed[0], ppg_processed[1], ppg_processed[2],  # PPG 3通道
            acc_bp[:, 0], acc_bp[:, 1], acc_bp[:, 2]  # IMU 3通道
        ])
        
        return processed_data
    
    def _create_windows(self, data, label):
        """创建滑动窗口"""
        window_samples = int(self.config.WINDOW_SIZE * self.config.FS)
        stride_samples = int(window_samples * (1 - self.config.OVERLAP))
        
        windows = []
        labels = []
        
        for start in range(0, len(data) - window_samples + 1, stride_samples):
            end = start + window_samples
            window_data = data[start:end]
            
            windows.append(window_data)
            labels.append(label)
            
        return windows, labels
    
    def extract_features(self, data_windows):
        """提取特征"""
        print("正在提取STFT和MFCC特征...")
        
        stft_features = []
        mfcc_features = []
        
        for window_data in tqdm(data_windows):
            # STFT特征
            stft_feat = self.feature_extractor.extract_stft_features(window_data)
            stft_features.append(stft_feat)
            
            # MFCC特征  
            mfcc_feat = self.feature_extractor.extract_mfcc_features(window_data)
            mfcc_features.append(mfcc_feat)
            
        return np.array(stft_features), np.array(mfcc_features)
    
    def train_cnn_model(self, features, labels):
        """训练CNN模型"""
        print("正在训练CNN ResNet模型...")
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # 创建数据加载器
        train_dataset = BehaviorDataset(X_train, y_train)
        test_dataset = BehaviorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        # 创建模型
        input_shape = features.shape[1:]  # (channels, height, width)
        model = BehaviorCNN(input_shape, num_classes=4).to(self.config.DEVICE)
        
        # 优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        train_losses = []
        train_accuracies = []
        
        for epoch in range(self.config.NUM_EPOCHS):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.config.DEVICE)
                batch_labels = batch_labels.to(self.config.DEVICE)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels).sum().item()
            
            train_loss = total_loss / len(train_loader)
            train_acc = 100. * correct / total
            
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.config.NUM_EPOCHS}], Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        
        # 测试评估
        model.eval()
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_true_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(self.config.DEVICE)
                batch_labels = batch_labels.to(self.config.DEVICE)
                
                outputs = model(batch_features)
                _, predicted = outputs.max(1)
                
                test_total += batch_labels.size(0)
                test_correct += predicted.eq(batch_labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_true_labels.extend(batch_labels.cpu().numpy())
        
        test_acc = 100. * test_correct / test_total
        print(f'测试准确率: {test_acc:.2f}%')
        
        return model, train_losses, train_accuracies, all_predictions, all_true_labels
    
    def visualize_results(self, train_losses, train_accuracies, predictions, true_labels):
        """可视化结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练损失曲线
        axes[0, 0].plot(train_losses)
        axes[0, 0].set_title('训练损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        
        # 训练准确率曲线
        axes[0, 1].plot(train_accuracies)
        axes[0, 1].set_title('训练准确率')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        
        # 混淆矩阵
        cm = confusion_matrix(true_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0], 
                   xticklabels=['静息', '咀嚼', '咳嗽', '吞咽'],
                   yticklabels=['静息', '咀嚼', '咳嗽', '吞咽'])
        axes[1, 0].set_title('混淆矩阵')
        
        # 分类报告
        report = classification_report(true_labels, predictions, 
                                     target_names=['静息', '咀嚼', '咳嗽', '吞咽'],
                                     output_dict=True)
        
        # 绘制F1分数
        behaviors = ['静息', '咀嚼', '咳嗽', '吞咽']
        f1_scores = [report[str(i)]['f1-score'] for i in range(4)]
        
        axes[1, 1].bar(behaviors, f1_scores)
        axes[1, 1].set_title('各行为F1分数')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('behavior_classification_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印详细分类报告
        print("\n详细分类报告:")
        print(classification_report(true_labels, predictions, 
                                  target_names=['静息', '咀嚼', '咳嗽', '吞咽']))

def main():
    """主函数"""
    print("=== 基于耳道PPG&IMU的行为分类系统 ===")
    print("目标：静息/咀嚼/咳嗽/吞咽 四分类")
    
    # 初始化分类器
    classifier = BehaviorClassifier()
    
    # 加载和预处理数据
    data_windows, labels = classifier.load_and_preprocess_data()
    
    # 提取特征
    stft_features, mfcc_features = classifier.extract_features(data_windows)
    
    # 使用STFT特征训练CNN模型
    print("\n使用STFT特征训练CNN ResNet模型...")
    cnn_model, train_losses, train_accs, predictions, true_labels = classifier.train_cnn_model(
        stft_features, labels
    )
    
    # 可视化结果
    classifier.visualize_results(train_losses, train_accs, predictions, true_labels)
    
    # 保存模型
    torch.save(cnn_model.state_dict(), 'behavior_cnn_model.pth')
    print("\n模型已保存为 behavior_cnn_model.pth")
    
    print("\n=== 训练完成 ===")

if __name__ == "__main__":
    main()
