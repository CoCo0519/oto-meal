# -*- coding: utf-8 -*-
"""
完整的训练和评估流水线
整合数据标注、特征提取、模型训练、评估和推理的完整工作流

功能模块：
1. 数据加载和预处理
2. 特征提取（STFT + MFCC + 时间序列）
3. 模型训练（CNN + Transformer + 融合模型）
4. 模型评估和可视化
5. 模型推理和部署

作者：基于Project-Swallow项目扩展
"""

import os
import sys
import argparse
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch 未安装，将使用CPU模式")
    TORCH_AVAILABLE = False
    # 创建占位符类
    class Dataset:
        pass
    class DataLoader:
        pass
    class WeightedRandomSampler:
        pass
    class nn:
        class Module:
            pass
        class Linear:
            pass
        class Conv1d:
            pass
        class BatchNorm1d:
            pass
        class Dropout:
            pass
        class ReLU:
            pass
        class MaxPool1d:
            pass
        class AdaptiveAvgPool1d:
            pass
        class TransformerEncoder:
            pass
        class TransformerEncoderLayer:
            pass
        class LayerNorm:
            pass
        class MultiheadAttention:
            pass
        class Sequential:
            pass
    class optim:
        class Adam:
            def __init__(self, *args, **kwargs):
                pass
        class SGD:
            def __init__(self, *args, **kwargs):
                pass
    class CosineAnnealingLR:
        def __init__(self, *args, **kwargs):
            pass
    class ReduceLROnPlateau:
        def __init__(self, *args, **kwargs):
            pass
    class F:
        @staticmethod
        def relu(x):
            return x
        @staticmethod
        def max_pool1d(x, kernel_size):
            return x
        @staticmethod
        def adaptive_avg_pool1d(x, output_size):
            return x

from tqdm import tqdm
import logging

# 导入自定义模块
from data_labeling_system import DataLabeler, BehaviorDetector
from behavior_classification_system import FeatureExtractor, BehaviorClassificationConfig
from advanced_models import (
    AdvancedBehaviorCNN, BehaviorTransformer, MultiModalFusionModel,
    BehaviorClassificationLoss, create_model, model_summary
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultiModalDataset(Dataset):
    """多模态数据集：支持STFT特征和时间序列数据"""
    
    def __init__(self, stft_features, time_series_data, labels, transform=None):
        self.stft_features = torch.FloatTensor(stft_features)
        self.time_series_data = torch.FloatTensor(time_series_data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        stft = self.stft_features[idx]
        time_series = self.time_series_data[idx]
        label = self.labels[idx]
        
        if self.transform:
            stft = self.transform(stft)
            
        return stft, time_series, label

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class TrainingPipeline:
    """完整的训练流水线"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # 智能设备选择
        self.device = self._setup_device()
        self.feature_extractor = FeatureExtractor()
        self.scaler_stft = StandardScaler()
        self.scaler_time_series = StandardScaler()
        
        # 创建输出目录
        self.output_dir = f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"训练流水线初始化完成，使用设备: {self.device}")
        logger.info(f"输出目录: {self.output_dir}")
        
        # 初始化混合精度训练
        self.scaler = None
        if self.config.get('use_amp', False) and torch.cuda.is_available():
            try:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
                logger.info("✅ 自动混合精度训练已启用")
            except ImportError:
                logger.warning("⚠️ 无法导入GradScaler，禁用混合精度训练")
                self.config['use_amp'] = False
    
    def _setup_device(self):
        """智能设备设置"""
        if not torch.cuda.is_available():
            return torch.device('cpu')
        
        # 检查是否指定了GPU设备
        device_id = self.config.get('device_id', 0)
        
        # 检查CUDA_VISIBLE_DEVICES环境变量
        if 'CUDA_VISIBLE_DEVICES' in os.environ:
            visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
            if visible_devices:
                device_id = int(visible_devices.split(',')[0])
        
        # 设置设备
        device = torch.device(f'cuda:{device_id}')
        torch.cuda.set_device(device_id)
        
        # 应用GPU优化设置
        if self.config.get('use_gpu', True):
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        
        return device
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置"""
        default_config = {
            'data_dir': './hyx_data',
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 100,
            'patience': 15,
            'k_folds': 5,
            'test_size': 0.2,
            'model_types': ['cnn', 'transformer', 'fusion'],
            'use_class_weights': True,
            'augmentation': True,
            'feature_types': ['stft', 'time_series'],
            'use_gpu': True,
            'device_id': 0,
            'use_amp': False,
            'compile_model': False,
            'multi_gpu': False,
            'device_count': 1,
            'dataloader_params': {
                'num_workers': 4,
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 2,
                'drop_last': False
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            default_config.update(user_config)
            
        return default_config
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """数据准备：标注 + 特征提取"""
        logger.info("开始数据准备阶段...")
        
        # 1. 数据标注
        labeler = DataLabeler(self.config['data_dir'])
        labeled_data = labeler.create_labeled_dataset()
        
        if not labeled_data:
            raise ValueError("未找到可用的标注数据")
        
        # 保存标注数据
        labeled_file = os.path.join(self.output_dir, 'labeled_dataset.npz')
        labeler.save_labeled_dataset(labeled_data, labeled_file)
        
        # 2. 特征提取
        logger.info("开始特征提取...")
        
        all_stft_features = []
        all_time_series = []
        all_labels = []
        
        for sample in tqdm(labeled_data, desc="特征提取"):
            ear_data = sample['ear_data']
            labels = sample['labels']
            
            # 创建滑动窗口
            windows, window_labels = self._create_sliding_windows(ear_data, labels)
            
            for window_data, window_label in zip(windows, window_labels):
                # STFT特征
                stft_feat = self.feature_extractor.extract_stft_features(window_data)
                all_stft_features.append(stft_feat)
                
                # 时间序列数据（原始信号）
                all_time_series.append(window_data)
                all_labels.append(window_label)
        
        stft_features = np.array(all_stft_features)
        time_series_data = np.array(all_time_series)
        labels = np.array(all_labels)
        
        logger.info(f"特征提取完成:")
        logger.info(f"  STFT特征形状: {stft_features.shape}")
        logger.info(f"  时间序列形状: {time_series_data.shape}")
        logger.info(f"  标签分布: {np.bincount(labels)}")
        
        # 3. 特征标准化
        # 重塑STFT特征用于标准化
        stft_reshaped = stft_features.reshape(stft_features.shape[0], -1)
        stft_normalized = self.scaler_stft.fit_transform(stft_reshaped)
        stft_features = stft_normalized.reshape(stft_features.shape)
        
        # 标准化时间序列
        time_series_reshaped = time_series_data.reshape(time_series_data.shape[0], -1)
        time_series_normalized = self.scaler_time_series.fit_transform(time_series_reshaped)
        time_series_data = time_series_normalized.reshape(time_series_data.shape)
        
        # 保存预处理器
        with open(os.path.join(self.output_dir, 'scalers.pkl'), 'wb') as f:
            pickle.dump({
                'stft_scaler': self.scaler_stft,
                'time_series_scaler': self.scaler_time_series
            }, f)
        
        return stft_features, time_series_data, labels
    
    def _create_sliding_windows(self, data: np.ndarray, labels: np.ndarray, 
                               window_size: float = 5.0, overlap: float = 0.5) -> Tuple[List, List]:
        """创建滑动窗口"""
        fs = 100
        window_samples = int(window_size * fs)
        stride_samples = int(window_samples * (1 - overlap))
        
        windows = []
        window_labels = []
        
        for start in range(0, len(data) - window_samples + 1, stride_samples):
            end = start + window_samples
            window_data = data[start:end]
            
            # 窗口标签：取众数
            window_label_counts = np.bincount(labels[start:end])
            window_label = np.argmax(window_label_counts)
            
            windows.append(window_data)
            window_labels.append(window_label)
        
        return windows, window_labels
    
    def create_data_loaders(self, stft_features: np.ndarray, 
                           time_series_data: np.ndarray, 
                           labels: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """创建数据加载器"""
        # 数据分割
        X_stft_train, X_stft_test, X_ts_train, X_ts_test, y_train, y_test = train_test_split(
            stft_features, time_series_data, labels,
            test_size=self.config['test_size'],
            random_state=42,
            stratify=labels
        )
        
        # 创建数据集
        train_dataset = MultiModalDataset(X_stft_train, X_ts_train, y_train)
        test_dataset = MultiModalDataset(X_stft_test, X_ts_test, y_test)
        
        # 类别权重（处理不平衡数据）
        if self.config['use_class_weights']:
            class_counts = np.bincount(y_train)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[y_train]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
        else:
            sampler = None
        
        # 获取GPU优化的数据加载器参数
        dataloader_params = self.config.get('dataloader_params', {})
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=dataloader_params.get('num_workers', 4),
            pin_memory=dataloader_params.get('pin_memory', True),
            persistent_workers=dataloader_params.get('persistent_workers', True),
            prefetch_factor=dataloader_params.get('prefetch_factor', 2),
            drop_last=dataloader_params.get('drop_last', False)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=dataloader_params.get('num_workers', 4),
            pin_memory=dataloader_params.get('pin_memory', True),
            persistent_workers=dataloader_params.get('persistent_workers', True),
            prefetch_factor=dataloader_params.get('prefetch_factor', 2),
            drop_last=False  # 测试时不丢弃数据
        )
        
        return train_loader, test_loader
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   test_loader: DataLoader, model_name: str) -> Dict:
        """训练单个模型"""
        logger.info(f"开始训练 {model_name} 模型...")
        
        # 模型编译优化（如果支持）
        if self.config.get('compile_model', False) and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                logger.info("✅ 模型编译优化已启用")
            except Exception as e:
                logger.warning(f"⚠️ 模型编译失败: {e}")
        
        model = model.to(self.device)
        
        # 多GPU支持
        if self.config.get('multi_gpu', False) and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logger.info(f"✅ 多GPU训练已启用: {torch.cuda.device_count()} 个GPU")
        
        # 损失函数和优化器
        if self.config['use_class_weights']:
            # 计算类别权重
            all_labels = []
            for _, _, labels in train_loader:
                all_labels.extend(labels.numpy())
            class_counts = np.bincount(all_labels)
            class_weights = torch.FloatTensor(1.0 / class_counts).to(self.device)
        else:
            class_weights = None
        
        criterion = BehaviorClassificationLoss(
            alpha=1.0, gamma=2.0, class_weights=class_weights
        )
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=1e-4
        )
        
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config['num_epochs'])
        early_stopping = EarlyStopping(patience=self.config['patience'])
        
        # 训练历史
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(self.config['num_epochs']):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]} [Train]')
            
            for batch_stft, batch_ts, batch_labels in train_pbar:
                batch_stft = batch_stft.to(self.device, non_blocking=True)
                batch_ts = batch_ts.to(self.device, non_blocking=True)
                batch_labels = batch_labels.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # 混合精度训练
                if self.config.get('use_amp', False) and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        # 前向传播（根据模型类型选择输入）
                        if model_name == 'fusion':
                            outputs = model(batch_stft, batch_ts)
                        elif model_name == 'cnn':
                            outputs = model(batch_stft)
                        elif model_name == 'transformer':
                            outputs = model(batch_ts)
                        
                        loss = criterion(outputs, batch_labels)
                    
                    # 反向传播（混合精度）
                    self.scaler.scale(loss).backward()
                    
                    # 梯度裁剪
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # 优化器步骤
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # 标准精度训练
                    # 前向传播（根据模型类型选择输入）
                    if model_name == 'fusion':
                        outputs = model(batch_stft, batch_ts)
                    elif model_name == 'cnn':
                        outputs = model(batch_stft)
                    elif model_name == 'transformer':
                        outputs = model(batch_ts)
                    
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                
                # 统计
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_labels.size(0)
                train_correct += predicted.eq(batch_labels).sum().item()
                
                # 更新进度条
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_predictions = []
            all_true_labels = []
            
            with torch.no_grad():
                val_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]} [Val]')
                
                for batch_stft, batch_ts, batch_labels in val_pbar:
                    batch_stft = batch_stft.to(self.device, non_blocking=True)
                    batch_ts = batch_ts.to(self.device, non_blocking=True)
                    batch_labels = batch_labels.to(self.device, non_blocking=True)
                    
                    # 前向传播（验证时也使用混合精度）
                    if self.config.get('use_amp', False) and self.scaler is not None:
                        with torch.cuda.amp.autocast():
                            if model_name == 'fusion':
                                outputs = model(batch_stft, batch_ts)
                            elif model_name == 'cnn':
                                outputs = model(batch_stft)
                            elif model_name == 'transformer':
                                outputs = model(batch_ts)
                            
                            loss = criterion(outputs, batch_labels)
                    else:
                        if model_name == 'fusion':
                            outputs = model(batch_stft, batch_ts)
                        elif model_name == 'cnn':
                            outputs = model(batch_stft)
                        elif model_name == 'transformer':
                            outputs = model(batch_ts)
                        
                        loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += batch_labels.size(0)
                    val_correct += predicted.eq(batch_labels).sum().item()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_true_labels.extend(batch_labels.cpu().numpy())
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
            # 计算平均值
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # 更新历史
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            
            # 学习率调度
            scheduler.step()
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 
                          os.path.join(self.output_dir, f'best_{model_name}_model.pth'))
            
            # 早停检查
            if early_stopping(val_acc, model):
                logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
            
            # GPU内存监控
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f'GPU内存: {allocated:.2f}GB / {reserved:.2f}GB (已用/保留)')
                
                # 如果内存使用率过高，清理缓存
                if allocated > 0.8 * torch.cuda.get_device_properties(0).total_memory / (1024**3):
                    torch.cuda.empty_cache()
                    logger.info("🧹 GPU内存缓存已清理")
            
            # 记录日志
            logger.info(
                f'Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, '
                f'Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%'
            )
        
        # 计算最终评估指标
        f1_macro = f1_score(all_true_labels, all_predictions, average='macro')
        f1_weighted = f1_score(all_true_labels, all_predictions, average='weighted')
        
        results = {
            'model_name': model_name,
            'best_val_acc': best_val_acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'history': history,
            'predictions': all_predictions,
            'true_labels': all_true_labels
        }
        
        logger.info(f"{model_name} 训练完成:")
        logger.info(f"  最佳验证准确率: {best_val_acc:.2f}%")
        logger.info(f"  F1-macro: {f1_macro:.4f}")
        logger.info(f"  F1-weighted: {f1_weighted:.4f}")
        
        return results
    
    def run_complete_pipeline(self) -> Dict:
        """运行完整的训练流水线"""
        logger.info("开始运行完整训练流水线...")
        
        # 1. 数据准备
        stft_features, time_series_data, labels = self.prepare_data()
        
        # 2. 创建数据加载器
        train_loader, test_loader = self.create_data_loaders(
            stft_features, time_series_data, labels
        )
        
        # 3. 训练多个模型
        all_results = {}
        
        for model_type in self.config['model_types']:
            logger.info(f"\n{'='*50}")
            logger.info(f"训练 {model_type.upper()} 模型")
            logger.info(f"{'='*50}")
            
            # 创建模型
            if model_type == 'cnn':
                model = AdvancedBehaviorCNN(
                    input_shape=stft_features.shape[1:],
                    num_classes=4,
                    dropout=0.1
                )
            elif model_type == 'transformer':
                model = BehaviorTransformer(
                    input_dim=time_series_data.shape[-1],
                    d_model=512,
                    num_heads=8,
                    num_layers=6,
                    num_classes=4,
                    dropout=0.1
                )
            elif model_type == 'fusion':
                model = MultiModalFusionModel(
                    stft_input_shape=stft_features.shape[1:],
                    time_series_dim=time_series_data.shape[-1],
                    num_classes=4,
                    dropout=0.1
                )
            
            # 打印模型摘要
            model_summary(model, {})
            
            # 训练模型
            results = self.train_model(model, train_loader, test_loader, model_type)
            all_results[model_type] = results
        
        # 4. 生成综合报告
        self.generate_comprehensive_report(all_results)
        
        logger.info("完整训练流水线执行完成!")
        return all_results
    
    def generate_comprehensive_report(self, all_results: Dict):
        """生成综合评估报告"""
        logger.info("生成综合评估报告...")
        
        # 1. 模型性能对比
        performance_df = pd.DataFrame([
            {
                'Model': results['model_name'],
                'Best_Val_Acc': results['best_val_acc'],
                'F1_Macro': results['f1_macro'],
                'F1_Weighted': results['f1_weighted']
            }
            for results in all_results.values()
        ])
        
        performance_df.to_csv(os.path.join(self.output_dir, 'model_comparison.csv'), index=False)
        
        # 2. 可视化结果
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('模型训练和评估结果', fontsize=16)
        
        # 训练损失曲线
        for model_name, results in all_results.items():
            axes[0, 0].plot(results['history']['train_loss'], label=f'{model_name} Train')
            axes[0, 0].plot(results['history']['val_loss'], label=f'{model_name} Val')
        axes[0, 0].set_title('训练损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 训练准确率曲线
        for model_name, results in all_results.items():
            axes[0, 1].plot(results['history']['train_acc'], label=f'{model_name} Train')
            axes[0, 1].plot(results['history']['val_acc'], label=f'{model_name} Val')
        axes[0, 1].set_title('训练准确率曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 模型性能对比
        models = list(all_results.keys())
        val_accs = [all_results[m]['best_val_acc'] for m in models]
        f1_macros = [all_results[m]['f1_macro'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 2].bar(x - width/2, val_accs, width, label='Validation Accuracy', alpha=0.8)
        axes[0, 2].bar(x + width/2, [f*100 for f in f1_macros], width, label='F1-Macro (×100)', alpha=0.8)
        axes[0, 2].set_title('模型性能对比')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(models)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 混淆矩阵（选择最佳模型）
        best_model = max(all_results.keys(), key=lambda k: all_results[k]['best_val_acc'])
        best_results = all_results[best_model]
        
        cm = confusion_matrix(best_results['true_labels'], best_results['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1, 0],
                   xticklabels=['静息', '咀嚼', '咳嗽', '吞咽'],
                   yticklabels=['静息', '咀嚼', '咳嗽', '吞咽'],
                   cmap='Blues')
        axes[1, 0].set_title(f'混淆矩阵 - {best_model.upper()} (最佳模型)')
        
        # 类别F1分数
        class_names = ['静息', '咀嚼', '咳嗽', '吞咽']
        class_f1s = f1_score(best_results['true_labels'], best_results['predictions'], average=None)
        
        axes[1, 1].bar(class_names, class_f1s, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
        axes[1, 1].set_title(f'各类别F1分数 - {best_model.upper()}')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 标签分布
        unique_labels, label_counts = np.unique(best_results['true_labels'], return_counts=True)
        axes[1, 2].pie(label_counts, labels=[class_names[i] for i in unique_labels], 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('数据集标签分布')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comprehensive_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 生成详细分类报告
        report = classification_report(
            best_results['true_labels'], 
            best_results['predictions'],
            target_names=class_names,
            digits=4
        )
        
        with open(os.path.join(self.output_dir, 'classification_report.txt'), 'w', encoding='utf-8') as f:
            f.write(f"最佳模型: {best_model.upper()}\n")
            f.write("="*50 + "\n")
            f.write(report)
            f.write("\n\n模型性能汇总:\n")
            f.write("-"*30 + "\n")
            for model_name, results in all_results.items():
                f.write(f"{model_name.upper()}:\n")
                f.write(f"  验证准确率: {results['best_val_acc']:.2f}%\n")
                f.write(f"  F1-Macro: {results['f1_macro']:.4f}\n")
                f.write(f"  F1-Weighted: {results['f1_weighted']:.4f}\n\n")
        
        # 4. 保存配置和结果
        final_results = {
            'config': self.config,
            'performance_summary': performance_df.to_dict('records'),
            'best_model': best_model,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(self.output_dir, 'experiment_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"综合评估报告已保存到: {self.output_dir}")
        logger.info(f"最佳模型: {best_model.upper()} (验证准确率: {all_results[best_model]['best_val_acc']:.2f}%)")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于耳道PPG&IMU的行为分类完整训练流水线')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default='./hyx_data', help='数据目录')
    parser.add_argument('--models', nargs='+', default=['cnn', 'transformer', 'fusion'], 
                       choices=['cnn', 'transformer', 'fusion'], help='要训练的模型类型')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    
    args = parser.parse_args()
    
    # 创建配置
    config = {
        'data_dir': args.data_dir,
        'model_types': args.models,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'patience': 15,
        'test_size': 0.2,
        'use_class_weights': True,
        'augmentation': True,
        'feature_types': ['stft', 'time_series']
    }
    
    if args.config:
        with open(args.config, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    print("="*60)
    print("基于耳道PPG&IMU的行为分类系统")
    print("目标：静息/咀嚼/咳嗽/吞咽 四分类")
    print("="*60)
    
    # 初始化训练流水线
    pipeline = TrainingPipeline(args.config)
    
    # 运行完整流水线
    results = pipeline.run_complete_pipeline()
    
    print("\n" + "="*60)
    print("训练流水线执行完成!")
    print(f"结果保存在: {pipeline.output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
