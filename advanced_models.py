# -*- coding: utf-8 -*-
"""
高级深度学习模型：CNN ResNet + Transformer时间序列模型
用于基于耳道PPG&IMU信号的行为分类 (静息/咀嚼/咳嗽/吞咽)

模型架构：
1. CNN ResNet: 处理STFT特征的2D卷积网络
2. Transformer: 处理时间序列特征的注意力机制模型
3. 混合模型: 结合CNN和Transformer的多模态融合
4. 自适应特征融合: 动态权重分配机制

作者：基于Project-Swallow项目扩展
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        output = self.w_o(context)
        return output

class TransformerBlock(nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力 + 残差连接
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class BehaviorTransformer(nn.Module):
    """基于Transformer的行为分类模型"""
    
    def __init__(self, 
                 input_dim: int = 6,  # PPG(3) + IMU(3)
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 2048,
                 max_seq_len: int = 1000,
                 num_classes: int = 4,
                 dropout: float = 0.1):
        super(BehaviorTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer编码器层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 全局特征聚合
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            mask: [batch_size, seq_len] 可选的掩码
        Returns:
            logits: [batch_size, num_classes]
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # 输入投影
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        x = x * math.sqrt(self.d_model)  # 缩放
        
        # 位置编码 (需要转置为 [seq_len, batch_size, d_model])
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Transformer编码器
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # 全局池化：时间维度平均
        x = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        x = self.global_pool(x).squeeze(-1)  # [batch_size, d_model]
        
        # 分类
        logits = self.classifier(x)
        
        return logits

class ResNetBlock(nn.Module):
    """ResNet基本块（改进版）"""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 dropout: float = 0.1):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout)
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class AdvancedBehaviorCNN(nn.Module):
    """改进的行为分类CNN（基于ResNet）"""
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int],  # (channels, height, width)
                 num_classes: int = 4,
                 dropout: float = 0.1):
        super(AdvancedBehaviorCNN, self).__init__()
        
        self.input_shape = input_shape
        channels, height, width = input_shape
        
        # 输入适配层
        self.input_conv = nn.Conv2d(channels, 64, 7, 2, 3, bias=False)
        self.input_bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        # ResNet层
        self.layer1 = self._make_layer(64, 64, 2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, dropout=dropout)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.Sigmoid()
        )
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, 
                   stride: int, dropout: float) -> nn.Sequential:
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride, dropout))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入处理
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.maxpool(x)
        
        # ResNet特征提取
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 注意力机制
        attention_weights = self.attention(x)
        
        # 全局池化
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 应用注意力权重
        x = x * attention_weights
        
        # 分类
        x = self.classifier(x)
        
        return x

class MultiModalFusionModel(nn.Module):
    """多模态融合模型：结合CNN和Transformer"""
    
    def __init__(self,
                 stft_input_shape: Tuple[int, int, int],  # STFT特征形状
                 time_series_dim: int = 6,  # 时间序列维度
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_transformer_layers: int = 4,
                 num_classes: int = 4,
                 dropout: float = 0.1):
        super(MultiModalFusionModel, self).__init__()
        
        # CNN分支：处理STFT特征
        self.cnn_branch = AdvancedBehaviorCNN(
            input_shape=stft_input_shape,
            num_classes=256,  # 输出特征向量而不是分类
            dropout=dropout
        )
        # 修改CNN的分类头为特征提取器
        self.cnn_branch.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 256)  # 输出256维特征
        )
        
        # Transformer分支：处理时间序列
        self.transformer_branch = BehaviorTransformer(
            input_dim=time_series_dim,
            d_model=d_model//2,  # 减少维度以平衡计算
            num_heads=num_heads//2,
            num_layers=num_transformer_layers,
            num_classes=256,  # 输出特征向量
            dropout=dropout
        )
        # 修改Transformer的分类头
        self.transformer_branch.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model//2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 256)  # 输出256维特征
        )
        
        # 特征融合层
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=256, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        
        self.fusion_norm = nn.LayerNorm(256)
        
        # 自适应权重学习
        self.adaptive_weights = nn.Sequential(
            nn.Linear(512, 128),  # CNN + Transformer特征
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),  # 两个分支的权重
            nn.Softmax(dim=1)
        )
        
        # 最终分类器
        self.final_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, stft_features: torch.Tensor, 
                time_series: torch.Tensor) -> torch.Tensor:
        """
        Args:
            stft_features: [batch_size, channels, height, width] STFT特征
            time_series: [batch_size, seq_len, features] 时间序列数据
        Returns:
            logits: [batch_size, num_classes]
        """
        # CNN分支
        cnn_features = self.cnn_branch(stft_features)  # [batch_size, 256]
        
        # Transformer分支
        transformer_features = self.transformer_branch(time_series)  # [batch_size, 256]
        
        # 准备注意力输入 [batch_size, 2, 256]
        features_stack = torch.stack([cnn_features, transformer_features], dim=1)
        
        # 交叉注意力融合
        fused_features, _ = self.fusion_attention(
            features_stack, features_stack, features_stack
        )  # [batch_size, 2, 256]
        
        # 残差连接
        fused_features = self.fusion_norm(features_stack + fused_features)
        
        # 自适应权重融合
        concat_features = torch.cat([cnn_features, transformer_features], dim=1)
        adaptive_weights = self.adaptive_weights(concat_features)  # [batch_size, 2]
        
        # 加权融合
        weighted_features = (fused_features * adaptive_weights.unsqueeze(-1)).sum(dim=1)
        
        # 最终分类
        logits = self.final_classifier(weighted_features)
        
        return logits

class BehaviorClassificationLoss(nn.Module):
    """自定义损失函数：结合交叉熵和焦点损失"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 class_weights: Optional[torch.Tensor] = None):
        super(BehaviorClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size]
        """
        # 计算交叉熵损失
        ce_loss = self.ce_loss(logits, targets)
        
        # 计算概率
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        
        # 焦点损失
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()

def create_model(model_type: str, **kwargs):
    """模型工厂函数"""
    if model_type == 'cnn':
        return AdvancedBehaviorCNN(**kwargs)
    elif model_type == 'transformer':
        return BehaviorTransformer(**kwargs)
    elif model_type == 'fusion':
        return MultiModalFusionModel(**kwargs)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")

def count_parameters(model: nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model: nn.Module, input_shapes: dict):
    """打印模型摘要"""
    print(f"模型类型: {model.__class__.__name__}")
    print(f"可训练参数数量: {count_parameters(model):,}")
    
    # 计算模型大小（MB）
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    print(f"模型大小: {model_size_mb:.2f} MB")
    
    print("\n模型结构:")
    print(model)

if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 测试CNN模型
    print("\n=== 测试CNN模型 ===")
    cnn_model = AdvancedBehaviorCNN(
        input_shape=(6, 129, 39),  # STFT特征形状示例
        num_classes=4
    ).to(device)
    
    stft_input = torch.randn(8, 6, 129, 39).to(device)  # batch_size=8
    cnn_output = cnn_model(stft_input)
    print(f"CNN输出形状: {cnn_output.shape}")
    model_summary(cnn_model, {'stft': (6, 129, 39)})
    
    # 测试Transformer模型
    print("\n=== 测试Transformer模型 ===")
    transformer_model = BehaviorTransformer(
        input_dim=6,
        d_model=512,
        num_classes=4
    ).to(device)
    
    time_series_input = torch.randn(8, 500, 6).to(device)  # batch_size=8, seq_len=500
    transformer_output = transformer_model(time_series_input)
    print(f"Transformer输出形状: {transformer_output.shape}")
    model_summary(transformer_model, {'time_series': (500, 6)})
    
    # 测试融合模型
    print("\n=== 测试多模态融合模型 ===")
    fusion_model = MultiModalFusionModel(
        stft_input_shape=(6, 129, 39),
        time_series_dim=6,
        num_classes=4
    ).to(device)
    
    fusion_output = fusion_model(stft_input, time_series_input)
    print(f"融合模型输出形状: {fusion_output.shape}")
    model_summary(fusion_model, {
        'stft': (6, 129, 39),
        'time_series': (500, 6)
    })
    
    print("\n=== 模型测试完成 ===")
