# src/rec/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention

from src.rec.config import GuitarRecConfig

class GuitarAttentionBlock(nn.Module):
    """注意力块，用于捕捉吉他特定的频率模式"""
    
    def __init__(self, channels, reduction=16):
        super(GuitarAttentionBlock, self).__init__()
        
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
        # 频率注意力（针对吉他特定频段）
        self.freq_attention = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(7, 1), padding=(3, 0), bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 通道注意力
        b, c, h, w = x.size()
        avg_out = self.channel_attention(self.avg_pool(x).view(b, c))
        max_out = self.channel_attention(self.max_pool(x).view(b, c))
        channel_att = (avg_out + max_out).view(b, c, 1, 1)
        
        # 频率注意力（关注吉他频率范围）
        freq_avg = torch.mean(x, dim=1, keepdim=True)  # (b, 1, h, w)
        freq_att = self.freq_attention(freq_avg)
        
        return x * channel_att * freq_att

class AdvancedGuitarClassifier(nn.Module):
    """改进的吉他二分类模型，专门处理节奏吉他和扫弦"""
    
    def __init__(self, input_shape=GuitarRecConfig.INPUT_SHAPE, dropout_rate=0.3):
        super(AdvancedGuitarClassifier, self).__init__()
        
        self.input_shape = input_shape
        
        # CNN特征提取器 - 更深的结构来捕捉复杂特征
        self.features = nn.Sequential(
            # 块1：低频特征提取（吉他基频）
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # 只在频率维度池化
            nn.Dropout2d(dropout_rate/2),
            
            # 注意力块1
            GuitarAttentionBlock(32),
            
            # 块2：中频特征提取（和声特征）
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # 注意力块2
            GuitarAttentionBlock(64),
            
            # 块3：高频特征提取（扫弦泛音）
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, (3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # 块4：时间模式提取（节奏特征）
            nn.Conv2d(128, 256, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 计算全连接层输入大小
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape)
            dummy_output = self.features(dummy_input)
            fc_input_size = dummy_output.view(1, -1).shape[1]
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fc_input_size, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate/2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # 二分类输出
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class GuitarClassifier(nn.Module):
    """吉他二分类模型（原版，保持兼容性）"""
    
    def __init__(self, input_shape=GuitarRecConfig.INPUT_SHAPE, dropout_rate=0.3):
        super(GuitarClassifier, self).__init__()
        
        self.input_shape = input_shape
        
        # CNN特征提取器
        self.features = nn.Sequential(
            # 块1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # 块2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # 块3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 计算全连接层输入大小
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape)
            dummy_output = self.features(dummy_input)
            fc_input_size = dummy_output.view(1, -1).shape[1]
        
        # 分类器 - 移除BatchNorm1d避免batch size=1的问题
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fc_input_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # 二分类输出
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SimplifiedGuitarClassifier(nn.Module):
    """简化的吉他二分类模型（更轻量）"""
    
    def __init__(self, input_shape=GuitarRecConfig.INPUT_SHAPE, dropout_rate=0.3):
        super(SimplifiedGuitarClassifier, self).__init__()
        
        self.input_shape = input_shape
        
        # 更简单的CNN结构
        self.features = nn.Sequential(
            # 块1
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # 块2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
            
            # 块3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 计算全连接层输入大小
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape)
            dummy_output = self.features(dummy_input)
            fc_input_size = dummy_output.view(1, -1).shape[1]
        
        # 分类器 - 移除BatchNorm1d避免batch size=1的问题
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fc_input_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # 二分类输出
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x