# src/sep/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LightweightGuitarSeparationModel(nn.Module):
    """轻量级吉他分离模型（修复形状匹配问题）"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        print(f"🎛️  模型配置: 输入通道数={config.N_CHANNELS}")
        
        # 计算经过池化后的形状变化
        self.input_length = int(config.DURATION * config.SAMPLE_RATE)  # 3 * 22050 = 66150
        
        # 时域网络 - 使用适当的padding确保输入输出形状匹配
        self.encoder = nn.Sequential(
            # 第一层: 输入 (channels, 66150) -> 输出 (32, 33075)
            nn.Conv1d(config.N_CHANNELS, 32, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 66150 -> 33075
            
            # 第二层: 输入 (32, 33075) -> 输出 (64, 16537)
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 33075 -> 16537（向下取整）
            
            # 中间层: 保持形状 (64, 16537) -> (128, 16537)
            nn.Conv1d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # 解码器 - 使用转置卷积恢复形状
        self.decoder = nn.Sequential(
            # 第一层转置卷积: 输入 (128, 16537) -> 输出 (64, 33074)
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # 第二层转置卷积: 输入 (64, 33074) -> 输出 (32, 66148)
            nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # 输出层: 输入 (32, 66148) -> 输出 (channels, 66148)
            nn.Conv1d(32, config.N_CHANNELS, kernel_size=1),
            nn.Tanh()  # 使用Tanh将输出限制在[-1, 1]
        )
        
        # 最后的调整层，确保输出与输入形状完全一致
        self.final_adjust = nn.Sequential(
            nn.Conv1d(config.N_CHANNELS, config.N_CHANNELS, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        self._initialize_weights()
        
        # 测试形状匹配
        self.test_shape_matching()
    
    def test_shape_matching(self):
        """测试输入输出形状是否匹配"""
        print("🧪 测试形状匹配...")
        with torch.no_grad():
            test_input = torch.randn(1, self.config.N_CHANNELS, self.input_length)
            test_output = self.forward(test_input)
            print(f"  输入形状: {test_input.shape}")
            print(f"  输出形状: {test_output.shape}")
            
            if test_input.shape == test_output.shape:
                print("✅ 形状匹配成功!")
            else:
                print(f"⚠️  形状不匹配: 输入{test_input.shape} vs 输出{test_output.shape}")
                print("   调整模型中...")
                self._adjust_for_shape_mismatch(test_input.shape[2], test_output.shape[2])
    
    def _adjust_for_shape_mismatch(self, input_len, output_len):
        """调整模型以解决形状不匹配问题"""
        diff = output_len - input_len
        
        if diff > 0:
            # 如果输出比输入长，添加裁剪
            self.crop_output = lambda x: x[:, :, :input_len]
            print(f"  将裁剪输出: 从{output_len}到{input_len}")
        elif diff < 0:
            # 如果输出比输入短，添加填充
            self.crop_output = lambda x: F.pad(x, (0, -diff))
            print(f"  将填充输出: 从{output_len}到{input_len}")
        else:
            self.crop_output = lambda x: x
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        # 记录原始长度
        original_length = x.shape[2]
        
        # 编码
        encoded = self.encoder(x)
        
        # 解码
        decoded = self.decoder(encoded)
        
        # 最终调整
        output = self.final_adjust(decoded)
        
        # 确保输出与输入形状一致
        if output.shape[2] != original_length:
            # 裁剪或填充到原始长度
            if output.shape[2] > original_length:
                output = output[:, :, :original_length]
            else:
                # 填充
                pad_size = original_length - output.shape[2]
                output = F.pad(output, (0, pad_size))
        
        return output


# 更简单的模型，确保形状匹配
class SimpleGuitarSeparationModel(nn.Module):
    """简单的吉他分离模型，确保输入输出形状一致"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 使用不改变长度的卷积（padding='same'效果）
        def conv1d_same(in_channels, out_channels, kernel_size=3):
            padding = (kernel_size - 1) // 2
            return nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        
        self.model = nn.Sequential(
            # 第一层
            conv1d_same(config.N_CHANNELS, 16, 7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            # 第二层
            conv1d_same(16, 32, 5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # 第三层
            conv1d_same(32, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # 第四层
            conv1d_same(64, 32, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # 第五层
            conv1d_same(32, 16, 3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            # 输出层
            conv1d_same(16, config.N_CHANNELS, 1),
            nn.Tanh()
        )
        
        print(f"🎛️  创建简单分离模型，输入通道={config.N_CHANNELS}")
        
        # 测试形状
        self._test_shape()
    
    def _test_shape(self):
        """测试形状"""
        with torch.no_grad():
            test_input = torch.randn(1, self.config.N_CHANNELS, 66150)
            test_output = self.model(test_input)
            print(f"🧪 形状测试: 输入{test_input.shape} -> 输出{test_output.shape}")
            
            if test_input.shape == test_output.shape:
                print("✅ 形状匹配!")
            else:
                print("❌ 形状不匹配!")
    
    def forward(self, x):
        """前向传播"""
        return self.model(x)


# 使用简单模型作为默认
GuitarSeparationModel = SimpleGuitarSeparationModel