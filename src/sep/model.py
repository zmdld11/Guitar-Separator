import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Config
from ..common.audio_utils import stft, istft

class SEBlock(nn.Module):
    """ Squeeze-and-Excitation (SE) 通道注意力机制 """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DoubleConv(nn.Module):
    """ (Conv2D -> BatchNorm -> LeakyReLU) * 2 + SEBlock """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.se = SEBlock(out_channels)

    def forward(self, x):
        return self.se(self.double_conv(x))

class TemporalBottleneck(nn.Module):
    """ 在U-Net最深处用于捕捉长时间上下文的时序膨胀卷积 """
    def __init__(self, channels):
        super().__init__()
        # 沿时间轴的膨胀，扩大时间感受野以识别吉他延音
        self.dilated1 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 1), dilation=(1, 1))
        self.dilated2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 2), dilation=(1, 2))
        self.dilated3 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=(1, 4), dilation=(1, 4))
        self.act = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        res = x
        x1 = self.act(self.dilated1(x))
        x2 = self.act(self.dilated2(x1))
        x3 = self.act(self.dilated3(x2))
        return res + x3    

class Down(nn.Module):
    """ 下采样：Maxpool -> DoubleConv """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """ 上采样：UpSample -> Concat -> DoubleConv """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 对齐尺寸
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class HTDemucs(nn.Module):
    """
    [V3.0 破壳版] - Complex-Aware & Phase-Correction U-Net
    - 绝不仅仅只预测幅度(Mask)，还要预测相位的偏移量 (Phase Delta)！
    - 这是真正能解决混音和原声相位不合导致负SDR终极瓶颈的高级解法。
    - 网络输出2个通道:
       1) Mag Mask (Sigmoid) 用来过滤幅度
       2) Phase Delta (Tanh*PI) 用来给原相加上修正量旋转回目标相位
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        c = 16
        
        # 输入通道为2:对数幅度 + 归一化初始相位
        self.inc = DoubleConv(2, c)
        self.down1 = Down(c, c*2)
        self.down2 = Down(c*2, c*4)
        self.down3 = Down(c*4, c*8)
        self.down4 = Down(c*8, c*16)
        
        # 核心增强：时间膨胀模块，加深极低抽象层的时态理解
        self.bottleneck = TemporalBottleneck(c*16)
        
        self.up1 = Up(c*16 + c*8, c*8)
        self.up2 = Up(c*8 + c*4, c*4)
        self.up3 = Up(c*4 + c*2, c*2)
        self.up4 = Up(c*2 + c, c)
        
        # 输出通道为2: [幅度mask标量, 相位补偿角度]
        self.outc = nn.Conv2d(c, 2, kernel_size=1)

    def forward(self, waveform):
        input_length = waveform.size(2)
        
        # 1. STFT
        spec = stft(waveform,
                    fft_size=self.config.stft_fft_size,
                    hop_length=self.config.stft_hop_length,
                    win_length=self.config.stft_win_length,
                    window=self.config.stft_window,
                    normalized=self.config.stft_normalized)
        
        spec_real = spec[:, 0:1, :, :]
        spec_imag = spec[:, 1:2, :, :]
        
        # 2. 提取幅度谱与初始相位
        mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-8)
        phase_angle = torch.atan2(spec_imag, spec_real) # 弧度 [-pi, pi]
        
        # 压缩幅度和归一化相位给网络
        mag_compressed = torch.log1p(mag)
        phase_norm = phase_angle / torch.pi
        
        # 将输入组合成 (batch, 2, freq, time)
        x_in = torch.cat([mag_compressed, phase_norm], dim=1)
        x = x_in.permute(0, 1, 3, 2)
        
        # 3. 穿梭 U-Net
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x5 = self.bottleneck(x5)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        out = self.outc(x)
        out = out.permute(0, 1, 3, 2) # (batch, 2, t, freq)
        
        # 4. 解析 Mask 和 Phase Delta
        # 通道 0: Mag Mask -> Sigmoid压缩到 (0, 1) 然后乘以原始幅度
        mag_mask = torch.sigmoid(out[:, 0:1, :, :])
        est_mag = mag * mag_mask
        
        # 通道 1: Phase Delta -> Tanh乘以PI限制在 (-pi, pi)，加到初始相位上旋转
        phase_delta = torch.tanh(out[:, 1:2, :, :]) * torch.pi
        est_phase = phase_angle + phase_delta
        
        # 5. 生成新的实部虚部
        est_real = est_mag * torch.cos(est_phase)
        est_imag = est_mag * torch.sin(est_phase)
        est_spec = torch.cat([est_real, est_imag], dim=1)
        
        # 6. iSTFT 重构
        final_wave = istft(est_spec,
                        fft_size=self.config.stft_fft_size,
                        hop_length=self.config.stft_hop_length,
                        win_length=self.config.stft_win_length,
                        window=self.config.stft_window,
                        normalized=self.config.stft_normalized,
                        length=input_length)
                        
        return final_wave