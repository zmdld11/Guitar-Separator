import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
from .config import Config
from ..common.audio_utils import stft, istft

# ---------- 基础模块 ----------
class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        a, b = x.chunk(2, dim=self.dim)
        return a * torch.sigmoid(b)

class LayerScale(nn.Module):
    def __init__(self, dim, init=1e-3):
        super().__init__()
        self.gamma = nn.Parameter(init * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, *([1] * (x.dim() - 2)))
        return gamma * x

class FreqPositionalEmbedding(nn.Module):
    """频率位置嵌入,用于频域分支"""
    def __init__(self, dim, max_freq=2048):
        super().__init__()
        self.embed = nn.Embedding(max_freq, dim)

    def forward(self, x, freq_idx):
        # x: (batch, ch, time, freq)
        x = x.permute(0, 2, 3, 1)          # (batch, time, freq, ch)
        pos = self.embed(freq_idx)         # (freq, ch)
        pos = pos.unsqueeze(0).unsqueeze(0) # (1, 1, freq, ch)
        x = x + pos
        x = x.permute(0, 3, 1, 2)           # 转回 (batch, ch, time, freq)
        return x

class ResidualBlock(nn.Module):
    """带扩张卷积的残差块(Demucs风格)"""
    def __init__(self, dim, dilation=1, use_lstm=False, use_attn=False, attn_heads=4):
        super().__init__()
        self.dim = dim
        self.use_lstm = use_lstm
        self.use_attn = use_attn

        self.conv1 = nn.Conv1d(dim, dim//4, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.LayerNorm(dim//4)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(dim//4, dim//4, kernel_size=3, padding=2*dilation, dilation=2*dilation)
        self.norm2 = nn.LayerNorm(dim//4)
        self.act2 = nn.GELU()

        if use_lstm:
            self.lstm = nn.LSTM(dim//4, dim//4, batch_first=True, bidirectional=True)
            self.lstm_proj = nn.Linear(dim//4*2, dim//4)

        if use_attn:
            self.attn = nn.MultiheadAttention(dim//4, attn_heads, batch_first=True)
            self.local_bias = nn.Parameter(torch.zeros(1, attn_heads, 1, 200))

        self.proj = nn.Conv1d(dim//4, dim*2, kernel_size=1)
        self.glu = GLU(1)
        self.ls = LayerScale(dim)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = self.act1(x)
        x = x.transpose(1, 2)

        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = self.act2(x)

        if self.use_lstm:
            lstm_out, _ = self.lstm(x)
            lstm_out = self.lstm_proj(lstm_out)
            x = x + lstm_out

        if self.use_attn:
            attn_out, _ = self.attn(x, x, x)
            x = x + attn_out

        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.glu(x)
        x = self.ls(x)
        return x + identity

class EncoderLayer(nn.Module):
    """时域编码层(1D卷积 + 残差块)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1, use_lstm=False, use_attn=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding=(kernel_size - stride)//2)
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()
        self.residual = ResidualBlock(out_channels, dilation, use_lstm, use_attn)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.residual(x)
        return x

class FreqEncoderLayer(nn.Module):
    """频域编码层(2D卷积,仅频率方向下采样)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1, use_lstm=False, use_attn=False):
        super().__init__()
        time_ks, freq_ks = kernel_size
        time_s, freq_s = stride
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(time_ks, freq_ks),
                              stride=(1, freq_s),
                              padding=(time_ks//2, (freq_ks - freq_s)//2))
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()
        self.residual = ResidualBlock(out_channels, dilation, use_lstm, use_attn)

    def forward(self, x):
        x = self.conv(x)                     # (b, out_ch, t, f)
        x = self.norm(x)
        x = self.act(x)
        b, c, t, f = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # (b, f, c, t)
        x = x.view(-1, c, t)                  # (b*f, c, t)
        x = self.residual(x)
        x = x.view(b, f, c, t).permute(0, 2, 3, 1)  # (b, c, t, f)
        return x

class DecoderLayer(nn.Module):
    """时域解码层(上采样 + 跳跃连接拼接)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, skip_channels=0):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels,
                                         kernel_size, stride,
                                         padding=(kernel_size - stride)//2,
                                         output_padding=stride-1)
        # 用 1x1 卷积将在 cat 后增加的通道压缩回 out_channels
        self.mix_conv = nn.Conv1d(out_channels + skip_channels, out_channels, 1)
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()
        self.conv = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x, skip=None):
        x = self.deconv(x)
        if skip is not None:
            # 对齐时间维度
            min_len = min(x.size(2), skip.size(2))
            x = x[:, :, :min_len]
            skip = skip[:, :, :min_len]
            x = torch.cat([x, skip], dim=1)   # 沿通道拼接而不是直接相加
        x = self.mix_conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x

class FreqDecoderLayer(nn.Module):
    """频域解码层(频率维上采样 + 跳跃连接相加)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, skip_channels=0):
        super().__init__()
        time_ks, freq_ks = kernel_size
        time_s, freq_s = stride
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=(time_ks, freq_ks),
                                         stride=(1, freq_s),
                                         padding=(time_ks//2, (freq_ks - freq_s)//2),
                                         output_padding=(0, freq_s-1))
        # 用于将在 cat 后增加的通道压回 out_channels
        self.mix_conv = nn.Conv2d(out_channels + skip_channels, out_channels, 1)
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()
        self.conv = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x, skip=None):
        x = self.deconv(x)
        if skip is not None:
            # 对齐时空维度
            min_t = min(x.size(2), skip.size(2))
            min_f = min(x.size(3), skip.size(3))
            x = x[:, :, :min_t, :min_f]
            skip = skip[:, :, :min_t, :min_f]
            x = torch.cat([x, skip], dim=1)
        x = self.mix_conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x

class CrossTransformer(nn.Module):
    """瓶颈Transformer,时域和频域共享特征处理"""
    def __init__(self, dim, heads, layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(dim, heads, dropout, batch_first=True),
                'ffn': nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim)),
                'norm1': nn.LayerNorm(dim),
                'norm2': nn.LayerNorm(dim),
            }))

    def forward(self, x):
        # x: (batch, seq, dim)
        for layer in self.layers:
            x2 = layer['norm1'](x)
            x2, _ = layer['self_attn'](x2, x2, x2)
            x = x + x2

            x2 = layer['norm2'](x)
            x2 = layer['ffn'](x2)
            x = x + x2
        return x

def rescale_module(module, reference=1.0):
    """权重重缩放,使每层输出的方差接近1"""
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
            if sub.weight.numel() > 1:
                std = sub.weight.std().detach()
            else:
                std = torch.tensor(1.0, device=sub.weight.device)
            if std > 0:
                scale = (reference / std).clamp_(0.1, 10.0)
                sub.weight.data.mul_(scale)
                if sub.bias is not None:
                    sub.bias.data.mul_(scale)

class HTDemucs(nn.Module):
    """HT Demucs v4 风格模型(无扩散,注入式融合,直接相加skip)"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # ---------- 时域编码器 ----------
        time_encoders = []
        in_ch = 1
        out_ch = config.time_channels
        time_channels_list = []  # 记录每层通道数
        for i in range(config.time_depth):
            use_lstm = (i >= config.time_depth - 2)
            use_attn = (i >= config.time_depth - 2)
            layer = EncoderLayer(in_ch, out_ch,
                                 kernel_size=config.time_kernel_size,
                                 stride=config.time_stride,
                                 dilation=1,
                                 use_lstm=use_lstm,
                                 use_attn=use_attn)
            time_encoders.append(layer)
            time_channels_list.append(out_ch)
            in_ch = out_ch
            out_ch = min(out_ch * 2, config.transformer_dim)
        self.time_encoders = nn.ModuleList(time_encoders)
        self.time_encoder_out_ch = in_ch

        # ---------- 频域编码器 ----------
        freq_encoders = []
        in_ch_f = 2   # 实部+虚部
        out_ch_f = config.freq_channels
        freq_channels_list = []  # 记录每层通道数
        for i in range(config.freq_depth):
            use_lstm = (i >= config.freq_depth - 2)
            use_attn = (i >= config.freq_depth - 2)
            layer = FreqEncoderLayer(in_ch_f, out_ch_f,
                                     kernel_size=config.freq_kernel_size,
                                     stride=config.freq_stride,
                                     dilation=1,
                                     use_lstm=use_lstm,
                                     use_attn=use_attn)
            freq_encoders.append(layer)
            freq_channels_list.append(out_ch_f)
            in_ch_f = out_ch_f
            out_ch_f = min(out_ch_f * 2, config.transformer_dim)
        self.freq_encoders = nn.ModuleList(freq_encoders)
        self.freq_encoder_out_ch = in_ch_f

        self.freq_pos = FreqPositionalEmbedding(config.freq_channels)

        # 注入投影:将时域通道投影到频域通道
        self.inject_projs = nn.ModuleList()
        for t_ch, f_ch in zip(time_channels_list, freq_channels_list):
            self.inject_projs.append(nn.Conv1d(t_ch, f_ch, 1) if t_ch != f_ch else nn.Identity())

        # ---------- 共享编码器 ----------
        self.shared_encoder = nn.ModuleList()
        in_ch_s = self.time_encoder_out_ch + self.freq_encoder_out_ch  # 融合后
        out_ch_s = config.shared_channels
        shared_channels_list = []  # 记录每层通道数
        for _ in range(config.shared_depth):
            self.shared_encoder.append(
                EncoderLayer(in_ch_s, out_ch_s,
                             kernel_size=config.shared_kernel_size,
                             stride=2,
                             use_lstm=True,
                             use_attn=True)
            )
            shared_channels_list.append(out_ch_s)
            in_ch_s = out_ch_s
            out_ch_s = min(out_ch_s * 2, config.transformer_dim)

        # ---------- Transformer ----------
        self.transformer = CrossTransformer(
            dim=in_ch_s,
            heads=config.transformer_heads,
            layers=config.transformer_layers,
            dropout=config.transformer_dropout
        )

        # ---------- 共享解码器 ----------
        self.shared_decoder = nn.ModuleList()
        dec_in_ch = in_ch_s
        for i, enc in enumerate(reversed(self.shared_encoder)):
            out_ch = enc.conv.in_channels
            skip_ch = shared_channels_list[-(i+1)]  # 对应编码器层的输出通道数
            self.shared_decoder.append(
                DecoderLayer(dec_in_ch, out_ch,
                             kernel_size=config.shared_kernel_size,
                             stride=2,
                             skip_channels=skip_ch)
            )
            dec_in_ch = out_ch

        # ---------- 时域解码器 ----------
        self.time_decoders = nn.ModuleList()
        dec_in_ch_t = dec_in_ch
        for i, enc in enumerate(reversed(self.time_encoders)):
            out_ch = enc.conv.in_channels
            skip_ch = time_channels_list[-(i+1)]  # 对应编码器层的输出通道数
            self.time_decoders.append(
                DecoderLayer(dec_in_ch_t, out_ch,
                             kernel_size=config.time_kernel_size,
                             stride=config.time_stride,
                             skip_channels=skip_ch)
            )
            dec_in_ch_t = out_ch
        self.time_out = nn.Conv1d(dec_in_ch_t, 1, 1)

        # ---------- 频域解码器 ----------
        self.freq_decoders = nn.ModuleList()
        dec_in_ch_f = dec_in_ch
        for i, enc in enumerate(reversed(self.freq_encoders)):
            out_ch = enc.conv.in_channels
            skip_ch = freq_channels_list[-(i+1)]  # 对应编码器层的输出通道数
            self.freq_decoders.append(
                FreqDecoderLayer(dec_in_ch_f, out_ch,
                                 kernel_size=config.freq_kernel_size,
                                 stride=config.freq_stride,
                                 skip_channels=skip_ch)
            )
            dec_in_ch_f = out_ch
        self.freq_out = nn.Conv2d(dec_in_ch_f, 2, 1)   # 输出 mask (实/虚)

        # 权重重缩放 (删除原有的导致爆炸的rescale逻辑)
        # self.apply(lambda m: rescale_module(m, reference=1.0) if hasattr(m, 'weight') else None)

    def forward(self, waveform):
        """
        waveform: (batch, 1, T)
        返回分离的吉他波形 (batch, 1, T)
        """
        input_length = waveform.size(2)  # 记录输入长度
        
        # ---------- 时域编码 ----------
        t = waveform
        time_skips = []
        for enc in self.time_encoders:
            t = enc(t)
            time_skips.append(t)

        # ---------- 频域编码 ----------
        spec = stft(waveform,
                    fft_size=self.config.stft_fft_size,
                    hop_length=self.config.stft_hop_length,
                    win_length=self.config.stft_win_length,
                    window=self.config.stft_window,
                    normalized=self.config.stft_normalized)  # (b, 2, t, freq)
        
        original_freq_bins = spec.size(3)  # 记录原始频率维度
        original_time_frames = spec.size(2)  # 记录原始时间帧数
        
        f = spec
        freq_skips = []
        for i, enc in enumerate(self.freq_encoders):
            f = enc(f)  # 先经过编码器卷积
            # 在第一层后添加频率位置嵌入
            if i == 0:
                freq_idx = torch.arange(f.size(3), device=f.device)
                f = self.freq_pos(f, freq_idx)
            # 注入时域特征
            time_feat = time_skips[i]   # (b, t_ch, t_len)
            # 对齐时间维度
            if time_feat.size(2) != f.size(2):
                min_len = min(time_feat.size(2), f.size(2))
                time_feat = time_feat[:, :, :min_len]
                f = f[:, :, :min_len, :]
            proj = self.inject_projs[i](time_feat)   # (b, f_ch, t)
            proj = proj.unsqueeze(-1)                # (b, f_ch, t, 1)
            f = f + proj                              # 注入
            freq_skips.append(f)

        # ---------- 对齐维度并融合 ----------
        # 压缩频率维到1(通过平均池化)
        f_pooled = f.mean(dim=3, keepdim=False)   # (b, c_f, t)
        # 对齐时间
        min_len = min(t.size(2), f_pooled.size(2))
        t = t[:, :, :min_len]
        f_pooled = f_pooled[:, :, :min_len]
        combined = torch.cat([t, f_pooled], dim=1)  # (b, c_t+c_f, t)

        # ---------- 共享编码 ----------
        shared_skips = []
        for enc in self.shared_encoder:
            combined = enc(combined)
            shared_skips.append(combined)

        # ---------- Transformer瓶颈 ----------
        bottleneck = combined.permute(0, 2, 1)  # (b, t, c)
        bottleneck = self.transformer(bottleneck)
        combined = bottleneck.permute(0, 2, 1)  # (b, c, t)

        # ---------- 共享解码 ----------
        for i, dec in enumerate(self.shared_decoder):
            skip = shared_skips[-(i+1)]
            combined = dec(combined, skip)

        # ---------- 时域解码 ----------
        t_dec = combined
        for i, dec in enumerate(self.time_decoders):
            skip = time_skips[-(i+1)]
            t_dec = dec(t_dec, skip)
        time_out = self.time_out(t_dec)  # (b, 1, T)
        
        # 确保时域输出长度匹配
        if time_out.size(2) != input_length:
            time_out = F.interpolate(time_out, size=input_length, mode='linear', align_corners=False)

        # ---------- 频域解码 ----------
        f_dec = combined.unsqueeze(3)  # (b, c, t, 1)
        for i, dec in enumerate(self.freq_decoders):
            skip = freq_skips[-(i+1)]
            f_dec = dec(f_dec, skip)
        mask = self.freq_out(f_dec)   # (b, 2, t, f_dec)

        # 调整mask的频率和时间维度以匹配原始谱
        if mask.size(2) != original_time_frames or mask.size(3) != original_freq_bins:
            mask = F.interpolate(mask, size=(original_time_frames, original_freq_bins), 
                               mode='bilinear', align_corners=False)
        
        # 对齐时间维度
        min_t = min(spec.size(2), mask.size(2))
        spec = spec[:, :, :min_t, :]
        mask = mask[:, :, :min_t, :]

        # 应用sigmoid约束mask范围
        mask = torch.sigmoid(mask)

        # 应用掩码:估计谱 = 输入谱 * mask
        est_spec = spec * mask
        
        # iSTFT 还原波形
        est_wave = istft(est_spec,
                        fft_size=self.config.stft_fft_size,
                        hop_length=self.config.stft_hop_length,
                        win_length=self.config.stft_win_length,
                        window=self.config.stft_window,
                        normalized=self.config.stft_normalized,
                        length=input_length)  # 指定输出长度为输入长度

        # 融合两个分支
        final_wave = time_out + est_wave
        
        # 最终确保输出长度完全匹配
        if final_wave.size(2) != input_length:
            final_wave = F.interpolate(final_wave, size=input_length, mode='linear', align_corners=False)
            
        return final_wave