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
        # x shape: (batch, dim, time) 或 (batch, dim, time, freq)
        # 将 gamma 重塑为 (1, dim, 1, 1...) 以匹配 x 的维度
        gamma = self.gamma.view(1, -1, *([1] * (x.dim() - 2)))
        return gamma * x

class FreqPositionalEmbedding(nn.Module):
    """频率位置嵌入，用于频域分支"""
    def __init__(self, dim, max_freq=2048):
        super().__init__()
        self.embed = nn.Embedding(max_freq, dim)

    def forward(self, x, freq_idx):
        # x 实际传入形状: (batch, ch, time, freq)
        # 需要转换为 (batch, time, freq, ch) 以添加位置嵌入
        x = x.permute(0, 2, 3, 1)          # (batch, time, freq, ch)
        pos = self.embed(freq_idx)         # (freq, ch)
        pos = pos.unsqueeze(0).unsqueeze(0) # (1, 1, freq, ch)
        x = x + pos                         # 广播到 (batch, time, freq, ch)
        x = x.permute(0, 3, 1, 2)           # 转回 (batch, ch, time, freq)
        return x

# ---------- 残差块（带扩张卷积和局部注意力） ----------
class ResidualBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_lstm=False, use_attn=False, attn_heads=4):
        super().__init__()
        self.dim = dim
        self.use_lstm = use_lstm
        self.use_attn = use_attn

        # 第一个扩张卷积 (dilation=1)
        self.conv1 = nn.Conv1d(dim, dim//4, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.LayerNorm(dim//4)
        self.act1 = nn.GELU()

        # 第二个扩张卷积 (dilation=2)
        self.conv2 = nn.Conv1d(dim//4, dim//4, kernel_size=3, padding=2*dilation, dilation=2*dilation)
        self.norm2 = nn.LayerNorm(dim//4)
        self.act2 = nn.GELU()

        # LSTM（如果启用）
        if use_lstm:
            self.lstm = nn.LSTM(dim//4, dim//4, batch_first=True, bidirectional=True)
            self.lstm_proj = nn.Linear(dim//4*2, dim//4)

        # 局部注意力（如果启用）
        if use_attn:
            self.attn = nn.MultiheadAttention(dim//4, attn_heads, batch_first=True)
            # 可学习的位置偏置
            self.local_bias = nn.Parameter(torch.zeros(1, attn_heads, 1, 200))  # 最大跨度200

        # 输出投影 (1x1 conv + GLU)
        self.proj = nn.Conv1d(dim//4, dim*2, kernel_size=1)
        self.glu = GLU(1)
        self.ls = LayerScale(dim)

    def forward(self, x):
        # x: (batch, dim, time)
        identity = x

        # 第一个卷积
        x = self.conv1(x)
        x = x.transpose(1, 2)  # (b, t, c)
        x = self.norm1(x)
        x = self.act1(x)
        x = x.transpose(1, 2)  # (b, c, t)

        # 第二个卷积
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.norm2(x)
        x = self.act2(x)

        # LSTM（时间维处理）
        if self.use_lstm:
            lstm_out, _ = self.lstm(x)
            lstm_out = self.lstm_proj(lstm_out)
            x = x + lstm_out

        # 局部注意力
        if self.use_attn:
            # 为注意力添加位置偏置（简化：使用固定最大长度）
            attn_out, _ = self.attn(x, x, x)
            x = x + attn_out

        x = x.transpose(1, 2)  # (b, c, t)

        # 输出投影
        x = self.proj(x)        # (b, dim*2, t)
        x = self.glu(x)          # (b, dim, t)
        x = self.ls(x)
        return x + identity

# ---------- 编码器层 ----------
class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1, use_lstm=False, use_attn=False):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                              padding=(kernel_size - stride)//2)  # 保持长度 / stride
        self.residual = ResidualBlock(out_channels, dilation, use_lstm, use_attn)

    def forward(self, x):
        x = self.conv(x)
        x = self.residual(x)
        return x

class FreqEncoderLayer(nn.Module):
    """频域编码器层，沿频率维卷积"""
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1, use_lstm=False, use_attn=False):
        super().__init__()
        # 卷积核: (time_kernel, freq_kernel) 这里 kernel_size 是元组
        time_ks, freq_ks = kernel_size
        time_s, freq_s = stride
        # 仅在频率方向下采样，时间方向保持（stride=(1, freq_s)）
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(time_ks, freq_ks),
                              stride=(1, freq_s),
                              padding=(time_ks//2, (freq_ks - freq_s)//2))
        # 残差块处理在时间维度（reshape后）
        self.residual = ResidualBlock(out_channels, dilation, use_lstm, use_attn)

    def forward(self, x):
        # x: (batch, ch, time, freq)
        x = self.conv(x)                 # (b, out_ch, time, freq')
        b, c, t, f = x.shape
        # 合并 freq 到 batch 或通道？我们reshape为 (b*f, c, t) 以便残差块处理时间维
        x = x.permute(0, 3, 1, 2).contiguous()  # (b, f, c, t)
        x = x.view(-1, c, t)              # (b*f, c, t)
        x = self.residual(x)
        x = x.view(b, f, c, t).permute(0, 2, 3, 1)  # (b, c, t, f)
        return x

# ---------- 解码器层 ----------
class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 skip_channels=0):
        super().__init__()
        self.skip_conv = nn.Conv1d(skip_channels, out_channels, 1) if skip_channels > 0 else None
        self.glu = GLU(1)
        self.conv1 = nn.Conv1d(in_channels + (out_channels if skip_channels>0 else 0),
                               out_channels*2, 1)
        self.deconv = nn.ConvTranspose1d(out_channels, out_channels,
                                         kernel_size, stride,
                                         padding=(kernel_size - stride)//2,
                                         output_padding=stride-1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x, skip=None):
        # x: (batch, in_ch, t)
        if skip is not None and self.skip_conv is not None:
            skip = self.skip_conv(skip)   # (batch, out_ch, t_skip)
            # 对齐时间长度：裁剪到较小的长度
            min_len = min(x.size(2), skip.size(2))
            x = x[:, :, :min_len]
            skip = skip[:, :, :min_len]
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)                  # (b, out_ch*2, t)
        x = self.glu(x)                     # (b, out_ch, t)
        x = self.deconv(x)                  # (b, out_ch, t*stride)
        x = self.conv2(x)
        return x

class FreqDecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, skip_channels=0):
        super().__init__()
        time_ks, freq_ks = kernel_size
        time_s, freq_s = stride
        self.skip_conv = nn.Conv2d(skip_channels, out_channels, 1) if skip_channels > 0 else None
        self.glu = GLU(1)
        # 合并 skip 后的输入
        combined_in = in_channels + (out_channels if skip_channels>0 else 0)
        self.conv1 = nn.Conv2d(combined_in, out_channels*2, 1)
        # 转置卷积（频率维上采样）
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels,
                                         kernel_size=(time_ks, freq_ks),
                                         stride=(1, freq_s),
                                         padding=(time_ks//2, (freq_ks - freq_s)//2),
                                         output_padding=(0, freq_s-1))
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, x, skip=None):
        # x: (batch, in_ch, t, f)
        if skip is not None and self.skip_conv is not None:
            skip = self.skip_conv(skip)   # (batch, out_ch, t_skip, f_skip)
            # 对齐时间和频率维度
            min_t = min(x.size(2), skip.size(2))
            min_f = min(x.size(3), skip.size(3))
            x = x[:, :, :min_t, :min_f]
            skip = skip[:, :, :min_t, :min_f]
            x = torch.cat([x, skip], dim=1)
        # GLU 在通道维
        x = self.conv1(x)
        a, b = x.chunk(2, dim=1)
        x = a * torch.sigmoid(b)
        x = self.deconv(x)
        x = self.conv2(x)
        return x

# ---------- 跨域 Transformer ----------
class CrossTransformer(nn.Module):
    def __init__(self, dim, heads, layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleDict({
                'self_attn_time': nn.MultiheadAttention(dim, heads, dropout, batch_first=True),
                'self_attn_freq': nn.MultiheadAttention(dim, heads, dropout, batch_first=True),
                'cross_attn_t2f': nn.MultiheadAttention(dim, heads, dropout, batch_first=True),
                'cross_attn_f2t': nn.MultiheadAttention(dim, heads, dropout, batch_first=True),
                'ffn_time': nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim)),
                'ffn_freq': nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim)),
                'norm1_t': nn.LayerNorm(dim),
                'norm1_f': nn.LayerNorm(dim),
                'norm2_t': nn.LayerNorm(dim),
                'norm2_f': nn.LayerNorm(dim),
                'norm3_t': nn.LayerNorm(dim),
                'norm3_f': nn.LayerNorm(dim),
            }))

    def forward(self, time_feat, freq_feat):
        # time_feat: (batch, time, dim)
        # freq_feat: (batch, freq, dim) 假设频率维已压缩为1？但Demucs中bottleneck时频已对齐，频率维可能为1
        # 这里我们假设两者都是 (batch, seq, dim)
        for layer in self.layers:
            # 自注意力
            t2 = layer['norm1_t'](time_feat)
            t2, _ = layer['self_attn_time'](t2, t2, t2)
            time_feat = time_feat + t2

            f2 = layer['norm1_f'](freq_feat)
            f2, _ = layer['self_attn_freq'](f2, f2, f2)
            freq_feat = freq_feat + f2

            # 交叉注意力
            t2 = layer['norm2_t'](time_feat)
            f2 = layer['norm2_f'](freq_feat)
            t2_cross, _ = layer['cross_attn_t2f'](t2, f2, f2)
            f2_cross, _ = layer['cross_attn_f2t'](f2, t2, t2)
            time_feat = time_feat + t2_cross
            freq_feat = freq_feat + f2_cross

            # FFN
            t2 = layer['norm3_t'](time_feat)
            t2 = layer['ffn_time'](t2)
            time_feat = time_feat + t2

            f2 = layer['norm3_f'](freq_feat)
            f2 = layer['ffn_freq'](f2)
            freq_feat = freq_feat + f2

        return time_feat, freq_feat

# ---------- 扩散模块（简化版） ----------
class DiffusionModule(nn.Module):
    """
    在最底层特征上应用扩散过程。
    训练时：对特征加噪，预测噪声（DDPM风格）
    推理时：从纯噪声逐步去噪（可选步数）
    """
    def __init__(self, dim, steps=100, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.dim = dim
        self.steps = steps

        # 定义beta schedule
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, steps))
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        # 噪声预测网络（简单的MLP，可替换为更复杂的结构）
        self.net = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim)
        )

    def forward_train(self, x, t):
        """
        x: 干净特征，形状 (batch, seq_len, dim) 或 (batch, dim)
        t: 时间步，形状 (batch,)
        返回预测的噪声和真实噪声
        """
        noise = torch.randn_like(x)
        # 将系数 reshape 以匹配 x 的维度
        alpha_cumprod = self.sqrt_alphas_cumprod[t].view(-1, *([1] * (x.dim() - 1)))
        one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].view(-1, *([1] * (x.dim() - 1)))
        noisy = alpha_cumprod * x + one_minus_alpha_cumprod * noise
        pred_noise = self.net(noisy)
        return pred_noise, noise

    def forward_sample(self, x_T, steps=None):
        """
        推理时从噪声恢复
        x_T: 初始噪声
        steps: 使用的步数（<= self.steps）
        """
        steps = steps or self.steps
        x = x_T
        for i in reversed(range(steps)):
            t = torch.full((x.size(0),), i, device=x.device, dtype=torch.long)
            pred_noise = self.net(x)
            alpha = 1. - self.betas[t]
            alpha_cumprod = self.sqrt_alphas_cumprod[t]**2
            # 简单的DDIM采样（一步）
            x = (x - (1-alpha)/torch.sqrt(1-alpha_cumprod) * pred_noise) / torch.sqrt(alpha)
        return x

    def forward(self, x, t=None, reverse=False):
        if self.training:
            # 训练模式：输入干净特征，返回噪声损失
            return self.forward_train(x, t)
        else:
            # 推理模式：若 reverse=True 则从噪声恢复，否则直接返回输入（或可选）
            if reverse:
                return self.forward_sample(x)
            else:
                return x

# ---------- 主模型 ----------
class HybridDemucsWithDiffusion(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # ---------- 时域分支 ----------
        time_encoders = []
        in_ch = 1  # 单声道
        out_ch = config.time_channels
        for i in range(config.time_depth):
            use_lstm = (i >= config.time_depth - 2)  # 最后两层使用LSTM和注意力
            use_attn = (i >= config.time_depth - 2)
            layer = EncoderLayer(in_ch, out_ch,
                                 kernel_size=config.time_kernel_size,
                                 stride=config.time_stride,
                                 dilation=1,
                                 use_lstm=use_lstm,
                                 use_attn=use_attn)
            time_encoders.append(layer)
            in_ch = out_ch
            out_ch = min(out_ch * 2, config.transformer_dim)  # 通道数加倍，不超过transformer_dim
        self.time_encoders = nn.ModuleList(time_encoders)

        # 时域编码器的输出通道数
        self.time_encoder_out_ch = in_ch

        # ---------- 频域分支 ----------
        freq_encoders = []
        in_ch_f = 2  # STFT输出为实部和虚部两个通道
        out_ch_f = config.freq_channels
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
            in_ch_f = out_ch_f
            out_ch_f = min(out_ch_f * 2, config.transformer_dim)
        self.freq_encoders = nn.ModuleList(freq_encoders)
        self.freq_encoder_out_ch = in_ch_f

        # 频率位置嵌入，维度应与第一个频域编码层的输出通道数一致
        self.freq_pos = FreqPositionalEmbedding(config.freq_channels)

        # ---------- 共享编码层 ----------
        self.shared_encoder = nn.ModuleList()
        in_ch_s = self.time_encoder_out_ch + self.freq_encoder_out_ch  # 融合后
        out_ch_s = config.shared_channels
        for _ in range(config.shared_depth):
            self.shared_encoder.append(
                EncoderLayer(in_ch_s, out_ch_s,
                             kernel_size=config.shared_kernel_size,
                             stride=2,  # 下采样2倍
                             use_lstm=True,
                             use_attn=True)
            )
            in_ch_s = out_ch_s
            out_ch_s = min(out_ch_s * 2, config.transformer_dim)

        # ---------- Transformer 瓶颈 ----------
        self.transformer = CrossTransformer(
            dim=in_ch_s,
            heads=config.transformer_heads,
            layers=config.transformer_layers,
            dropout=config.transformer_dropout
        )

        # ---------- 扩散模块 ----------
        if config.use_diffusion:
            self.diffusion = DiffusionModule(
                dim=in_ch_s,
                steps=config.diffusion_steps,
                beta_start=config.diffusion_beta_start,
                beta_end=config.diffusion_beta_end
            )
        else:
            self.diffusion = None

        # ---------- 共享解码层 ----------
        self.shared_decoder = nn.ModuleList()
        # 逆序
        dec_in_ch = in_ch_s
        for i, enc in enumerate(reversed(self.shared_encoder)):
            # 解码器输入通道数 = 上一解码器输出 (或瓶颈) + 跳跃连接
            out_ch = enc.conv.in_channels  # 对应编码器的输入通道
            skip_ch = enc.conv.out_channels  # 编码器输出通道（跳跃连接）
            self.shared_decoder.append(
                DecoderLayer(dec_in_ch, out_ch,
                             kernel_size=config.shared_kernel_size,
                             stride=2,
                             skip_channels=skip_ch)
            )
            dec_in_ch = out_ch

        # ---------- 时域解码分支 ----------
        self.time_decoders = nn.ModuleList()
        # 从共享解码器输出开始，逐步上采样
        # 注意：共享解码器最后输出应与最顶层时域编码器输入通道数相同
        dec_in_ch_t = dec_in_ch  # 来自共享解码器的输出
        for i, enc in enumerate(reversed(self.time_encoders)):
            # 编码器对应层的输入通道（即跳跃连接通道）
            skip_ch = enc.conv.out_channels
            out_ch = enc.conv.in_channels
            self.time_decoders.append(
                DecoderLayer(dec_in_ch_t, out_ch,
                             kernel_size=config.time_kernel_size,
                             stride=config.time_stride,
                             skip_channels=skip_ch)
            )
            dec_in_ch_t = out_ch
        # 最终输出层
        self.time_out = nn.Conv1d(dec_in_ch_t, 1, 1)

        # ---------- 频域解码分支 ----------
        self.freq_decoders = nn.ModuleList()
        dec_in_ch_f = dec_in_ch  # 共享解码器输出（也用于频域）
        for i, enc in enumerate(reversed(self.freq_encoders)):
            skip_ch = enc.conv.out_channels
            out_ch = enc.conv.in_channels
            self.freq_decoders.append(
                FreqDecoderLayer(dec_in_ch_f, out_ch,
                                 kernel_size=config.freq_kernel_size,
                                 stride=config.freq_stride,
                                 skip_channels=skip_ch)
            )
            dec_in_ch_f = out_ch
        # 最终频域输出层，输出2通道（实部虚部）
        self.freq_out = nn.Conv2d(dec_in_ch_f, 2, 1)

    def forward(self, waveform, return_diffusion_loss=False):
        """
        waveform: (batch, 1, T)
        返回分离的吉他波形 (batch, 1, T)
        如果 return_diffusion_loss=True 且 diffusion 启用，额外返回扩散损失
        """
        # ---------- 时域编码 ----------
        t = waveform
        time_skips = []
        for enc in self.time_encoders:
            t = enc(t)
            time_skips.append(t)

        # ---------- 频域编码 ----------
        # 计算STFT
        spec = stft(waveform,
                    fft_size=self.config.stft_fft_size,
                    hop_length=self.config.stft_hop_length,
                    win_length=self.config.stft_win_length,
                    window=self.config.stft_window,
                    normalized=self.config.stft_normalized)  # (b, 2, time, freq)
        f = spec
        freq_skips = []
        for i, enc in enumerate(self.freq_encoders):
            if i == 1:  # 在第二层后加入频率位置嵌入（论文做法）
                freq_idx = torch.arange(f.size(3), device=f.device)
                f = self.freq_pos(f, freq_idx)
            f = enc(f)
            freq_skips.append(f)

        # ---------- 对齐维度并融合 ----------
        # 时域特征 t: (b, c_t, t_len)
        # 频域特征 f: (b, c_f, t_len, f_len) 其中 f_len 应为1（经过频率下采样后）
        # 论文中频域分支最终将频率维压缩为1，这里假设已经为1，否则进行全局平均池化
        if f.size(3) != 1:
            f = f.mean(dim=3, keepdim=True)  # 平均频率维
        f = f.squeeze(3)  # (b, c_f, t_len)

        # 对齐时间长度：时域t的长度可能因步长不同而有微小差异，截断或填充至相同
        min_len = min(t.size(2), f.size(2))
        t = t[:, :, :min_len]
        f = f[:, :, :min_len]

        # 融合
        combined = torch.cat([t, f], dim=1)  # (b, c_t+c_f, t_len)

        # ---------- 共享编码 ----------
        shared_skips = []
        for enc in self.shared_encoder:
            combined = enc(combined)
            shared_skips.append(combined)

        # ---------- Transformer 瓶颈 ----------
        # 将 combined 转为序列格式 (b, t, c)
        bottleneck = combined.permute(0, 2, 1)  # (b, t, c)
        # 分别作为时域和频域特征传入（实际上相同，但Transformer会分别处理）
        time_feat, freq_feat = self.transformer(bottleneck, bottleneck)

        # ---------- 扩散模块 ----------
        diffusion_loss = None
        if self.diffusion is not None:
            if self.training and return_diffusion_loss:
                # 训练时：对特征加噪并预测噪声
                t = torch.randint(0, self.config.diffusion_steps, (bottleneck.size(0),), device=bottleneck.device)
                pred_noise, noise = self.diffusion.forward_train(bottleneck, t)
                diffusion_loss = F.mse_loss(pred_noise, noise)
                # 可选：将去噪后的特征传回
                # 简单起见，仍使用原始特征
            else:
                # 推理时：可选择应用扩散去噪
                if hasattr(self, 'apply_diffusion_inference') and self.apply_diffusion_inference:
                    bottleneck = self.diffusion.forward_sample(bottleneck)
                # 否则保持原样

        # 合并回 combined 形状
        combined = (time_feat + freq_feat) / 2  # 平均
        combined = combined.permute(0, 2, 1)  # (b, c, t)

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

        # ---------- 频域解码 ----------
        f_dec = combined
        # 需要恢复频率维
        f_dec = f_dec.unsqueeze(3)  # (b, c, t, 1)
        for i, dec in enumerate(self.freq_decoders):
            skip = freq_skips[-(i+1)]
            f_dec = dec(f_dec, skip)
        spec_out = self.freq_out(f_dec)  # (b, 2, t, f)

        # 确保频率维度正确（应为 n_fft/2+1）
        expected_freq = self.config.stft_fft_size // 2 + 1
        if spec_out.size(3) != expected_freq:
            spec_out = F.interpolate(spec_out, size=(spec_out.size(2), expected_freq), 
                                    mode='bilinear', align_corners=False)
            
        # 通过 ISTFT 还原波形
        # 注意：需要知道原始长度，可能需要对spec进行长度调整
        waveform_length = waveform.size(2)
        freq_out = istft(spec_out,
                        fft_size=self.config.stft_fft_size,
                        hop_length=self.config.stft_hop_length,
                        win_length=self.config.stft_win_length,
                        window=self.config.stft_window,
                        normalized=self.config.stft_normalized)  # 不指定长度，让istft自动计算
        # 自动计算的长度可能与原始长度不同，需要对齐
        if freq_out.size(2) > waveform_length:
            freq_out = freq_out[:, :, :waveform_length]  # 裁剪
        elif freq_out.size(2) < waveform_length:
            # 填充零
            pad = waveform_length - freq_out.size(2)
            freq_out = torch.nn.functional.pad(freq_out, (0, pad))

        # 融合两个分支的输出
        # 对齐时间维度
        min_len = min(time_out.size(2), freq_out.size(2))
        time_out = time_out[:, :, :min_len]
        freq_out = freq_out[:, :, :min_len]
        final_wave = time_out + freq_out

        if diffusion_loss is not None:
            return final_wave, diffusion_loss
        return final_wave