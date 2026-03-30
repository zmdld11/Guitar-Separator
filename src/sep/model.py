import torch
import torch.nn as nn
from .config import Config
from ..common.audio_utils import stft, istft

def get_rope_cos_sin(seq_len, head_dim, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).type_as(inv_freq)
    freqs = torch.outer(t, inv_freq)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)
    return emb.cos()[None, :, None, :], emb.sin()[None, :, None, :]

def apply_rope(x, cos, sin):
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    x_rot = torch.stack([-x_odd, x_even], dim=-1).reshape_as(x)
    return x * cos + x_rot * sin

class RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x, cos, sin):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class TransformerLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPEAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
        
    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x

class DualPathBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.time_trans = TransformerLayer(dim, num_heads)
        self.freq_trans = TransformerLayer(dim, num_heads)
    
    def forward(self, x, cos_t, sin_t, cos_f, sin_f):
        B, K, T, D = x.shape
        x_t = x.permute(0, 1, 2, 3).reshape(B*K, T, D)
        x_t = self.time_trans(x_t, cos_t, sin_t)
        x = x_t.reshape(B, K, T, D)
        
        x_f = x.permute(0, 2, 1, 3).reshape(B*T, K, D)
        x_f = self.freq_trans(x_f, cos_f, sin_f)
        x = x_f.reshape(B, T, K, D).permute(0, 2, 1, 3) 
        return x

class MiniBSRoFormer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        dim, num_heads, num_blocks = 128, 4, 4
        self.bounds = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1025]
        self.num_bands = len(self.bounds) - 1
        
        self.in_projs, self.out_projs = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_bands):
            bw = self.bounds[i+1] - self.bounds[i]
            self.in_projs.append(nn.Linear(bw * 2, dim))
            self.out_projs.append(nn.Linear(dim, bw * 2))
            
        self.blocks = nn.ModuleList([DualPathBlock(dim, num_heads) for _ in range(num_blocks)])
        self.dim, self.num_heads = dim, num_heads

    def forward(self, waveform):
        input_length = waveform.size(2)
        spec = stft(waveform, fft_size=self.config.stft_fft_size, hop_length=self.config.stft_hop_length, win_length=self.config.stft_win_length, window=self.config.stft_window, normalized=self.config.stft_normalized)
        mag = torch.sqrt(spec[:, 0:1]**2 + spec[:, 1:2]**2 + 1e-8)
        phase_angle = torch.atan2(spec[:, 1:2], spec[:, 0:1])
        x_in = torch.cat([torch.log1p(mag), phase_angle / torch.pi], dim=1).permute(0, 3, 2, 1)
        B, F, T, _ = x_in.shape
        
        band_features = []
        for i in range(self.num_bands):
            low, high = self.bounds[i], self.bounds[i+1]
            bw = high - low
            x_band = x_in[:, low:high].permute(0, 2, 1, 3).reshape(B, T, bw * 2)
            band_features.append(self.in_projs[i](x_band))
            
        x = torch.stack(band_features, dim=1)
        head_dim = self.dim // self.num_heads
        cos_t, sin_t = get_rope_cos_sin(max(T, 1000), head_dim, x.device)
        cos_f, sin_f = get_rope_cos_sin(self.num_bands, head_dim, x.device)
        cos_t, sin_t = cos_t[:, :T], sin_t[:, :T]
        cos_f, sin_f = cos_f[:, :self.num_bands], sin_f[:, :self.num_bands]
        
        for block in self.blocks:
            x = block(x, cos_t, sin_t, cos_f, sin_f)
            
        out_bands = []
        for i in range(self.num_bands):
            bw = self.bounds[i+1] - self.bounds[i]
            out_band = self.out_projs[i](x[:, i]).reshape(B, T, bw, 2).permute(0, 3, 1, 2)
            out_bands.append(out_band)
            
        out = torch.cat(out_bands, dim=3)
        est_mag = mag * torch.sigmoid(out[:, 0:1])
        est_phase = phase_angle + torch.tanh(out[:, 1:2]) * torch.pi
        est_spec = torch.cat([est_mag * torch.cos(est_phase), est_mag * torch.sin(est_phase)], dim=1)
        return istft(est_spec, fft_size=self.config.stft_fft_size, hop_length=self.config.stft_hop_length, win_length=self.config.stft_win_length, window=self.config.stft_window, normalized=self.config.stft_normalized, length=input_length)
