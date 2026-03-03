import torch
import torchaudio

def stft(x, fft_size, hop_length, win_length, window='hann', normalized=False):
    """返回复数谱，形状 (batch, freq, time, 2) 实部和虚部拼接"""
    window = torch.hann_window(win_length, device=x.device) if window == 'hann' else torch.ones(win_length, device=x.device)
    spec = torch.stft(
        x.squeeze(1), n_fft=fft_size, hop_length=hop_length,
        win_length=win_length, window=window, center=True,
        normalized=normalized, return_complex=True
    )  # (batch, freq, time)
    # 转为实部和虚部拼接 (batch, freq, time, 2)
    spec = torch.view_as_real(spec)  # (batch, freq, time, 2)
    # 转置为 (batch, time, freq, 2) 便于后续卷积（时间作为序列维）
    spec = spec.permute(0, 3, 2, 1)  # (batch, 2, time, freq)
    return spec

def istft(spec, fft_size, hop_length, win_length, window='hann', normalized=False, length=None):
    """输入 spec 形状 (batch, 2, time, freq) 或 (batch, freq, time, 2) 取决于输入"""
    if spec.dim() == 4 and spec.size(1) == 2:
        # 输入为 (batch, 2, time, freq) -> 转为 (batch, freq, time, 2)
        spec = spec.permute(0, 3, 2, 1)  # (batch, freq, time, 2)
    spec = torch.view_as_complex(spec.contiguous())  # (batch, freq, time)
    window = torch.hann_window(win_length, device=spec.device) if window == 'hann' else torch.ones(win_length, device=spec.device)
    waveform = torch.istft(
        spec, n_fft=fft_size, hop_length=hop_length,
        win_length=win_length, window=window, center=True,
        normalized=normalized, length=length
    )
    return waveform.unsqueeze(1)  # (batch, 1, T)