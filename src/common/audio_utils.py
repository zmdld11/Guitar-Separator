import torch
import torchaudio

def stft(x, fft_size, hop_length, win_length, window='hann', normalized=False):
    """返回复数谱,形状 (batch, 2, time, freq)"""
    if window == 'hann':
        window_tensor = torch.hann_window(win_length, device=x.device)
    else:
        window_tensor = torch.ones(win_length, device=x.device)
    
    # 确保输入是2D
    x_2d = x.squeeze(1) if x.dim() == 3 else x
    
    spec = torch.stft(
        x_2d, 
        n_fft=fft_size, 
        hop_length=hop_length,
        win_length=win_length, 
        window=window_tensor, 
        center=True,
        normalized=normalized, 
        onesided=True,
        return_complex=True
    )  # (batch, freq, time)
    
    # 转为实部和虚部拼接
    spec = torch.view_as_real(spec)  # (batch, freq, time, 2)
    spec = spec.permute(0, 3, 2, 1)  # (batch, 2, time, freq)
    return spec


def istft(spec, fft_size, hop_length, win_length, window='hann', normalized=False, length=None):
    """输入 spec 形状 (batch, 2, time, freq)"""
    if spec.dim() == 4 and spec.size(1) == 2:
        spec = spec.permute(0, 3, 2, 1)  # (batch, freq, time, 2)
    
    # 转换为复数
    spec_complex = torch.view_as_complex(spec.contiguous())  # (batch, freq, time)
    
    if window == 'hann':
        window_tensor = torch.hann_window(win_length, device=spec.device)
    else:
        window_tensor = torch.ones(win_length, device=spec.device)
    
    waveform = torch.istft(
        spec_complex, 
        n_fft=fft_size, 
        hop_length=hop_length,
        win_length=win_length, 
        window=window_tensor, 
        center=True,
        normalized=normalized, 
        onesided=True,
        length=length,
        return_complex=False
    )
    
    return waveform.unsqueeze(1)  # (batch, 1, T)