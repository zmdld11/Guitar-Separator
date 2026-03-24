import torch
import torchaudio
import numpy as np
import warnings
from mir_eval.separation import bss_eval_sources

def compute_sdr(estimated, target):
    """
    计算单通道SDR（信号失真比）
    estimated, target: 形状为 (batch, 1, T) 或 (1, T)
    """
    # 将张量转为 numpy 并压缩所有为1的维度
    estimated = estimated.cpu().detach().numpy().squeeze()
    target = target.cpu().detach().numpy().squeeze()
    
    # 确保形状为 (1, T)
    if estimated.ndim == 1:
        estimated = estimated[np.newaxis, :]
        target = target[np.newaxis, :]
    elif estimated.ndim == 2 and estimated.shape[0] == 1:
        pass  # 已经是 (1, T)
    else:
        raise ValueError(f"Unexpected shape: {estimated.shape}")
    
    # mir_eval 要求形状 (nsrc, nsample)
    # 屏蔽 bss_eval_sources 即将在 0.9 版本移除的警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sdr, sir, sar, _ = bss_eval_sources(target, estimated)
        
    return float(sdr[0])

def save_audio(waveform, path, sample_rate=44100):
    """保存波形到文件"""
    torchaudio.save(path, waveform.cpu(), sample_rate)