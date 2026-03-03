import torch
import torchaudio
import numpy as np
from mir_eval.separation import bss_eval_sources

def compute_sdr(estimated, target):
    """
    计算单通道SDR（信号失真比）
    estimated, target: (1, T) 或 (T,)
    """
    if estimated.dim() == 2:
        estimated = estimated.squeeze(0).cpu().numpy()
        target = target.squeeze(0).cpu().numpy()
    else:
        estimated = estimated.cpu().numpy()
        target = target.cpu().numpy()

    # mir_eval 要求形状 (nsrc, nsample)
    sdr, sir, sar, _ = bss_eval_sources(target[np.newaxis, :], estimated[np.newaxis, :])
    return float(sdr[0])

def save_audio(waveform, path, sample_rate=44100):
    """保存波形到文件"""
    torchaudio.save(path, waveform.cpu(), sample_rate)