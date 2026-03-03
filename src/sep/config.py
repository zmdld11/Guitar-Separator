import os
from dataclasses import dataclass

@dataclass
class Config:
    # ---------- 路径配置 ----------
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_root = os.path.join(project_root, 'data/extract')
    mix_dir = os.path.join(data_root, 'separation/mix_segments')
    guitar_dir = os.path.join(data_root, 'separation/guitar_segments')
    checkpoint_dir = os.path.join(project_root, 'sep_model/checkpoints')
    log_dir = os.path.join(project_root, 'sep_model/logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ---------- 音频参数 ----------
    sample_rate = 44100
    duration = 6.0                     # 训练片段长度（秒）
    n_samples = int(sample_rate * duration)

    # ---------- STFT 参数（频域分支） ----------
    stft_fft_size = 4096
    stft_hop_length = 1024
    stft_win_length = 4096
    stft_normalized = False
    stft_window = 'hann'

    # ---------- 模型结构参数 ----------
    # 时域分支
    time_channels = 64                  # 初始通道数
    time_depth = 5                      # 编码器层数（不包括共享层）
    time_kernel_size = 8
    time_stride = 4

    # 频域分支
    freq_channels = 64
    freq_depth = 5
    freq_kernel_size = (8, 4)           # (时间方向, 频率方向) 卷积核
    freq_stride = (4, 4)                 # (时间, 频率) 步长

    # 共享层
    shared_channels = 128
    shared_depth = 1                     # 额外共享层数（每层下采样2倍）
    shared_kernel_size = 4

    # Transformer 参数（最底层）
    transformer_dim = 512                 # 与最底层通道数一致
    transformer_heads = 8
    transformer_layers = 4
    transformer_dropout = 0.1

    # 扩散模块参数（可选）
    use_diffusion = True
    diffusion_steps = 100                 # 训练时使用的扩散步数（DDPM）
    diffusion_beta_start = 1e-4
    diffusion_beta_end = 0.02
    diffusion_dim = transformer_dim       # 扩散网络隐藏维度

    # ---------- 训练参数 ----------
    batch_size = 4                         # 根据实际GPU调整
    num_workers = 8
    epochs = 300
    learning_rate = 3e-4
    weight_decay = 0.0
    gradient_clip = 5.0
    accumulate_grad_batches = 2            # 梯度累积（等效batch_size = batch_size * accumulate）
    use_amp = True                          # 混合精度训练
    save_top_k = 3
    monitor_metric = 'val_sdr'              # 监控验证SDR
    monitor_mode = 'max'

    # 验证集比例
    val_ratio = 0.1
    seed = 42