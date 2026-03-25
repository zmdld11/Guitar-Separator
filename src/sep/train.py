import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # 忽略用户警告

import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime
import random

from src.sep.config import Config
from src.sep.dataset import SepDataset
from src.sep.model import HTDemucs
from src.sep.utils import compute_sdr

# ---------- 多分辨率STFT损失 ----------
def multi_resolution_stft_loss(est_wave, target_wave, fft_sizes=[2048, 1024, 512], hop_sizes=None, win_sizes=None):
    """
    计算多个STFT分辨率的谱收敛损失(Spectral Convergence)、复数L1损失和对数幅度L1损失。
    V3.0: 既然模型有了相位修正能力，我们重新引入对复数实虚部的损失惩罚，以及波形级别的约束。
    """
    if hop_sizes is None:
        hop_sizes = [fft_size//4 for fft_size in fft_sizes]
    if win_sizes is None:
        win_sizes = fft_sizes
    loss = 0.0
    
    est_wave = est_wave.float()
    target_wave = target_wave.float()
    
    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        window = torch.hann_window(win_size).to(est_wave.device)
        est_spec = torch.stft(est_wave.squeeze(1), n_fft=fft_size, hop_length=hop_size,
                              win_length=win_size, window=window, return_complex=True)
        target_spec = torch.stft(target_wave.squeeze(1), n_fft=fft_size, hop_length=hop_size,
                                 win_length=win_size, window=window, return_complex=True)
        
        est_mag = torch.sqrt(est_spec.real**2 + est_spec.imag**2 + 1e-8)
        target_mag = torch.sqrt(target_spec.real**2 + target_spec.imag**2 + 1e-8)
        
        # 1. 谱收敛损失 (Spectral Convergence)
        sc_loss = torch.norm(target_mag - est_mag, p="fro") / (torch.norm(target_mag, p="fro") + 1e-8)
        
        # 2. 对数幅度损失 (Log Magnitude L1)
        log_mag_loss = torch.mean(torch.abs(torch.log(est_mag + 1e-7) - torch.log(target_mag + 1e-7)))
        
        # 3. 复数实虚部损失 (Complex L1) - 施加相位对齐压力
        complex_loss = torch.mean(torch.abs(est_spec.real - target_spec.real)) + \
                       torch.mean(torch.abs(est_spec.imag - target_spec.imag))
        
        loss += (sc_loss + log_mag_loss + complex_loss)
        
    return loss / len(fft_sizes)

def train():
    config = Config()

    # 创建日志目录
    log_dir = config.log_dir
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"log_{timestamp}.txt")

    def log_message(msg, also_print=True):
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')
        if also_print:
            print(msg)

    log_message("=" * 60)
    log_message(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message("=" * 60)
    log_message("Configuration:")
    for key, value in config.__dict__.items():
        if not key.startswith('_'):
            log_message(f"  {key}: {value}")
    log_message("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_message(f'Using device: {device}')

    # 数据集
    train_dataset = SepDataset(config, split='train')
    val_dataset = SepDataset(config, split='val')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=config.num_workers)
    log_message(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')

    # 模型
    model = HTDemucs(config).to(device)
    log_message(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

    # 损失函数
    l1_loss = nn.L1Loss()

    # 混合精度
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)

    best_val_sdr = -float('inf')
    nan_counter = 0
    max_nan_epochs = 5

    # 数据增强函数
    def augment(mix, guitar):
        if not config.use_augmentation:
            return mix, guitar
            
        # 1. 批次内的随机混音 (Remixing / Mixup) - 解决数据量匮乏的最强增强
        # 通过将 batch 内的 mix - guitar 得到其他伴奏，然后随机打乱伴奏与吉他的搭配
        # 这要求 batch size > 1
        if mix.size(0) > 1 and random.random() < 0.7:  # 70%概率进行批次内伴奏重组
            other_instruments = mix - guitar
            # 将 other_instruments 沿 batch 维度随机滚动（打乱搭配）
            roll_shift = random.randint(1, mix.size(0) - 1)
            other_instruments_shuffled = torch.roll(other_instruments, shifts=roll_shift, dims=0)
            mix = guitar + other_instruments_shuffled  # 重新合并为新的 mix

        gain = random.uniform(*config.gain_augment_range)
        mix = mix * gain
        guitar = guitar * gain
        if mix.size(1) > 1 and random.random() < config.channel_swap_prob:
            mix = torch.flip(mix, dims=[1])
            guitar = torch.flip(guitar, dims=[1])
        noise = torch.randn_like(mix) * config.noise_floor
        mix = mix + noise
        guitar = guitar + noise
        return mix, guitar

    start_epoch = 1

    # 断点续训逻辑
    if hasattr(config, 'resume_training') and config.resume_training:
        checkpoint_path = config.resume_checkpoint if getattr(config, 'resume_checkpoint', '') else os.path.join(config.checkpoint_dir, 'last_checkpoint.pth')
        if os.path.exists(checkpoint_path):
            log_message(f"Loading checkpoint from {checkpoint_path}...")
            try:
                # 明确指定 weights_only=False 静音 FutureWarning (本地模型安全)
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_val_sdr = checkpoint.get('best_val_sdr', checkpoint.get('val_sdr', -float('inf')))
                if 'scaler_state_dict' in checkpoint and scaler is not None:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                log_message(f"Successfully resumed training from epoch {start_epoch - 1}. Best SDR so far: {best_val_sdr:.2f}")
            except Exception as e:
                log_message(f"Failed to load checkpoint: {e}. Starting from scratch.")
        else:
            log_message(f"Checkpoint not found at {checkpoint_path}. Starting from scratch.")

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        train_loss = 0.0
        valid_batches = 0
        nan_batches = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, (mix, guitar) in enumerate(progress_bar):
            mix = mix.to(device)
            guitar = guitar.to(device)

            mix, guitar = augment(mix, guitar)

            if torch.isnan(mix).any() or torch.isnan(guitar).any():
                log_message(f"⚠️ Batch {batch_idx} contains NaN in input, skipping", also_print=False)
                nan_batches += 1
                continue

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=config.use_amp):
                out = model(mix)
                
                # 重新加入波形级损失作为细微相位偏移的补充约束
                loss_l1 = l1_loss(out, guitar)
                loss_stft = multi_resolution_stft_loss(out, guitar)
                
                loss = loss_l1 + loss_stft

            if torch.isnan(loss).any():
                log_message(f"⚠️ Batch {batch_idx} loss is NaN, skipping", also_print=False)
                nan_batches += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            valid_batches += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        if nan_batches > 0:
            log_message(f"Epoch {epoch}: {nan_batches} batches were skipped due to NaN")

        if valid_batches > 0:
            avg_train_loss = train_loss / valid_batches
            log_message(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}')
        else:
            log_message(f'Epoch {epoch}: All batches were NaN, skipping epoch')
            nan_counter += 1
            if nan_counter >= max_nan_epochs:
                log_message(f"❌ {max_nan_epochs} consecutive NaN epochs, stopping training.")
                break
            continue

        # 验证
        model.eval()
        val_sdr_list = []
        with torch.no_grad():
            for mix, guitar in tqdm(val_loader, desc='Validating'):
                mix = mix.to(device)
                guitar = guitar.to(device)

                if torch.isnan(mix).any() or torch.isnan(guitar).any():
                    continue

                with torch.amp.autocast('cuda', enabled=config.use_amp):
                    out = model(mix)
                if torch.isnan(out).any():
                    continue

                sdr = compute_sdr(out, guitar)
                if not np.isnan(sdr) and not np.isinf(sdr):
                    val_sdr_list.append(sdr)

        if len(val_sdr_list) > 0:
            val_sdr = np.mean(val_sdr_list)
            log_message(f'Epoch {epoch}: Val SDR = {val_sdr:.2f} dB')
        else:
            val_sdr = -float('inf')
            log_message(f'Epoch {epoch}: No valid SDR values')

        scheduler.step()

        if val_sdr > best_val_sdr:
            best_val_sdr = val_sdr
            torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, 'best_model.pth'))
            log_message(f'New best model saved with SDR {val_sdr:.2f}')
            nan_counter = 0

        # 保存最新检查点，用于断点续训
        last_checkpoint_path = os.path.join(config.checkpoint_dir, 'last_checkpoint.pth')
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_sdr': val_sdr,
            'best_val_sdr': best_val_sdr,
        }
        torch.save(checkpoint_dict, last_checkpoint_path)

        # 调高了定期保存的频率，并且使用拷贝方式
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch{epoch}.pth')
            import shutil
            shutil.copyfile(last_checkpoint_path, checkpoint_path)
            log_message(f'Checkpoint saved to {checkpoint_path}')

    log_message("=" * 60)
    log_message(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Best validation SDR: {best_val_sdr:.2f} dB")
    log_message("=" * 60)

if __name__ == '__main__':
    train()