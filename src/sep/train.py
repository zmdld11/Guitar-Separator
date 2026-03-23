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
from torch.cuda.amp import GradScaler, autocast
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
    计算多个STFT分辨率的L1损失（实部和虚部之和）
    返回所有分辨率的平均损失
    """
    if hop_sizes is None:
        hop_sizes = [fft_size//4 for fft_size in fft_sizes]
    if win_sizes is None:
        win_sizes = fft_sizes
    loss = 0.0
    for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
        window = torch.hann_window(win_size).to(est_wave.device)
        est_spec = torch.stft(est_wave.squeeze(1), n_fft=fft_size, hop_length=hop_size,
                              win_length=win_size, window=window, return_complex=True)
        target_spec = torch.stft(target_wave.squeeze(1), n_fft=fft_size, hop_length=hop_size,
                                 win_length=win_size, window=window, return_complex=True)
        loss += torch.mean(torch.abs(est_spec.real - target_spec.real)) + \
                torch.mean(torch.abs(est_spec.imag - target_spec.imag))
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

    for epoch in range(1, config.epochs + 1):
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
                loss_l1 = l1_loss(out, guitar)
                loss_stft = multi_resolution_stft_loss(out, guitar)
                loss = loss_l1 + 0.5 * loss_stft

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

                with autocast(enabled=config.use_amp):
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

        if epoch % 50 == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_epoch{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_sdr': val_sdr,
            }, checkpoint_path)
            log_message(f'Checkpoint saved to {checkpoint_path}')

    log_message("=" * 60)
    log_message(f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Best validation SDR: {best_val_sdr:.2f} dB")
    log_message("=" * 60)

if __name__ == '__main__':
    train()