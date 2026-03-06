# src/sep/train.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from datetime import datetime

from src.sep.config import Config
from src.sep.dataset import SepDataset
from src.sep.model import HybridDemucsWithDiffusion
from src.sep.utils import compute_sdr


def normalize_audio(waveform, eps=1e-8):
    """将音频标准化到均值为0，标准差为1"""
    mean = waveform.mean(dim=-1, keepdim=True)
    std = waveform.std(dim=-1, keepdim=True) + eps
    return (waveform - mean) / std


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
    model = HybridDemucsWithDiffusion(config).to(device)
    log_message(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    # 优化器（学习率已调低）
    optimizer = optim.Adam(model.parameters(), lr=1e-4,  # 原为 config.learning_rate (3e-4)
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                      factor=0.5, patience=10)

    # 损失函数
    criterion = nn.SmoothL1Loss()
    if config.use_diffusion:
        diffusion_weight = 0.01  # 可进一步调低，如 0.005

    # 混合精度
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)

    best_val_sdr = -float('inf')
    nan_counter = 0
    max_nan_epochs = 5

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_diff_loss = 0.0
        valid_batches = 0
        nan_batches = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, (mix, guitar) in enumerate(progress_bar):
            mix = mix.to(device)
            guitar = guitar.to(device)

            # 输入标准化（可选，但强烈建议）
            mix = normalize_audio(mix)
            guitar = normalize_audio(guitar)

            # 检查输入是否有 NaN
            if torch.isnan(mix).any() or torch.isnan(guitar).any():
                log_message(f"⚠️ Batch {batch_idx} contains NaN in input, skipping", also_print=False)
                nan_batches += 1
                continue

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=config.use_amp):
                if config.use_diffusion:
                    out, diff_loss = model(mix, return_diffusion_loss=True)
                    loss = criterion(out, guitar) + diffusion_weight * diff_loss
                else:
                    out = model(mix)
                    loss = criterion(out, guitar)

            # 检查损失是否为 NaN
            if torch.isnan(loss).any():
                log_message(f"⚠️ Batch {batch_idx} loss is NaN, skipping", also_print=False)
                nan_batches += 1
                continue

            # 检查输出是否有 NaN（调试用）
            if torch.isnan(out).any():
                log_message(f"⚠️ Batch {batch_idx} model output contains NaN, skipping", also_print=False)
                nan_batches += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            # 检查参数是否有 NaN（可选）
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    log_message(f"❌ Parameter {name} became NaN after step! Skipping batch.")
                    nan_batches += 1
                    # 可以选择回滚到之前的状态，但这里简单跳过
                    break

            train_loss += loss.item()
            if config.use_diffusion:
                train_diff_loss += diff_loss.item()
            valid_batches += 1

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 统计 NaN batch 数量
        if nan_batches > 0:
            log_message(f"Epoch {epoch}: {nan_batches} batches were skipped due to NaN")

        if valid_batches > 0:
            avg_train_loss = train_loss / valid_batches
            if config.use_diffusion:
                avg_train_diff = train_diff_loss / valid_batches
                log_message(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Diff Loss = {avg_train_diff:.6f}')
            else:
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

                # 同样对验证集输入做标准化
                mix = normalize_audio(mix)
                guitar = normalize_audio(guitar)

                if torch.isnan(mix).any() or torch.isnan(guitar).any():
                    log_message("⚠️ Validation sample contains NaN, skipping", also_print=False)
                    continue

                with autocast(enabled=config.use_amp):
                    out = model(mix)
                if torch.isnan(out).any():
                    log_message("⚠️ Validation output contains NaN, skipping", also_print=False)
                    continue

                sdr = compute_sdr(out, guitar)
                if not np.isnan(sdr) and not np.isinf(sdr):
                    val_sdr_list.append(sdr)
                else:
                    log_message(f"⚠️ Skipping SDR value {sdr}", also_print=False)

        if len(val_sdr_list) > 0:
            val_sdr = np.mean(val_sdr_list)
            log_message(f'Epoch {epoch}: Val SDR = {val_sdr:.2f} dB')
        else:
            val_sdr = -float('inf')
            log_message(f'Epoch {epoch}: No valid SDR values')

        scheduler.step(val_sdr if val_sdr != -float('inf') else best_val_sdr)

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