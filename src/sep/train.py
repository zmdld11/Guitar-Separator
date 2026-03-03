import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb  # 可选，用于可视化
from src.sep.config import Config
from src.sep.dataset import SepDataset
from src.sep.model import HybridDemucsWithDiffusion
from src.sep.utils import compute_sdr, save_audio

def train():
    config = Config()

    # 初始化wandb（可选）
    # wandb.init(project='guitar-separation', config=config)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据集
    train_dataset = SepDataset(config, split='train')
    val_dataset = SepDataset(config, split='val')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1,  # 验证时batch=1便于计算SDR
                            shuffle=False, num_workers=config.num_workers)

    # 模型
    model = HybridDemucsWithDiffusion(config).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                      factor=0.5, patience=10)

    # 损失函数
    criterion = nn.L1Loss()  # 波形L1损失
    if config.use_diffusion:
        diffusion_weight = 0.01  # 扩散损失权重

    # 混合精度
    scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp)

    # 训练循环
    best_val_sdr = -float('inf')
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_diff_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, (mix, guitar) in enumerate(progress_bar):
            mix = mix.to(device)
            guitar = guitar.to(device)

            with torch.amp.autocast('cuda', enabled=config.use_amp):
                if config.use_diffusion:
                    out, diff_loss = model(mix, return_diffusion_loss=True)
                    loss = criterion(out, guitar) + diffusion_weight * diff_loss
                else:
                    out = model(mix)
                    loss = criterion(out, guitar)

            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.accumulate_grad_batches == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()
            if config.use_diffusion:
                train_diff_loss += diff_loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        if config.use_diffusion:
            avg_train_diff = train_diff_loss / len(train_loader)
            print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Diff Loss = {avg_train_diff:.6f}')
        else:
            print(f'Epoch {epoch}: Train Loss = {avg_train_loss:.4f}')

        # 验证
        model.eval()
        val_sdr = 0.0
        with torch.no_grad():
            for mix, guitar in tqdm(val_loader, desc='Validating'):
                mix = mix.to(device)
                guitar = guitar.to(device)
                with autocast(enabled=config.use_amp):
                    out = model(mix)
                sdr = compute_sdr(out, guitar)  # 计算SDR
                val_sdr += sdr
        val_sdr /= len(val_loader)
        print(f'Epoch {epoch}: Val SDR = {val_sdr:.2f} dB')

        # 调整学习率
        scheduler.step(val_sdr)

        # 保存最佳模型
        if val_sdr > best_val_sdr:
            best_val_sdr = val_sdr
            torch.save(model.state_dict(), os.path.join(config.checkpoint_dir, 'best_model.pth'))
            print(f'New best model saved with SDR {val_sdr:.2f}')

        # 定期保存检查点
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_sdr': val_sdr,
            }, os.path.join(config.checkpoint_dir, f'checkpoint_epoch{epoch}.pth'))

        # wandb log #（可选）
        # wandb.log({'train_loss': avg_train_loss, 'val_sdr': val_sdr})

    print('Training completed.')

if __name__ == '__main__':
    train()