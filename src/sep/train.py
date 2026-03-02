# src/sep/train.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)

from src.sep.config import GuitarSeparationConfig
from src.sep.dataset import SeparationDataLoader
from src.sep.model import GuitarSeparationModel

class SeparationTrainer:
    """吉他分离模型训练器"""
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        
        # 简单的损失函数
        self.loss_fn = nn.L1Loss()  # 使用L1损失，更稳定
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 混合精度训练
        self.scaler = GradScaler(enabled=config.USE_AMP)
        
        # 训练历史
        self.history = {
            'train_loss': [], 'val_loss': [],
            'learning_rates': []
        }
        
        # 创建目录
        config.create_directories()
    
    def safe_loss(self, pred, target):
        """安全的损失计算，确保形状一致"""
        # 确保预测和目标形状一致
        if pred.shape != target.shape:
            min_len = min(pred.shape[2], target.shape[2])
            
            # 裁剪到相同长度
            pred = pred[:, :, :min_len]
            target = target[:, :, :min_len]
            
            if pred.shape[2] != target.shape[2]:
                print(f"⚠️  警告: 无法匹配形状, pred={pred.shape}, target={target.shape}")
                # 如果仍然不匹配，返回0损失
                return torch.tensor(0.0, device=pred.device)
        
        return self.loss_fn(pred, target)
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        pbar = tqdm(train_loader, desc="🎸 训练批次")
        for batch_idx, (mix, target) in enumerate(pbar):
            mix = mix.to(self.device)
            target = target.to(self.device)
            
            # 打印第一个batch的形状用于调试
            # if batch_idx == 0:
            #     print(f"🎵 Batch {batch_idx} 形状:")
            #     print(f"  Mix: {mix.shape}")
            #     print(f"  Target: {target.shape}")
            #     print(f"  期望输入通道数: {self.config.N_CHANNELS}")
            
            # 混合精度训练
            with autocast(enabled=self.config.USE_AMP):
                # 前向传播
                pred = self.model(mix)
                
                # 打印预测形状用于调试
                # if batch_idx == 0:
                #     print(f"  Pred: {pred.shape}")
                
                # 计算损失（使用安全版本）
                loss = self.safe_loss(pred, target)
            
            # 如果损失是无效的，跳过这个batch
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  Batch {batch_idx} 损失无效: {loss.item()}, 跳过")
                continue
            
            # 梯度累积
            loss = loss / self.config.GRADIENT_ACCUMULATION_STEPS
            
            # 反向传播
            self.scaler.scale(loss).backward()
            
            # 每GRADIENT_ACCUMULATION_STEPS步更新一次
            if (batch_idx + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                # 梯度裁剪
                if self.config.USE_GRADIENT_CLIPPING:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.GRADIENT_CLIP_VALUE
                    )
                
                # 优化器步骤
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            # 统计
            total_loss += loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # 定期清理缓存
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return avg_loss
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(val_loader, desc="🎸 验证批次")
        with torch.no_grad():
            for batch_idx, (mix, target) in enumerate(pbar):
                mix = mix.to(self.device)
                target = target.to(self.device)
                
                # 打印第一个batch的形状用于调试
                if batch_idx == 0:
                    print(f"🎵 验证Batch {batch_idx} 形状:")
                    print(f"  Mix: {mix.shape}")
                    print(f"  Target: {target.shape}")
                
                # 前向传播
                pred = self.model(mix)
                
                if batch_idx == 0:
                    print(f"  Pred: {pred.shape}")
                
                # 计算损失（使用安全版本）
                loss = self.safe_loss(pred, target)
                
                # 跳过无效损失
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️  验证Batch {batch_idx} 损失无效: {loss.item()}")
                    continue
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs):
        """训练模型"""
        print(f"🚀 开始训练吉他分离模型，共 {epochs} 个epoch")
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10  # 早停耐心值
        
        for epoch in range(epochs):
            print(f"\n🔄 Epoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_loss = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # 打印结果
            print(f"📊 训练损失: {train_loss:.4f}")
            print(f"📊 验证损失: {val_loss:.4f}")
            print(f"📊 学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(best=True)
                print(f"💾 保存最佳模型，验证损失: {val_loss:.4f}")
            else:
                patience_counter += 1
            
            # 定期保存检查点
            if (epoch + 1) % 5 == 0:
                self.save_model(best=False, epoch=epoch+1)
            
            # 早停检查
            if patience_counter >= patience:
                print(f"🛑 早停触发，验证损失连续{patience}个epoch未提升")
                break
        
        print(f"🏁 训练完成，最佳验证损失: {best_val_loss:.4f}")
        
        # 保存最终模型
        self.save_model(best=False, epoch='final')
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        return self.history
    
    def save_model(self, best=True, epoch=None):
        """保存模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if best:
            model_path = os.path.join(self.config.MODEL_DIR, "best_model.pth")
        elif epoch:
            model_path = os.path.join(
                self.config.MODEL_DIR, 
                "checkpoints", 
                f"model_epoch_{epoch}.pth"
            )
        else:
            model_path = os.path.join(self.config.MODEL_DIR, f"model_final_{timestamp}.pth")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 保存模型权重
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.config.USE_AMP else None,
            'history': self.history,
            'config': self.config.__dict__,
            'epoch': len(self.history['train_loss'])
        }, model_path)
        
        print(f"💾 模型已保存到: {model_path}")
        return model_path
    
    def load_model(self, model_path):
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"📂 模型已从 {model_path} 加载")
        return True
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if len(self.history['train_loss']) == 0:
            print("⚠️  没有训练历史数据可绘制")
            return
        
        plt.figure(figsize=(10, 4))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Guitar Separation Model Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 学习率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.history['learning_rates'], label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = os.path.join(self.config.OUTPUT_DIR, "separation_training_curves.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"🖼️ 训练曲线已保存到: {output_path}")
        plt.show()

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='吉他分离模型训练')
    parser.add_argument('--epochs', type=int, default=GuitarSeparationConfig.EPOCHS,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=GuitarSeparationConfig.BATCH_SIZE,
                       help='批大小')
    parser.add_argument('--learning-rate', type=float, default=GuitarSeparationConfig.LEARNING_RATE,
                       help='学习率')
    parser.add_argument('--no-amp', action='store_true',
                       help='禁用混合精度训练')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练')
    parser.add_argument('--test-only', action='store_true',
                       help='仅测试，不训练')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    print("=" * 60)
    print("🎸 吉他音轨分离模型训练")
    print("=" * 60)
    
    # 创建配置
    config = GuitarSeparationConfig()
    
    # 更新配置参数
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = min(args.batch_size, 2)  # 确保batch size不会太大
    config.LEARNING_RATE = args.learning_rate
    config.USE_AMP = not args.no_amp
    
    # 创建目录
    config.create_directories()
    
    # 设置设备
    device = config.get_device()
    
    try:
        # 分析数据集统计
        data_loader = SeparationDataLoader(config)
        data_loader.analyze_dataset_stats()
        
        # 创建数据加载器
        print("\n📊 准备数据集...")
        loaders = data_loader.create_data_loaders()
        
        if loaders is None:
            print("❌ 数据加载失败，退出程序")
            return
        
        train_loader, val_loader, test_loader = loaders
        
        # 检查数据集是否为空
        if len(train_loader.dataset) == 0:
            print("❌ 训练数据集为空，请检查数据文件")
            return
        
        # 创建模型
        print("\n🧩 创建吉他分离模型...")
        model = GuitarSeparationModel(config).to(device)
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📈 模型参数量: {total_params:,} (可训练: {trainable_params:,})")
        
        # 估算内存使用
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"💾 GPU内存: {total_memory:.2f} GB")
            
            # 估算模型内存
            model_memory = total_params * 4 / 1e9  # 参数 * 4字节
            # 音频数据内存估算：batch_size * 时长(秒) * 采样率 * 通道数 * 4字节
            batch_memory = config.BATCH_SIZE * config.DURATION * config.SAMPLE_RATE * config.N_CHANNELS * 4 / 1e9
            estimated_memory = model_memory + batch_memory * 3  # 模型 + 输入 + 输出 + 梯度
            
            print(f"💾 估计内存使用: {estimated_memory:.2f} GB")
            
            # 如果估计内存超过可用内存的70%，警告
            if estimated_memory > total_memory * 0.7:
                print("⚠️  警告：估计内存使用可能过高，考虑减小batch size或模型大小")
        
        # 创建训练器
        trainer = SeparationTrainer(model, device, config)
        
        # 如果指定了恢复训练，则加载模型
        if args.resume and os.path.exists(args.resume):
            print(f"♻️ 从检查点恢复训练: {args.resume}")
            trainer.load_model(args.resume)
        
        if not args.test_only:
            # 训练模型
            print("\n🚀 开始训练...")
            history = trainer.train(train_loader, val_loader, config.EPOCHS)
        else:
            print("\n⏭️  跳过训练，仅进行测试...")
        
        # 评估模型
        print("\n🎯 在测试集上评估模型...")
        test_loss = trainer.validate(test_loader)
        
        print("\n" + "=" * 60)
        print("🏁 训练完成!")
        print(f"🎯 最终测试损失: {test_loss:.4f}")
        print(f"💾 最佳模型保存在: {os.path.join(config.MODEL_DIR, 'best_model.pth')}")
        print(f"📊 训练曲线保存在: {os.path.join(config.OUTPUT_DIR, 'separation_training_curves.png')}")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 如果出现内存错误，提供建议
        if "memory" in str(e).lower() or "allocate" in str(e).lower():
            print("\n💡 内存问题建议:")
            print("1. 减小 batch size: --batch-size 1")
            print("2. 禁用混合精度训练: --no-amp")
            print("3. 减小音频时长: 修改 config.DURATION")
            print("4. 降低采样率: 修改 config.SAMPLE_RATE")
        # 如果出现形状不匹配错误，提供建议
        elif "size" in str(e).lower() and "tensor" in str(e).lower():
            print("\n💡 形状不匹配问题建议:")
            print("1. 检查模型输入输出形状")
            print("2. 确保数据集返回的音频长度一致")
            print("3. 使用更简单的模型结构")

if __name__ == "__main__":
    main()