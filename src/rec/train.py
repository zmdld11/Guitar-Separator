import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
# 设置matplotlib使用英文字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.rec.config import GuitarRecConfig
    from src.rec.dataset import GuitarDataPreprocessor
    from src.rec.model import GuitarClassifier, SimplifiedGuitarClassifier, AdvancedGuitarClassifier
except ImportError as e:
    print(f"导入错误: {e}")
    sys.exit(1)

class GuitarModelTrainer:
    """吉他模型训练器"""
    
    def __init__(self, model, device, model_type='guitar'):
        self.model = model
        self.device = device
        self.model_type = model_type
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=GuitarRecConfig.LEARNING_RATE,
            weight_decay=GuitarRecConfig.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # 训练历史记录
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rates': []
        }
        
        # 创建输出目录
        GuitarRecConfig.create_directories()
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="🎸 训练批次", leave=False)
        for batch_idx, (data, labels) in enumerate(pbar):
            data, labels = data.to(self.device), labels.to(self.device)
            
            # 前向传播
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            current_acc = correct / total if total > 0 else 0
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.4f}'})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(val_loader, desc="🎸 验证批次", leave=False)
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(pbar):
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                current_acc = correct / total if total > 0 else 0
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.4f}'})
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=GuitarRecConfig.EPOCHS):
        """训练模型"""
        print(f"🚀 开始训练吉他二分类模型，共 {epochs} 个epoch")
        
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\n🔄 Epoch {epoch+1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step(val_acc)
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # 打印结果
            print(f"📊 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"📊 验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            print(f"📊 学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model(best=True)
                print(f"💾 保存最佳模型，验证准确率: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # 早停检查
            if patience_counter >= GuitarRecConfig.EARLY_STOPPING_PATIENCE:
                print(f"🛑 早停触发，验证准确率连续{GuitarRecConfig.EARLY_STOPPING_PATIENCE}个epoch未提升")
                break
        
        print(f"🏁 训练完成，最佳验证准确率: {best_val_acc:.4f}")
        
        # 保存最终模型
        self.save_model(best=False)
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        return self.history
    
    def evaluate(self, test_loader):
        """评估模型"""
        print("🎯 评估模型性能...")
        test_loss, test_acc = self.validate(test_loader)
        
        # 计算更多指标
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # 计算二分类指标
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
        
        print(f"🎯 测试准确率: {test_acc:.4f}")
        print(f"🎯 精确率: {precision:.4f}")
        print(f"🎯 召回率: {recall:.4f}")
        print(f"🎯 F1分数: {f1:.4f}")
        
        # 绘制混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        self.plot_confusion_matrix(cm)
        
        return test_loss, test_acc
    
    def save_model(self, best=True):
        """保存模型"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if best:
            model_path = os.path.join(GuitarRecConfig.MODEL_DIR, "best_model.pth")
        else:
            model_path = os.path.join(GuitarRecConfig.MODEL_DIR, f"model_final_{timestamp}.pth")
        
        # 保存模型权重
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'epoch': len(self.history['train_loss']),
            'config': {
                'model_type': self.model_type,
                'input_shape': self.model.input_shape
            }
        }, model_path)
        
        print(f"💾 模型已保存到: {model_path}")
        return model_path
    
    def load_model(self, model_path):
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"📂 模型已从 {model_path} 加载")
        return True
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 4))
        
        # 准确率曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_acc'], label='Training Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Guitar Recognition Model Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 损失曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Guitar Recognition Model Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        output_path = os.path.join(GuitarRecConfig.OUTPUT_DIR, "guitar_training_curves.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"🖼️ 训练曲线已保存到: {output_path}")
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        """绘制混淆矩阵"""
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Non-guitar', 'Guitar'],
                   yticklabels=['Non-guitar', 'Guitar'])
        plt.title('Guitar Recognition Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # 保存图像
        output_path = os.path.join(GuitarRecConfig.OUTPUT_DIR, "guitar_confusion_matrix.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"🖼️ 混淆矩阵已保存到: {output_path}")
        plt.show()

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='吉他识别二分模型训练')
    parser.add_argument('--model-type', type=str, default='simplified', 
                       choices=['standard', 'simplified', 'advanced'],
                       help='模型类型: standard(标准), simplified(简化), advanced(高级)')
    parser.add_argument('--epochs', type=int, default=GuitarRecConfig.EPOCHS,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=GuitarRecConfig.BATCH_SIZE,
                       help='批大小')
    parser.add_argument('--no-augment', action='store_true',
                       help='禁用数据增强')
    parser.add_argument('--mix-prob', type=float, default=0.5,
                       help='混合增强的概率')
    parser.add_argument('--max-mix-ratio', type=float, default=0.7,
                       help='最大吉他混合比例')
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练')
    parser.add_argument('--test-only', action='store_true',
                       help='仅测试，不训练')
    
    return parser.parse_args()

def setup_device():
    """设置训练设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🎮 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("🖥️  使用CPU")
    
    return device

def main():
    """主函数"""
    args = parse_arguments()
    
    print("=" * 60)
    print("🎸 吉他识别二分模型训练")
    print("=" * 60)
    
    # 创建目录
    GuitarRecConfig.create_directories()
    
    # 设置设备
    device = setup_device()
    
    # 创建数据预处理器
    preprocessor = GuitarDataPreprocessor()
    
    # 分析数据集分布
    print("\n🔍 分析数据集分布...")
    preprocessor.analyze_dataset_distribution()
    
    # 创建数据加载器
    print("\n📊 准备数据集...")
    data_loaders = preprocessor.create_data_loaders(
        batch_size=args.batch_size,
        augment=not args.no_augment,
        mix_prob=args.mix_prob,
        max_mix_ratio=args.max_mix_ratio
    )
    
    if data_loaders is None:
        print("❌ 数据加载失败，退出程序")
        return
    
    train_loader, val_loader, test_loader, class_weights = data_loaders
    
    # 创建模型
    input_shape = GuitarRecConfig.INPUT_SHAPE
    
    if args.model_type == 'standard':
        model = GuitarClassifier(input_shape)
        print("🧩 使用标准吉他分类模型")
    elif args.model_type == 'advanced':
        model = AdvancedGuitarClassifier(input_shape)
        print("🧩 使用高级吉他分类模型（专门处理节奏吉他和扫弦）")
    else:
        model = SimplifiedGuitarClassifier(input_shape)
        print("🧩 使用简化吉他分类模型")
    
    model = model.to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 模型参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    # 创建训练器
    trainer = GuitarModelTrainer(model, device, model_type=args.model_type)
    
    # 如果指定了类别权重，修改损失函数
    if class_weights is not None and GuitarRecConfig.USE_CLASS_WEIGHTS:
        class_weights_tensor = torch.FloatTensor(class_weights).to(device)
        trainer.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        print(f"⚖️  使用加权交叉熵损失，权重: {class_weights}")
    
    # 如果指定了恢复训练，则加载模型
    if args.resume and os.path.exists(args.resume):
        print(f"♻️ 从检查点恢复训练: {args.resume}")
        trainer.load_model(args.resume)
    
    if not args.test_only:
        # 训练模型
        print("\n🚀 开始训练...")
        history = trainer.train(train_loader, val_loader, epochs=args.epochs)
    else:
        print("\n⏭️  跳过训练，仅进行测试...")
    
    # 评估模型
    print("\n🎯 在测试集上评估模型...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    
    print("\n" + "=" * 60)
    print("🏁 训练完成!")
    print(f"🎯 最终测试准确率: {test_acc:.4f}")
    print(f"💾 最佳模型保存在: {os.path.join(GuitarRecConfig.MODEL_DIR, 'best_model.pth')}")
    print(f"📊 训练曲线保存在: {os.path.join(GuitarRecConfig.OUTPUT_DIR, 'guitar_training_curves.png')}")
    print("=" * 60)

if __name__ == "__main__":
    main()