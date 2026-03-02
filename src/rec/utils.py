# src/rec/utils.py
"""
吉他二分模型工具函数
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def check_dataset_paths():
    """检查数据集路径是否存在"""
    from src.rec.config import GuitarRecConfig
    
    print("🔍 检查数据集路径...")
    
    # 检查正样本路径
    positive_dir = GuitarRecConfig.POSITIVE_SAMPLES_DIR
    if os.path.exists(positive_dir):
        print(f"✅ 吉他正样本路径存在: {positive_dir}")
        
        # 统计文件数量
        wav_files = []
        for root, dirs, files in os.walk(positive_dir):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        
        print(f"📊 找到 {len(wav_files)} 个.wav文件")
        
        # 显示一些文件示例
        if wav_files:
            print("📁 文件示例:")
            for file in wav_files[:3]:
                print(f"  - {os.path.relpath(file, positive_dir)}")
    else:
        print(f"❌ 吉他正样本路径不存在: {positive_dir}")
    
    # 检查负样本路径
    negative_dir = GuitarRecConfig.NEGATIVE_SAMPLES_DIR
    if os.path.exists(negative_dir):
        print(f"\n✅ 非吉他负样本路径存在: {negative_dir}")
        
        # 统计文件数量
        wav_files = []
        for root, dirs, files in os.walk(negative_dir):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        
        print(f"📊 找到 {len(wav_files)} 个.wav文件")
    else:
        print(f"❌ 非吉他负样本路径不存在: {negative_dir}")
    
    return os.path.exists(positive_dir) and os.path.exists(negative_dir)

def plot_prediction_distribution(predictions, labels, save_path=None):
    """绘制预测结果分布图"""
    # 将预测结果分为正确和错误
    correct_idx = np.where(predictions == labels)[0]
    wrong_idx = np.where(predictions != labels)[0]
    
    # 统计各类别的准确率
    unique_labels = np.unique(labels)
    class_names = ['非吉他', '吉他']
    
    accuracy_by_class = []
    for label in unique_labels:
        class_mask = labels == label
        class_correct = np.sum(predictions[class_mask] == labels[class_mask])
        class_total = np.sum(class_mask)
        accuracy = class_correct / class_total if class_total > 0 else 0
        accuracy_by_class.append(accuracy)
    
    # 绘制图表
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 准确率柱状图
    axes[0].bar(class_names, accuracy_by_class, color=['skyblue', 'lightgreen'])
    axes[0].set_title('各类别准确率')
    axes[0].set_ylabel('准确率')
    axes[0].set_ylim(0, 1.0)
    
    # 添加数值标签
    for i, acc in enumerate(accuracy_by_class):
        axes[0].text(i, acc + 0.02, f'{acc:.3f}', ha='center')
    
    # 预测分布散点图
    axes[1].scatter(range(len(correct_idx)), predictions[correct_idx], 
                    alpha=0.5, label='正确预测', color='green')
    axes[1].scatter(range(len(wrong_idx)), predictions[wrong_idx], 
                    alpha=0.5, label='错误预测', color='red')
    axes[1].set_title('预测结果分布')
    axes[1].set_xlabel('样本索引')
    axes[1].set_ylabel('预测类别')
    axes[1].legend()
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(class_names)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 预测分布图已保存: {save_path}")
    
    plt.show()
    
    return accuracy_by_class