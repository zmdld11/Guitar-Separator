# src/sep/dataset.py
import os
import json
import random
import numpy as np
import torch
import librosa
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.sep.config import GuitarSeparationConfig

class SeparationDataset(Dataset):
    """吉他分离数据集"""
    
    def __init__(self, samples, config, is_train=True):
        """
        Args:
            samples: 样本列表，每个元素为(mix_path, guitar_path)
            config: 配置对象
            is_train: 是否为训练集
        """
        self.samples = samples
        self.config = config
        self.is_train = is_train
        self.sr = config.SAMPLE_RATE
        self.duration_samples = int(config.DURATION * self.sr)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        mix_path, guitar_path = self.samples[idx]
        
        # 加载音频（使用优化的加载方法）
        try:
            mix_audio = self.load_audio_safe(mix_path)
            guitar_audio = self.load_audio_safe(guitar_path)
        except Exception as e:
            print(f"❌ 加载音频失败: {e}")
            # 返回静音数据，使用正确的通道数
            mix_audio = np.zeros((self.config.N_CHANNELS, self.duration_samples), dtype=np.float32)
            guitar_audio = np.zeros((self.config.N_CHANNELS, self.duration_samples), dtype=np.float32)
            return torch.FloatTensor(mix_audio), torch.FloatTensor(guitar_audio)
        
        # 确保音频长度合适
        mix_audio = self.ensure_audio_length(mix_audio, self.duration_samples)
        guitar_audio = self.ensure_audio_length(guitar_audio, self.duration_samples)
        
        # 数据增强（仅训练集）
        if self.is_train and self.config.USE_AUGMENTATION:
            mix_audio, guitar_audio = self.augment_audio(mix_audio, guitar_audio)
        
        # 转换为PyTorch张量
        mix_tensor = torch.FloatTensor(mix_audio)
        guitar_tensor = torch.FloatTensor(guitar_audio)
        
        return mix_tensor, guitar_tensor
    
    def load_audio_safe(self, path):
        """安全地加载音频文件，避免内存问题"""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"音频文件不存在: {path}")
            
            # 使用librosa加载，限制最大时长
            max_duration = self.config.DURATION * 10  # 最多加载10倍的训练时长
            audio, sr = librosa.load(
                path, 
                sr=self.sr, 
                mono=False,  # 保持立体声
                duration=max_duration  # 限制时长，避免加载太长的音频
            )
            
            # 确保正确的形状 (channels, samples)
            if audio.ndim == 1:
                # 单声道转立体声（复制通道）
                if self.config.N_CHANNELS == 2:
                    audio = np.stack([audio, audio])
                else:
                    audio = audio.reshape(1, -1)
            elif audio.ndim == 2:
                # 确保形状是 (channels, samples)
                if audio.shape[0] > audio.shape[1]:
                    audio = audio.T
                
                # 如果需要单声道但音频是立体声，取平均值
                if self.config.N_CHANNELS == 1 and audio.shape[0] == 2:
                    audio = audio.mean(axis=0, keepdims=True)
                # 如果需要立体声但音频是单声道，复制通道
                elif self.config.N_CHANNELS == 2 and audio.shape[0] == 1:
                    audio = np.tile(audio, (2, 1))
            
            # 确保数据类型为float32，减少内存使用
            audio = audio.astype(np.float32)
            
            return audio
            
        except Exception as e:
            raise Exception(f"加载音频 {path} 失败: {e}")
    
    def ensure_audio_length(self, audio, target_length):
        """确保音频长度合适，避免内存溢出"""
        if audio.shape[1] < target_length:
            # 如果音频太短，重复填充
            repeats = int(np.ceil(target_length / audio.shape[1]))
            
            # 使用小块重复，避免一次性创建大数组
            audio_len = audio.shape[1]
            result = []
            
            for _ in range(repeats):
                result.append(audio)
            
            # 合并并裁剪到目标长度
            if len(result) > 0:
                audio = np.concatenate(result, axis=1)
            
            audio = audio[:, :target_length]
        elif audio.shape[1] > target_length:
            # 如果音频太长，随机裁剪一段
            start = np.random.randint(0, audio.shape[1] - target_length)
            audio = audio[:, start:start + target_length]
        
        return audio
    
    def augment_audio(self, mix_audio, guitar_audio):
        """数据增强（优化版本）"""
        if random.random() < self.config.AUGMENT_PROB:
            # 随机增益
            if random.random() < 0.3:
                gain = random.uniform(*self.config.GAIN_RANGE)
                mix_audio = mix_audio * (10 ** (gain / 20))
            
            # 时间掩码
            if random.random() < self.config.TIME_MASK_PROB:
                mask_len = min(int(self.sr * 0.2), mix_audio.shape[1] // 10)  # 限制掩码长度
                mask_len = max(100, mask_len)  # 确保最小长度
                
                if mask_len < mix_audio.shape[1]:
                    mask_start = random.randint(0, mix_audio.shape[1] - mask_len)
                    mask_factor = random.uniform(0.05, 0.2)  # 随机掩码因子
                    mix_audio[:, mask_start:mask_start + mask_len] *= mask_factor
                    guitar_audio[:, mask_start:mask_start + mask_len] *= mask_factor
            
            # 添加轻微噪声
            if random.random() < 0.2:
                noise_level = random.uniform(0.001, 0.005)
                noise = np.random.normal(0, noise_level, mix_audio.shape).astype(np.float32)
                mix_audio = mix_audio + noise
                guitar_audio = guitar_audio + noise
        
        return mix_audio, guitar_audio

class SeparationDataLoader:
    """吉他分离数据加载器"""
    
    def __init__(self, config):
        self.config = config
        self.dataset_json = config.DATASET_JSON
        self.stats_json = config.DATASET_STATS
        
    def load_dataset_from_json(self):
        """从JSON文件加载数据集"""
        if not os.path.exists(self.dataset_json):
            print(f"❌ 数据集JSON文件不存在: {self.dataset_json}")
            return None
        
        try:
            with open(self.dataset_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"✅ 成功加载分离数据集JSON，包含 {len(data)} 个样本")
            
            # 检查JSON结构
            if data and len(data) > 0:
                sample_keys = list(data[0].keys())
                print(f"🔍 样本键名: {sample_keys}")
            
            # 提取样本对
            samples = []
            invalid_count = 0
            
            for i, item in enumerate(data):
                # 尝试不同的键名组合
                mix_path = None
                guitar_path = None
                
                # 尝试可能的键名
                if 'mix_path' in item and 'guitar_path' in item:
                    mix_path = item['mix_path']
                    guitar_path = item['guitar_path']
                elif 'mix' in item and 'guitar' in item:
                    mix_path = item['mix']
                    guitar_path = item['guitar']
                elif 'file' in item and 'guitar_file' in item:
                    mix_path = item['file']
                    guitar_path = item['guitar_file']
                elif isinstance(item, dict) and len(item) >= 2:
                    # 假设前两个键是mix和guitar
                    keys = list(item.keys())
                    mix_path = item.get(keys[0])
                    guitar_path = item.get(keys[1])
                
                # 检查路径是否存在
                if mix_path and guitar_path:
                    # 确保路径是字符串
                    if not isinstance(mix_path, str):
                        mix_path = str(mix_path)
                    if not isinstance(guitar_path, str):
                        guitar_path = str(guitar_path)
                    
                    # 检查文件是否存在
                    if os.path.exists(mix_path) and os.path.exists(guitar_path):
                        samples.append((mix_path, guitar_path))
                    else:
                        invalid_count += 1
                        if invalid_count <= 10:  # 只打印前10个无效样本
                            print(f"⚠️  跳过无效样本 [{i}]: mix={mix_path}, guitar={guitar_path}")
                else:
                    invalid_count += 1
                    if invalid_count <= 10:
                        print(f"⚠️  跳过格式错误样本 [{i}]: {item}")
            
            print(f"📊 有效样本数: {len(samples)} / {len(data)}")
            print(f"📊 无效样本数: {invalid_count}")
            
            if len(samples) == 0:
                print("❌ 没有找到有效样本，将尝试从目录结构加载")
                return None
            
            return samples
            
        except Exception as e:
            print(f"❌ 加载数据集失败: {e}")
            return None
    
    def load_dataset_from_directory(self):
        """从目录结构加载数据集（备选方案）"""
        print("🔍 尝试从目录结构加载数据集...")
        
        # 查找可能的目录结构
        possible_mix_dirs = [
            os.path.join(self.config.EXTRACT_DIR, "moisesdb", "mix_segments"),
            os.path.join(self.config.EXTRACT_DIR, "medleydb", "mix_segments"),
        ]
        
        possible_guitar_dirs = [
            os.path.join(self.config.EXTRACT_DIR, "moisesdb", "guitar_segments"),
            os.path.join(self.config.EXTRACT_DIR, "medleydb", "guitar_segments"),
        ]
        
        samples = []
        
        # 遍历目录查找匹配的文件
        for mix_dir in possible_mix_dirs:
            if os.path.exists(mix_dir):
                print(f"📂 搜索混合音频目录: {mix_dir}")
                
                # 获取所有WAV文件
                mix_files = []
                for root, dirs, files in os.walk(mix_dir):
                    for file in files:
                        if file.lower().endswith('.wav'):
                            mix_files.append(os.path.join(root, file))
                
                print(f"  找到 {len(mix_files)} 个混合音频文件")
                
                # 为每个混合文件查找对应的吉他文件
                for mix_file in mix_files[:100]:  # 限制数量，避免太多
                    # 提取相对路径
                    rel_path = os.path.relpath(mix_file, mix_dir)
                    
                    # 尝试在不同的目录中查找对应的吉他文件
                    guitar_file = None
                    
                    # 在吉他目录中查找相同相对路径的文件
                    for guitar_dir in possible_guitar_dirs:
                        if os.path.exists(guitar_dir):
                            possible_guitar_path = os.path.join(guitar_dir, rel_path)
                            if os.path.exists(possible_guitar_path):
                                guitar_file = possible_guitar_path
                                break
                    
                    if guitar_file and os.path.exists(guitar_file):
                        samples.append((mix_file, guitar_file))
                    else:
                        if len(samples) < 10:  # 只在前几个样本中显示警告
                            print(f"⚠️  未找到匹配的吉他文件: {mix_file}")
        
        print(f"📊 从目录结构加载了 {len(samples)} 个样本")
        
        # 如果样本太少，尝试简化匹配
        if len(samples) < 10:
            print("🔍 尝试简化匹配...")
            self.load_simple_matches(samples, possible_mix_dirs, possible_guitar_dirs)
        
        return samples
    
    def load_simple_matches(self, samples, possible_mix_dirs, possible_guitar_dirs):
        """简化匹配：只匹配文件名"""
        for mix_dir in possible_mix_dirs:
            if os.path.exists(mix_dir):
                # 收集所有混合文件
                mix_files = {}
                for root, dirs, files in os.walk(mix_dir):
                    for file in files:
                        if file.lower().endswith('.wav'):
                            base_name = os.path.splitext(file)[0]
                            mix_files[base_name] = os.path.join(root, file)
                
                # 在吉他目录中查找匹配
                for guitar_dir in possible_guitar_dirs:
                    if os.path.exists(guitar_dir):
                        for root, dirs, files in os.walk(guitar_dir):
                            for file in files:
                                if file.lower().endswith('.wav'):
                                    base_name = os.path.splitext(file)[0]
                                    if base_name in mix_files:
                                        mix_file = mix_files[base_name]
                                        guitar_file = os.path.join(root, file)
                                        samples.append((mix_file, guitar_file))
                                        if len(samples) % 100 == 0:
                                            print(f"  已匹配 {len(samples)} 个样本")
    
    def create_data_loaders(self):
        """创建数据加载器"""
        print("🔄 准备吉他分离数据集...")
        
        # 加载样本
        samples = self.load_dataset_from_json()
        if not samples or len(samples) == 0:
            print("❌ JSON数据集加载失败或为空")
            
            # 尝试从其他位置加载
            print("🔍 尝试从目录结构加载数据集...")
            samples = self.load_dataset_from_directory()
            
            if not samples or len(samples) == 0:
                print("❌ 所有数据加载方法都失败")
                print("💡 请确保已运行 extract_active_segments.py 提取数据集")
                return None, None, None
        
        print(f"📊 成功加载 {len(samples)} 个样本")
        
        # 显示前几个样本路径
        for i in range(min(3, len(samples))):
            mix_path, guitar_path = samples[i]
            print(f"  样本 {i+1}: {os.path.basename(mix_path)} -> {os.path.basename(guitar_path)}")
        
        # 随机打乱
        random.shuffle(samples)
        
        # 如果样本太多，限制数量以避免内存问题
        max_samples = 1000  # 限制最大样本数
        if len(samples) > max_samples:
            print(f"📊 样本数过多 ({len(samples)})，限制到 {max_samples} 个")
            samples = random.sample(samples, max_samples)
        
        # 划分数据集
        total = len(samples)
        test_size = int(total * self.config.TEST_SPLIT)
        val_size = int(total * self.config.VALIDATION_SPLIT)
        
        test_samples = samples[:test_size]
        val_samples = samples[test_size:test_size + val_size]
        train_samples = samples[test_size + val_size:]
        
        print(f"📊 数据集划分:")
        print(f"  训练集: {len(train_samples)} 个样本")
        print(f"  验证集: {len(val_samples)} 个样本")
        print(f"  测试集: {len(test_samples)} 个样本")
        
        # 创建数据集
        train_dataset = SeparationDataset(train_samples, self.config, is_train=True)
        val_dataset = SeparationDataset(val_samples, self.config, is_train=False)
        test_dataset = SeparationDataset(test_samples, self.config, is_train=False)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(self.config.BATCH_SIZE, 2),  # 使用更小的batch size
            shuffle=True,
            num_workers=0,  # 设为0避免多进程问题
            pin_memory=False,  # 关闭pin_memory以节省内存
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(self.config.BATCH_SIZE, 2),
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=min(self.config.BATCH_SIZE, 2),
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        return train_loader, val_loader, test_loader
    
    def analyze_dataset_stats(self):
        """分析数据集统计信息"""
        print(f"📂 统计文件路径: {self.stats_json}")
        
        if os.path.exists(self.stats_json):
            try:
                with open(self.stats_json, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                
                print(f"🔍 统计文件内容类型: {type(stats)}")
                
                print("\n📊 数据集统计信息:")
                print(f"  总片段数: {stats.get('total_segments', 'N/A')}")
                
                # 处理总时长
                total_duration = stats.get('total_duration_hours')
                if isinstance(total_duration, (int, float)):
                    print(f"  总时长: {total_duration:.2f} 小时")
                elif total_duration is not None:
                    print(f"  总时长: {total_duration}")
                else:
                    print(f"  总时长: N/A")
                
                # 显示数据分布
                if 'dataset_distribution' in stats:
                    print("  数据分布:")
                    for source, dist in stats['dataset_distribution'].items():
                        count = dist.get('count', 0)
                        duration = dist.get('total_duration', 0)
                        if isinstance(duration, (int, float)):
                            duration_str = f"{duration:.1f} 秒"
                        else:
                            duration_str = str(duration)
                        print(f"    {source}: {count} 个片段, {duration_str}")
            except Exception as e:
                print(f"⚠️  解析统计文件失败: {e}")
        else:
            print(f"⚠️  数据集统计文件不存在: {self.stats_json}")