# src/rec/dataset.py
import os
import json
import librosa
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
# 设置matplotlib使用英文字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from src.rec.config import GuitarRecConfig

class GuitarDataset(Dataset):
    """吉他二分类数据集"""
    
    def __init__(self, features, labels, negative_features=None, transform=None, augment=False, 
                 mix_prob=0.5, max_mix_ratio=0.7):
        """
        Args:
            features: 所有样本特征
            labels: 所有样本标签
            negative_features: 负样本特征（用于混合增强）
            transform: 数据转换
            augment: 是否进行数据增强
            mix_prob: 混合增强的概率
            max_mix_ratio: 最大混合比例（吉他音量占比）
        """
        self.features = features
        self.labels = labels
        self.negative_features = negative_features
        self.transform = transform
        self.augment = augment
        self.mix_prob = mix_prob
        self.max_mix_ratio = max_mix_ratio
        
        # 为快速访问准备索引
        if negative_features is not None:
            self.neg_indices = list(range(len(negative_features)))
        
        # 存储正负样本索引
        self.pos_indices = [i for i, label in enumerate(labels) if label == 1]
        self.neg_indices_orig = [i for i, label in enumerate(labels) if label == 0]
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx].copy()
        label = self.labels[idx]
        
        # 训练期间的数据增强
        if self.augment:
            # 混合增强：对于正样本，随机混合负样本；对于负样本，也混合增强
            if random.random() < self.mix_prob:
                if label == 1 and self.negative_features is not None and len(self.negative_features) > 0:
                    # 正样本混合：吉他 + 其他乐器
                    feature = self._mix_guitar_with_instrument(feature)
                elif label == 0 and self.pos_indices:
                    # 负样本混合：其他乐器 + 少量吉他（模拟吉他被掩盖的情况）
                    if random.random() < 0.3:  # 30%概率给负样本添加少量吉他
                        feature = self._mix_instrument_with_guitar(feature)
            
            # 应用其他数据增强
            feature = self._apply_augmentation(feature)
        
        # 转换为PyTorch张量
        feature = torch.FloatTensor(feature).unsqueeze(0)  # (1, 128, 130)
        label = torch.LongTensor([label])[0]
        
        return feature, label
    
    def _mix_guitar_with_instrument(self, guitar_feature):
        """混合吉他特征与其他乐器特征"""
        if self.negative_features is None or len(self.negative_features) == 0:
            return guitar_feature
        
        # 随机选择一个负样本（其他乐器）
        neg_idx = random.choice(range(len(self.negative_features)))
        instrument_feature = self.negative_features[neg_idx].copy()
        
        # 随机混合比例：吉他占主导 (0.6-0.9)，模拟真实混合
        guitar_ratio = random.uniform(0.6, self.max_mix_ratio)
        
        # 混合特征（在log-mel域近似线性混合）
        mixed = guitar_ratio * guitar_feature + (1 - guitar_ratio) * instrument_feature
        
        # 添加轻微噪声模拟真实录音
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.02, mixed.shape)
            mixed = mixed + noise
        
        # 重新标准化
        mixed = (mixed - np.mean(mixed)) / (np.std(mixed) + 1e-8)
        
        return mixed
    
    def _mix_instrument_with_guitar(self, instrument_feature):
        """混合其他乐器特征与少量吉他特征"""
        if not self.pos_indices:
            return instrument_feature
        
        # 随机选择一个正样本（吉他）
        pos_idx = random.choice(self.pos_indices)
        guitar_feature = self.features[pos_idx].copy()
        
        # 少量吉他混合 (0.1-0.3)，模拟吉他被掩盖的情况
        guitar_ratio = random.uniform(0.1, 0.3)
        
        # 混合特征
        mixed = (1 - guitar_ratio) * instrument_feature + guitar_ratio * guitar_feature
        
        # 重新标准化
        mixed = (mixed - np.mean(mixed)) / (np.std(mixed) + 1e-8)
        
        return mixed
    
    def _apply_augmentation(self, feature):
        """应用数据增强"""
        # 时间掩码
        if random.random() > 0.5:
            max_mask_width = feature.shape[1] // 4
            if max_mask_width > 0:
                mask_width = random.randint(1, max_mask_width)
                mask_start = random.randint(0, feature.shape[1] - mask_width)
                feature[:, mask_start:mask_start + mask_width] = 0
        
        # 频率掩码
        if random.random() > 0.5:
            max_mask_height = feature.shape[0] // 8
            if max_mask_height > 0:
                mask_height = random.randint(1, max_mask_height)
                mask_start = random.randint(0, feature.shape[0] - mask_height)
                feature[mask_start:mask_start + mask_height, :] = 0
        
        # 添加小噪声
        if random.random() > 0.7:
            noise = np.random.normal(0, 0.01, feature.shape)
            feature = feature + noise
        
        # 音高偏移（模拟不同调弦或变调）
        if random.random() > 0.8:
            shift = random.randint(-2, 2)
            if shift != 0:
                feature = np.roll(feature, shift, axis=0)
        
        return feature

class GuitarDataPreprocessor:
    """吉他音频数据预处理器"""
    
    def __init__(self, target_sr=GuitarRecConfig.TARGET_SAMPLE_RATE, 
                 duration=GuitarRecConfig.AUDIO_DURATION):
        self.target_sr = target_sr
        self.duration = duration
        self.json_path = os.path.join(
            GuitarRecConfig.DATA_DIR, "extract", "binary_classification", "dataset.json"
        )
    
    def extract_features(self, audio_path):
        """提取音频特征"""
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=self.target_sr)
            
            # 确保音频长度一致
            y = librosa.util.fix_length(y, size=self.target_sr * self.duration)
            
            # 提取Mel频谱图
            mel_spec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=GuitarRecConfig.N_MELS, fmax=8000, 
                n_fft=2048, hop_length=512
            )
            log_mel = librosa.power_to_db(mel_spec)
            
            # 标准化
            log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
            
            # 确保特征尺寸一致
            target_frames = 130
            if log_mel.shape[1] < target_frames:
                pad_width = target_frames - log_mel.shape[1]
                log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
            elif log_mel.shape[1] > target_frames:
                log_mel = log_mel[:, :target_frames]
            
            return log_mel
            
        except Exception as e:
            print(f"❌ 处理音频 {audio_path} 时出错: {e}")
            return None
    
    def load_data_from_json(self, max_samples_per_instrument=None, balance_classes=False, 
                           guitar_multiplier=1, max_total_samples=5000, include_mixed=True):
        """从JSON文件加载数据集，使用dataset.json的标注信息
        
        Args:
            max_samples_per_instrument: 每个乐器最大样本数
            balance_classes: 是否平衡类别（设为False，使用更多负样本）
            guitar_multiplier: 吉他样本扩增倍数
            max_total_samples: 最大总样本数
            include_mixed: 是否包含混合样本
        """
        print("📂 从dataset.json加载标注数据...")
        
        if not os.path.exists(self.json_path):
            print(f"❌ JSON文件不存在: {self.json_path}")
            print("⚠️  将使用目录扫描方式加载数据")
            return None, None
        
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ 成功加载JSON文件，包含 {len(data)} 个样本")
        except Exception as e:
            print(f"❌ 加载JSON文件失败: {e}")
            return None, None
        
        # 分离正负样本
        positive_samples = []
        negative_samples = []
        negative_by_instrument = {}
        
        for item in data:
            file_path = item['file']
            label = item['label']
            
            if not os.path.exists(file_path):
                print(f"⚠️  文件不存在，跳过: {file_path}")
                continue
            
            if label == 1:  # 吉他
                positive_samples.append((file_path, label, item.get('instrument', 'guitar')))
            else:  # 非吉他
                negative_samples.append((file_path, label, item.get('instrument', 'other')))
                
                # 按乐器类型统计
                instrument = item.get('instrument', 'unknown')
                if instrument not in negative_by_instrument:
                    negative_by_instrument[instrument] = []
                negative_by_instrument[instrument].append((file_path, label, instrument))
        
        print(f"📊 JSON原始数据统计:")
        print(f"  吉他样本数: {len(positive_samples)}")
        print(f"  非吉他样本数: {len(negative_samples)}")
        
        if negative_by_instrument:
            print(f"  非吉他样本乐器分布:")
            for instrument, samples in negative_by_instrument.items():
                print(f"    {instrument}: {len(samples)} 个样本")
        
        # 吉他样本扩增
        if guitar_multiplier > 1 and positive_samples:
            expanded_positive = []
            for _ in range(guitar_multiplier):
                expanded_positive.extend(positive_samples)
            positive_samples = expanded_positive
            print(f"🎸 吉他样本扩增 {guitar_multiplier} 倍，现在有 {len(positive_samples)} 个吉他样本")
        
        # 处理负样本：每个乐器采样
        processed_negative = []
        if negative_by_instrument:
            for instrument, samples in negative_by_instrument.items():
                if max_samples_per_instrument and len(samples) > max_samples_per_instrument:
                    # 采样指定数量的样本
                    sampled = random.sample(samples, max_samples_per_instrument)
                    processed_negative.extend(sampled)
                    print(f"  ⚖️  {instrument}: 采样 {max_samples_per_instrument} 个样本")
                else:
                    # 使用全部样本
                    processed_negative.extend(samples)
                    print(f"  📊 {instrument}: 使用全部 {len(samples)} 个样本")
        else:
            processed_negative = negative_samples
        
        # 创建混合样本（如果启用）
        mixed_samples = []
        if include_mixed and positive_samples and processed_negative:
            print(f"🔄 创建混合样本...")
            # 创建吉他+其他乐器的混合样本
            num_mixed = min(500, len(positive_samples) // 2)  # 最多500个混合样本
            for _ in range(num_mixed):
                # 随机选择吉他和非吉他样本
                guitar_sample = random.choice(positive_samples)
                non_guitar_sample = random.choice(processed_negative)
                
                # 创建混合样本记录（实际混合在训练时动态进行）
                mixed_samples.append((
                    guitar_sample[0],  # 吉他音频路径
                    non_guitar_sample[0],  # 非吉他音频路径
                    1,  # 标签为吉他（因为包含吉他）
                    f"mixed_{guitar_sample[2]}_{non_guitar_sample[2]}"
                ))
            print(f"  🎛️  创建了 {len(mixed_samples)} 个混合样本")
        
        # 平衡类别（如果不平衡）
        if balance_classes:
            # 确保正负样本数量相近
            min_samples = min(len(positive_samples), len(processed_negative))
            if len(positive_samples) > min_samples:
                positive_samples = random.sample(positive_samples, min_samples)
                print(f"  ⚖️  吉他样本采样到 {min_samples} 个")
            if len(processed_negative) > min_samples:
                processed_negative = random.sample(processed_negative, min_samples)
                print(f"  ⚖️  非吉他样本采样到 {min_samples} 个")
        else:
            # 如果不平衡，使用更多负样本
            print(f"  📈 使用不平衡数据集: {len(positive_samples)} 吉他 vs {len(processed_negative)} 非吉他")
        
        # 合并所有样本
        all_samples = positive_samples + processed_negative
        
        # 添加混合样本
        if mixed_samples:
            # 只添加路径和标签
            all_samples.extend([(mix[0], mix[2], mix[3]) for mix in mixed_samples])
            print(f"  🎚️  加入 {len(mixed_samples)} 个混合样本，总数: {len(all_samples)}")
        
        # 限制总样本数
        if max_total_samples and len(all_samples) > max_total_samples:
            all_samples = random.sample(all_samples, max_total_samples)
            print(f"  📊 总样本数限制到 {max_total_samples}")
        
        random.shuffle(all_samples)
        
        # 提取文件路径和标签
        file_paths = [item[0] for item in all_samples]
        labels = [item[1] for item in all_samples]
        
        print(f"📊 最终数据集统计:")
        print(f"  正样本(吉他): {sum(labels)}")
        print(f"  负样本(非吉他): {len(labels) - sum(labels)}")
        print(f"  总样本数: {len(labels)}")
        print(f"  吉他占比: {sum(labels)/len(labels)*100:.1f}%")
        
        return file_paths, labels
    
    def prepare_dataset(self, max_negative_samples=None, use_json=True):
        """准备数据集"""
        print("🔄 准备吉他二分类数据集...")
        
        # 优先使用JSON数据
        if use_json:
            # 使用更多负样本，不平衡训练
            file_paths, labels = self.load_data_from_json(
                max_samples_per_instrument=1000,  # 每个乐器最多1000个样本
                balance_classes=False,  # 不进行平衡
                guitar_multiplier=3,    # 吉他样本扩增3倍
                max_total_samples=5000,  # 最多5000个样本
                include_mixed=True      # 包含混合样本
            )
            
            if file_paths and labels:
                # 提取特征
                features = []
                filtered_labels = []
                failed_files = []
                
                print(f"🎵 提取 {len(file_paths)} 个样本的特征...")
                for i, (audio_path, label) in tqdm(enumerate(zip(file_paths, labels)), 
                                                   desc="处理音频", total=len(file_paths)):
                    feature = self.extract_features(audio_path)
                    if feature is not None:
                        features.append(feature)
                        filtered_labels.append(label)
                    else:
                        failed_files.append(audio_path)
                
                X = np.array(features)
                y = np.array(filtered_labels)
                
                print(f"📊 数据集统计:")
                print(f"  总样本数: {len(X)}")
                print(f"  吉他样本数: {np.sum(y == 1)}")
                print(f"  非吉他样本数: {np.sum(y == 0)}")
                print(f"  失败样本数: {len(failed_files)}")
                
                if len(failed_files) > 0 and len(failed_files) <= 10:
                    print(f"  失败文件列表: {failed_files}")
                
                return X, y
        
        # 如果JSON加载失败，使用目录扫描方式
        print("⚠️  使用目录扫描方式准备数据")
        
        # 加载所有样本
        positive_samples = self.load_guitar_samples()
        negative_samples = self.load_non_guitar_samples(max_samples=max_negative_samples)
        
        # 检查样本数量
        if len(positive_samples) == 0:
            print("❌ 错误：没有找到吉他正样本，请检查数据集路径")
            return None, None
        
        if len(negative_samples) == 0:
            print("❌ 错误：没有找到非吉他负样本，请检查数据集路径")
            return None, None
        
        # 合并并打乱
        all_samples = positive_samples + negative_samples
        random.shuffle(all_samples)
        
        # 提取特征
        features = []
        labels = []
        failed_files = []
        
        print(f"🎵 提取 {len(all_samples)} 个样本的特征...")
        for audio_path, label in tqdm(all_samples, desc="处理音频"):
            feature = self.extract_features(audio_path)
            if feature is not None:
                features.append(feature)
                labels.append(label)
            else:
                failed_files.append(audio_path)
        
        X = np.array(features)
        y = np.array(labels)
        
        print(f"📊 数据集统计:")
        print(f"  总样本数: {len(X)}")
        print(f"  吉他样本数: {np.sum(y == 1)}")
        print(f"  非吉他样本数: {np.sum(y == 0)}")
        print(f"  失败样本数: {len(failed_files)}")
        
        if len(failed_files) > 0 and len(failed_files) <= 10:
            print(f"  失败文件列表: {failed_files}")
        
        return X, y
    
    def create_data_loaders(self, batch_size=GuitarRecConfig.BATCH_SIZE, augment=True,
                           mix_prob=0.5, max_mix_ratio=0.7):
        """创建数据加载器"""
        # 准备数据集
        X, y = self.prepare_dataset(max_negative_samples=2000, use_json=True)
        
        if X is None or y is None:
            print("❌ 数据集准备失败，无法创建数据加载器")
            return None, None, None, None
        
        # 划分训练集、验证集、测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=GuitarRecConfig.TEST_SPLIT, 
            random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=GuitarRecConfig.VALIDATION_SPLIT, 
            random_state=42, stratify=y_temp
        )
        
        # 提取负样本特征（用于训练集混合增强）
        train_neg_features = None
        if augment:
            train_neg_indices = np.where(y_train == 0)[0]
            if len(train_neg_indices) > 0:
                train_neg_features = X_train[train_neg_indices]
                print(f"🎚️  训练集负样本特征数: {len(train_neg_features)} (用于混合增强)")
        
        # 创建数据集
        train_dataset = GuitarDataset(
            X_train, y_train, 
            negative_features=train_neg_features,
            augment=augment,
            mix_prob=mix_prob,
            max_mix_ratio=max_mix_ratio
        )
        val_dataset = GuitarDataset(X_val, y_val, augment=False)
        test_dataset = GuitarDataset(X_test, y_test, augment=False)
        
        # 计算类别权重（解决不平衡问题）
        if GuitarRecConfig.USE_CLASS_WEIGHTS:
            from sklearn.utils.class_weight import compute_class_weight
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            print(f"⚖️  类别权重: {class_weights}")
        else:
            class_weights = None
        
        # 创建数据加载器 - 训练集使用drop_last=True避免batch size=1的问题
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True  # 避免最后一个batch size=1
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"📊 数据加载器统计:")
        print(f"  训练集: {len(train_dataset)} 个样本")
        print(f"  验证集: {len(val_dataset)} 个样本")
        print(f"  测试集: {len(test_dataset)} 个样本")
        print(f"  增强配置: 混合概率={mix_prob}, 最大混合比={max_mix_ratio}")
        
        return train_loader, val_loader, test_loader, class_weights
    
    def analyze_dataset_distribution(self):
        """分析数据集分布情况"""
        print("🔍 分析数据集分布...")
        
        file_paths, labels = self.load_data_from_json(
            max_samples_per_instrument=None,
            balance_classes=False,
            guitar_multiplier=1,
            max_total_samples=None
        )
        
        if not file_paths:
            print("❌ 无法加载数据集进行分析")
            return
        
        # 加载JSON文件获取乐器信息
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 统计乐器分布
            instrument_counts = {}
            for item in data:
                instrument = item.get('instrument', 'unknown')
                if instrument not in instrument_counts:
                    instrument_counts[instrument] = 0
                instrument_counts[instrument] += 1
            
            print(f"\n📊 乐器分布统计:")
            for instrument, count in instrument_counts.items():
                print(f"  {instrument}: {count} 个样本")
        
        except Exception as e:
            print(f"⚠️  无法分析乐器分布: {e}")