import os
import random
import json
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
from .config import Config

class SepDataset(Dataset):
    """从extract/separation中加载混合音频和吉他音频"""
    def __init__(self, config: Config, split='train'):
        super().__init__()
        self.config = config
        self.sample_rate = config.sample_rate
        self.n_samples = config.n_samples

        # 加载数据集索引文件（假设已由extract过程生成）
        dataset_json = os.path.join(config.data_root, 'separation/dataset.json')
        with open(dataset_json, 'r') as f:
            all_items = json.load(f)   # 列表，每个元素为 {"mix": path, "guitar": path}

        # 简单划分训练/验证集
        random.seed(config.seed)
        random.shuffle(all_items)
        val_size = int(len(all_items) * config.val_ratio)
        if split == 'train':
            self.items = all_items[val_size:]
        else:
            self.items = all_items[:val_size]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        mix_path = item['mix']
        guitar_path = item['guitar']

        # 加载音频（如果长度不足则循环填充）
        mix, sr = torchaudio.load(mix_path)
        guitar, sr_ = torchaudio.load(guitar_path)
        assert sr == self.sample_rate and sr_ == self.sample_rate

        # 随机裁剪到固定长度
        if mix.size(1) > self.n_samples:
            start = random.randint(0, mix.size(1) - self.n_samples)
            mix = mix[:, start:start+self.n_samples]
            guitar = guitar[:, start:start+self.n_samples]
        else:
            # 重复填充至足够长度
            repeat = (self.n_samples + mix.size(1) - 1) // mix.size(1)
            mix = mix.repeat(1, repeat)[:, :self.n_samples]
            guitar = guitar.repeat(1, repeat)[:, :self.n_samples]

        # 归一化（可选，Demucs通常不做全局归一化）
        # 返回形状 (1, T) 单声道，若多声道则取平均
        if mix.size(0) > 1:
            mix = mix.mean(dim=0, keepdim=True)
        if guitar.size(0) > 1:
            guitar = guitar.mean(dim=0, keepdim=True)

        return mix, guitar