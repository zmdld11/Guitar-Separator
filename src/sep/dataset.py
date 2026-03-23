import os
import random
import json
import torch
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset
from .config import Config

class SepDataset(Dataset):
    """从extract/separation中加载混合音频和吉他音频"""
    def __init__(self, config: Config, split='train'):
        super().__init__()
        self.config = config
        self.sample_rate = config.sample_rate
        self.n_samples = config.n_samples

        # 定义当前的本地数据根目录
        self.local_data_root = os.path.abspath(config.data_root)

        # 加载数据集索引文件
        dataset_json = os.path.join(config.data_root, 'separation/dataset.json')
        with open(dataset_json, 'r') as f:
            all_items = json.load(f)   # 列表，每个元素为 {"input": path, "target": path}

        # 将路径转换为当前系统路径
        converted_items = []
        for item in all_items:
            converted_item = {
                'input': self._convert_path(item['input']),
                'target': self._convert_path(item['target'])
            }
            converted_items.append(converted_item)

        # 划分训练/验证集
        random.seed(config.seed)
        random.shuffle(converted_items)
        val_size = int(len(converted_items) * config.val_ratio)
        if split == 'train':
            self.items = converted_items[val_size:]
        else:
            self.items = converted_items[:val_size]

    def _convert_path(self, origin_path):
        """将 JSON 中的路径转换为当前本机的绝对路径"""
        # 统一替换斜杠风格
        std_path = origin_path.replace('\\', '/')
        # 如果路径里包含 'data/extract'，将其分割替换到当前的本地根目录
        if 'data/extract' in std_path:
            rel_path = std_path.split('data/extract')[-1].lstrip('/')
            return os.path.join(self.local_data_root, rel_path)
        else:
            return origin_path

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        mix_path = item['input']
        guitar_path = item['target']

        # 检查文件是否存在
        if not os.path.exists(mix_path):
            raise FileNotFoundError(f"混合音频文件不存在: {mix_path}")
        if not os.path.exists(guitar_path):
            raise FileNotFoundError(f"吉他音频文件不存在: {guitar_path}")

        # 使用 soundfile 读取音频
        mix, sr = sf.read(mix_path, dtype='float32')
        guitar, sr_ = sf.read(guitar_path, dtype='float32')

        # 检查采样率
        assert sr == self.sample_rate and sr_ == self.sample_rate, \
            f"采样率不匹配: {sr} vs {self.sample_rate} (期望 {self.sample_rate})"

        # 如果为立体声，转换为单声道（取平均）
        if mix.ndim > 1:
            mix = np.mean(mix, axis=1)
        if guitar.ndim > 1:
            guitar = np.mean(guitar, axis=1)

        # 转换为 PyTorch 张量，并添加通道维度 (1, T)
        mix = torch.from_numpy(mix).unsqueeze(0)
        guitar = torch.from_numpy(guitar).unsqueeze(0)

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

        return mix, guitar