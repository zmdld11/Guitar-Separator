# src/rec/inference.py
import os
import torch
import librosa
import numpy as np
import soundfile as sf

from src.rec.config import GuitarRecConfig
from src.rec.model import SimplifiedGuitarClassifier

class GuitarDetector:
    """吉他检测器"""
    
    def __init__(self, model_path=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 加载模型
        if model_path is None:
            model_path = os.path.join(GuitarRecConfig.MODEL_DIR, "best_model.pth")
        
        self.model = SimplifiedGuitarClassifier().to(self.device)
        self.load_model(model_path)
        self.model.eval()
    
    def load_model(self, model_path):
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 模型已从 {model_path} 加载")
    
    def extract_features(self, audio, sr):
        """提取音频特征"""
        # 确保音频长度一致
        target_length = GuitarRecConfig.TARGET_SAMPLE_RATE * GuitarRecConfig.AUDIO_DURATION
        audio = librosa.util.fix_length(audio, size=target_length)
        
        # 提取Mel频谱图
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=GuitarRecConfig.N_MELS, fmax=8000, 
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
    
    def detect_audio(self, audio_path, threshold=0.5):
        """检测单段音频中是否有吉他"""
        # 加载音频
        y, sr = librosa.load(audio_path, sr=GuitarRecConfig.TARGET_SAMPLE_RATE)
        
        # 提取特征
        features = self.extract_features(y, sr)
        
        # 转换为模型输入
        input_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 130)
        input_tensor = input_tensor.to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prob_guitar = probabilities[0, 1].item()  # 吉他概率
        
        is_guitar = prob_guitar > threshold
        
        return prob_guitar, is_guitar
    
    def detect_timeline(self, audio_path, window_size=3.0, hop_size=1.0, threshold=0.5):
        """检测音频时间线，返回吉他出现的时间段"""
        # 加载音频
        y, sr = librosa.load(audio_path, sr=GuitarRecConfig.TARGET_SAMPLE_RATE)
        duration = len(y) / sr
        
        print(f"🎵 分析音频: {audio_path}")
        print(f"⏱️  音频时长: {duration:.2f}秒")
        
        # 计算窗口参数
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)
        
        # 滑动窗口检测
        guitar_segments = []
        current_segment = None
        
        for start in range(0, len(y) - window_samples + 1, hop_samples):
            end = start + window_samples
            window_audio = y[start:end]
            
            # 提取特征
            features = self.extract_features(window_audio, sr)
            
            # 转换为模型输入
            input_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                prob_guitar = probabilities[0, 1].item()
            
            timestamp = start / sr
            is_guitar = prob_guitar > threshold
            
            # 打印进度
            if start % (sr * 5) < hop_samples:  # 每5秒打印一次
                print(f"⏳ {timestamp:.1f}s: 吉他概率={prob_guitar:.3f} {'(吉他)' if is_guitar else ''}")
            
            # 合并连续的时间段
            if is_guitar:
                if current_segment is None:
                    current_segment = {
                        'start': timestamp,
                        'end': timestamp + window_size,
                        'confidence': prob_guitar
                    }
                else:
                    # 扩展当前时间段
                    current_segment['end'] = timestamp + window_size
                    current_segment['confidence'] = max(current_segment['confidence'], prob_guitar)
            else:
                if current_segment is not None:
                    # 向前后延长1秒（如项目说明中提到的）
                    current_segment['start'] = max(0, current_segment['start'] - 1.0)
                    current_segment['end'] = min(duration, current_segment['end'] + 1.0)
                    guitar_segments.append(current_segment)
                    current_segment = None
        
        # 添加最后一个时间段
        if current_segment is not None:
            current_segment['start'] = max(0, current_segment['start'] - 1.0)
            current_segment['end'] = min(duration, current_segment['end'] + 1.0)
            guitar_segments.append(current_segment)
        
        # 合并重叠或接近的时间段
        merged_segments = self._merge_segments(guitar_segments, gap_threshold=2.0)
        
        # 打印结果
        print(f"\n🎸 检测到 {len(merged_segments)} 个吉他时间段:")
        for i, segment in enumerate(merged_segments):
            segment_duration = segment['end'] - segment['start']
            print(f"  {i+1}. {segment['start']:.1f}s - {segment['end']:.1f}s "
                  f"({segment_duration:.1f}s, 置信度: {segment['confidence']:.3f})")
        
        return merged_segments
    
    def _merge_segments(self, segments, gap_threshold=2.0):
        """合并接近的时间段"""
        if not segments:
            return []
        
        segments.sort(key=lambda x: x['start'])
        merged = [segments[0]]
        
        for segment in segments[1:]:
            last = merged[-1]
            
            # 如果时间段重叠或间隔小于阈值，则合并
            if segment['start'] <= last['end'] + gap_threshold:
                last['end'] = max(last['end'], segment['end'])
                last['confidence'] = max(last['confidence'], segment['confidence'])
            else:
                merged.append(segment)
        
        return merged
    
    def extract_guitar_audio(self, audio_path, output_path, threshold=0.5):
        """提取吉他音频片段到文件"""
        # 检测吉他时间段
        guitar_segments = self.detect_timeline(audio_path, threshold=threshold)
        
        if not guitar_segments:
            print("❌ 未检测到吉他，无法提取音频")
            return
        
        # 加载原始音频
        y, sr = librosa.load(audio_path, sr=GuitarRecConfig.TARGET_SAMPLE_RATE)
        
        # 提取所有吉他时间段
        guitar_audio = np.array([])
        for segment in guitar_segments:
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            segment_audio = y[start_sample:end_sample]
            guitar_audio = np.concatenate([guitar_audio, segment_audio])
        
        # 保存吉他音频
        sf.write(output_path, guitar_audio, sr)
        
        total_duration = len(guitar_audio) / sr
        print(f"✅ 吉他音频已提取到: {output_path}")
        print(f"⏱️  提取的总时长: {total_duration:.2f}秒")
        print(f"🎯 从原始音频中提取了 {len(guitar_segments)} 个吉他片段")