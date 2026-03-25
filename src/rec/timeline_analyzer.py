# src/rec/timeline_analyzer.py
"""
吉他时间线分析器
用于检测音频中吉他出现的时间段
"""

import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 设置matplotlib使用英文字体，避免中文警告
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import Rectangle
from tqdm import tqdm
import soundfile as sf
from scipy.ndimage import gaussian_filter1d

from src.rec.config import GuitarRecConfig
from src.rec.model import SimplifiedGuitarClassifier, AdvancedGuitarClassifier, GuitarClassifier

class GuitarTimelineAnalyzer:
    """吉他时间线分析器"""
    
    def __init__(self, model_path=None, device=None, model_type=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 加载模型路径
        if model_path is None:
            model_path = os.path.join(GuitarRecConfig.MODEL_DIR, "best_model.pth")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        # 先读取 checkpoint，动态判断模型架构 (这样 test 脚本就不需要手动指定模型结构了！)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if 'config' in checkpoint and 'model_type' in checkpoint['config']:
            loaded_model_type = checkpoint['config']['model_type']
            print(f"🔍 从权重中读取到模型架构: {loaded_model_type}")
            if model_type is not None and model_type != loaded_model_type:
                print(f"⚠️ 警告: 传入的 model_type ({model_type}) 与权重记录 ({loaded_model_type}) 不符，强制使用权重记录的架构！")
            self.model_type = loaded_model_type
            
            # 读取特征输入形状
            input_shape = checkpoint['config'].get('input_shape', GuitarRecConfig.INPUT_SHAPE)
        else:
            print("⚠️ 权重中没有保存模型配置，使用默认的 advanced 或传入的参数")
            self.model_type = model_type if model_type else 'advanced'
            input_shape = GuitarRecConfig.INPUT_SHAPE
            
        # 根据模型类型动态创建模型
        if self.model_type == 'advanced':
            self.model = AdvancedGuitarClassifier(input_shape).to(self.device)
        elif self.model_type == 'standard':
            self.model = GuitarClassifier(input_shape).to(self.device)
        elif self.model_type == 'simplified':
            self.model = SimplifiedGuitarClassifier(input_shape).to(self.device)
        else:
            raise ValueError(f"未知的模型类型: {self.model_type}")
        
        # 挂载权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"🎸 吉他时间线分析器初始化完成，使用设备: {self.device}, 模型类型: {self.model_type}")
    
    def load_model(self, model_path):
        """兼容性保留，但在 __init__ 中已完成实际加载"""
        pass
    
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
    
    def detect_guitar_probability(self, audio_path):
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
        
        return prob_guitar
    
    def analyze_audio_timeline(self, audio_path, window_size=3.0, hop_size=1.0, threshold=0.5, 
                               smooth_sigma=0.5, min_segment_duration=0.3, 
                               extend_before=1.0, extend_after=1.0):
        """
        分析音频时间线，检测吉他出现的时间段
        
        Args:
            audio_path: 音频文件路径
            window_size: 分析窗口大小（秒） - 改为3秒匹配训练
            hop_size: 窗口跳跃大小（秒）
            threshold: 检测阈值
            smooth_sigma: 高斯平滑参数 - 减小到0.5
            min_segment_duration: 最小段持续时间
            extend_before: 向前扩展秒数
            extend_after: 向后扩展秒数
        """
        print(f"🎸 分析音频时间线: {audio_path}")
        print(f"⚙️  参数: 窗口={window_size}s, 跳跃={hop_size}s, 阈值={threshold}")
        print(f"⚙️  平滑: sigma={smooth_sigma}")
        print(f"⚙️  扩展: 向前{extend_before}s, 向后{extend_after}s")
        
        # 加载完整音频
        y, sr = librosa.load(audio_path, sr=GuitarRecConfig.TARGET_SAMPLE_RATE)
        duration = len(y) / sr
        print(f"🎵 音频时长: {duration:.2f}秒, 采样率: {sr}Hz")
        
        # 计算窗口参数
        window_samples = int(window_size * sr)
        hop_samples = int(hop_size * sr)
        
        # 计算总窗口数用于进度条
        total_windows = max(0, (len(y) - window_samples)) // hop_samples + 1
        total_windows = max(0, total_windows)
        
        # 滑动窗口分析（使用进度条）
        guitar_probs = []  # 每窗的吉他概率
        timestamps = []    # 每窗的时间戳
        
        with tqdm(total=total_windows, desc="🎸 分析吉他时间窗口", unit="窗") as pbar:
            for start in range(0, max(1, len(y) - window_samples + 1), hop_samples):
                end = start + window_samples
                if end > len(y):
                    break
                window_audio = y[start:end]
                timestamp = start / sr  # 当前窗口开始时间
                
                # 提取特征
                features = self.extract_features(window_audio, sr)
                if features is not None:
                    # 转换为模型输入
                    input_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
                    input_tensor = input_tensor.to(self.device)
                    
                    # 预测
                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        prob_guitar = probabilities[0, 1].item()
                    
                    guitar_probs.append(prob_guitar)
                    timestamps.append(timestamp)
                
                pbar.update(1)
        
        print("✅ 时间序列分析完成!")
        
        # 应用高斯平滑减少抖动
        if smooth_sigma > 0 and len(guitar_probs) > 1:
            guitar_probs_smoothed = gaussian_filter1d(guitar_probs, sigma=smooth_sigma)
            print(f"📊 应用高斯平滑 (sigma={smooth_sigma})")
        else:
            guitar_probs_smoothed = guitar_probs
        
        # 处理时间线结果
        timeline = self._process_timeline_results(
            guitar_probs_smoothed, timestamps, window_size, threshold,
            min_segment_duration, extend_before, extend_after
        )
        
        return timeline, guitar_probs_smoothed, timestamps
    
    def _process_timeline_results(self, guitar_probs, timestamps, window_size, threshold=0.5, 
                                 min_duration=0.3, extend_before=1.0, extend_after=1.0):
        """处理时间线结果，合并连续的时间段"""
        timeline = {
            'segments': [],
            'total_duration': 0.0,
            'max_confidence': 0.0,
            'average_confidence': 0.0
        }
        
        # 找到吉他出现的时间段
        segments = []
        current_segment = None
        
        for i, (timestamp, prob) in enumerate(zip(timestamps, guitar_probs)):
            is_guitar = prob > threshold
            
            if is_guitar:
                if current_segment is None:
                    # 开始新的时间段
                    current_segment = {
                        'start': timestamp,
                        'end': timestamp + window_size,
                        'confidence': prob,
                        'probabilities': [prob]
                    }
                else:
                    # 扩展当前时间段
                    current_segment['end'] = timestamp + window_size
                    current_segment['confidence'] = max(current_segment['confidence'], prob)
                    current_segment['probabilities'].append(prob)
            else:
                if current_segment is not None:
                    # 计算平均置信度
                    if current_segment['probabilities']:
                        current_segment['avg_confidence'] = np.mean(current_segment['probabilities'])
                    
                    # 向前后扩展
                    current_segment['start'] = max(0, current_segment['start'] - extend_before)
                    current_segment['end'] = min(timestamps[-1] + window_size, 
                                                current_segment['end'] + extend_after)
                    
                    # 只添加达到最小时长的段
                    segment_duration = current_segment['end'] - current_segment['start']
                    if segment_duration >= min_duration:
                        segments.append(current_segment)
                    
                    current_segment = None
        
        # 添加最后一个时间段
        if current_segment is not None:
            if current_segment['probabilities']:
                current_segment['avg_confidence'] = np.mean(current_segment['probabilities'])
            current_segment['start'] = max(0, current_segment['start'] - extend_before)
            current_segment['end'] = min(timestamps[-1] + window_size, 
                                        current_segment['end'] + extend_after)
            segment_duration = current_segment['end'] - current_segment['start']
            if segment_duration >= min_duration:
                segments.append(current_segment)
        
        # 合并重叠或接近的时间段
        merged_segments = self._merge_segments(segments, gap_threshold=2.0)
        
        # 过滤掉太短的片段（小于min_duration秒）
        filtered_segments = [s for s in merged_segments if (s['end'] - s['start']) >= min_duration]
        
        # 更新统计信息
        timeline['segments'] = filtered_segments
        if filtered_segments:
            timeline['total_duration'] = sum(s['end'] - s['start'] for s in filtered_segments)
            confidences = [s.get('avg_confidence', s['confidence']) for s in filtered_segments]
            timeline['max_confidence'] = max(confidences)
            timeline['average_confidence'] = np.mean(confidences)
        
        return timeline
    
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
                
                # 合并概率列表
                if 'probabilities' in last and 'probabilities' in segment:
                    last['probabilities'].extend(segment['probabilities'])
                    if last['probabilities']:
                        last['avg_confidence'] = np.mean(last['probabilities'])
            else:
                merged.append(segment)
        
        return merged
    
    def visualize_timeline(self, timeline, guitar_probs, timestamps, audio_duration, save_path=None):
        """可视化时间线结果"""
        # 确保guitar_probs是可迭代的
        if guitar_probs is None or len(guitar_probs) == 0:
            print("⚠️  没有概率数据可可视化")
            return None
        
        # 确保timestamps和guitar_probs长度一致
        if len(timestamps) != len(guitar_probs):
            print(f"⚠️  时间戳({len(timestamps)})和概率({len(guitar_probs)})长度不一致")
            # 取最小长度
            min_len = min(len(timestamps), len(guitar_probs))
            timestamps = timestamps[:min_len]
            guitar_probs = guitar_probs[:min_len]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
        
        # 子图1: 吉他概率曲线
        guitar_probs_array = np.array(guitar_probs)
        ax1.plot(timestamps, guitar_probs_array, 'b-', alpha=0.7, linewidth=1.5)
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold=0.5')
        ax1.fill_between(timestamps, 0, guitar_probs_array, where=(guitar_probs_array > 0.5), 
                        color='red', alpha=0.3, label='Guitar regions')
        ax1.set_xlim(0, audio_duration)
        ax1.set_ylim(0, 1.0)
        ax1.set_xlabel('Time (seconds)', fontsize=12)
        ax1.set_ylabel('Guitar Probability', fontsize=12)
        ax1.set_title('Guitar Detection Probability Curve', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 吉他时间段可视化
        ax2.set_xlim(0, audio_duration)
        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xlabel('Time (seconds)', fontsize=12)
        ax2.set_ylabel('Guitar Detection', fontsize=12)
        ax2.set_title('Guitar Presence Time Segments', fontsize=14, fontweight='bold')
        
        # 绘制吉他时间段
        segments = timeline['segments']
        if segments:
            for i, segment in enumerate(segments):
                # 绘制时间段矩形
                rect = Rectangle(
                    (segment['start'], -0.4),
                    segment['end'] - segment['start'],
                    0.8,
                    facecolor='red',
                    alpha=0.7,
                    edgecolor='black',
                    linewidth=1
                )
                ax2.add_patch(rect)
                
                # 添加置信度文本
                if segment['end'] - segment['start'] > 2:
                    confidence = segment.get('avg_confidence', segment['confidence'])
                    ax2.text(
                        (segment['start'] + segment['end']) / 2,
                        0,
                        f'{confidence:.2f}',
                        ha='center',
                        va='center',
                        fontsize=8,
                        fontweight='bold',
                        color='white'
                    )
        
        # 设置y轴刻度
        ax2.set_yticks([0])
        ax2.set_yticklabels(['Guitar'])
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 子图3: 时长分布直方图
        if segments:
            durations = [s['end'] - s['start'] for s in segments]
            ax3.hist(durations, bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax3.axvline(np.mean(durations), color='red', linestyle='--', label=f'Mean: {np.mean(durations):.2f}s')
            ax3.axvline(np.median(durations), color='green', linestyle='--', label=f'Median: {np.median(durations):.2f}s')
        else:
            ax3.text(0.5, 0.5, 'No guitar segments detected', 
                    ha='center', va='center', transform=ax3.transAxes)
        
        ax3.set_xlabel('Segment Duration (seconds)', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Guitar Segment Duration Distribution', fontsize=14, fontweight='bold')
        if segments:
            ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 时间线图已保存: {save_path}")
        
        plt.show()
        return fig
    
    def extract_guitar_audio_with_silence(self, audio_path, timeline, output_path, preserve_original_sr=True):
        """
        提取吉他音频，非吉他部分用静音覆盖
        保持原始音频长度不变，方便对比
        
        Args:
            audio_path: 原始音频路径
            timeline: 时间线分析结果
            output_path: 输出音频路径
            preserve_original_sr: 是否保持原始采样率
        """
        if not timeline['segments']:
            print("❌ 未检测到吉他时间段，无法提取音频")
            return False
        
        # 加载原始音频，保持原始采样率
        if preserve_original_sr:
            # 获取原始采样率
            y, original_sr = librosa.load(audio_path, sr=None)
        else:
            # 使用目标采样率
            y, original_sr = librosa.load(audio_path, sr=GuitarRecConfig.TARGET_SAMPLE_RATE)
        
        # 创建一个全零数组（静音），长度与原始音频相同
        silent_audio = np.zeros_like(y)
        
        # 将吉他时间段对应的部分替换为原始音频
        for segment in timeline['segments']:
            start_sample = int(segment['start'] * original_sr)
            end_sample = int(segment['end'] * original_sr)
            
            # 确保索引在有效范围内
            start_sample = min(max(0, start_sample), len(y) - 1)
            end_sample = min(max(1, end_sample), len(y))
            
            if start_sample < end_sample:
                silent_audio[start_sample:end_sample] = y[start_sample:end_sample]
        
        # 保存处理后的音频
        sf.write(output_path, silent_audio, original_sr)
        
        # 计算统计信息
        total_guitar_samples = 0
        for segment in timeline['segments']:
            start_sample = int(segment['start'] * original_sr)
            end_sample = int(segment['end'] * original_sr)
            total_guitar_samples += max(0, end_sample - start_sample)
        
        total_duration = len(y) / original_sr
        guitar_duration = total_guitar_samples / original_sr
        
        print(f"✅ 吉他音频(静音覆盖版)已提取到: {output_path}")
        print(f"⏱️  总时长: {total_duration:.2f}s")
        print(f"🎸 吉他时长: {guitar_duration:.2f}s ({guitar_duration/total_duration*100:.1f}%)")
        print(f"🎸 吉他片段数: {len(timeline['segments'])}")
        
        return True
    
    def extract_guitar_only_audio(self, audio_path, timeline, output_path, preserve_original_sr=True):
        """
        只提取吉他音频片段（裁剪掉非吉他部分）
        旧版本，用于对比
        
        Args:
            audio_path: 原始音频路径
            timeline: 时间线分析结果
            output_path: 输出音频路径
            preserve_original_sr: 是否保持原始采样率
        """
        if not timeline['segments']:
            print("❌ 未检测到吉他时间段，无法提取音频")
            return False
        
        # 加载原始音频
        if preserve_original_sr:
            y, original_sr = librosa.load(audio_path, sr=None)
        else:
            y, original_sr = librosa.load(audio_path, sr=GuitarRecConfig.TARGET_SAMPLE_RATE)
        
        # 提取所有吉他时间段
        guitar_audio = np.array([])
        for segment in timeline['segments']:
            start_sample = int(segment['start'] * original_sr)
            end_sample = int(segment['end'] * original_sr)
            
            # 确保索引在有效范围内
            start_sample = min(max(0, start_sample), len(y) - 1)
            end_sample = min(max(1, end_sample), len(y))
            
            if start_sample < end_sample:
                segment_audio = y[start_sample:end_sample]
                guitar_audio = np.concatenate([guitar_audio, segment_audio])
        
        # 保存吉他音频
        sf.write(output_path, guitar_audio, original_sr)
        
        total_duration = len(guitar_audio) / original_sr
        print(f"✅ 吉他音频(裁剪版)已提取到: {output_path}")
        print(f"⏱️  提取时长: {total_duration:.2f}s")
        print(f"🎸 吉他片段数: {len(timeline['segments'])}")
        
        return True
    
    def generate_report(self, timeline, audio_duration, audio_path):
        """生成分析报告"""
        print("\n" + "="*70)
        print("🎸 吉他时间线分析报告")
        print("="*70)
        
        print(f"\n📁 音频文件: {os.path.basename(audio_path)}")
        print(f"⏱️  音频时长: {audio_duration:.2f}秒")
        
        if not timeline['segments']:
            print("\n❌ 未检测到吉他时间段")
            return
        
        segments = timeline['segments']
        print(f"\n🎸 检测到 {len(segments)} 个吉他时间段:")
        print("-" * 70)
        
        total_duration = timeline['total_duration']
        guitar_percentage = (total_duration / audio_duration) * 100
        
        print(f"🎯 吉他总时长: {total_duration:.2f}秒 ({guitar_percentage:.1f}%)")
        print(f"🎯 最高置信度: {timeline['max_confidence']:.3f}")
        print(f"🎯 平均置信度: {timeline['average_confidence']:.3f}")
        
        # 时长统计
        durations = [s['end'] - s['start'] for s in segments]
        print(f"\n📊 时长统计:")
        print(f"  最短片段: {min(durations):.2f}秒")
        print(f"  最长片段: {max(durations):.2f}秒")
        print(f"  平均片段: {np.mean(durations):.2f}秒")
        print(f"  中位数: {np.median(durations):.2f}秒")
        
        print(f"\n📋 详细时间段:")
        print("-" * 60)
        
        for i, segment in enumerate(segments, 1):
            segment_duration = segment['end'] - segment['start']
            confidence = segment.get('avg_confidence', segment['confidence'])
            print(f"  时间段 {i:2d}: {segment['start']:6.1f}s - {segment['end']:6.1f}s "
                  f"({segment_duration:5.1f}s, 置信度: {confidence:.3f})")
        
        return {
            'audio_file': os.path.basename(audio_path),
            'audio_duration': audio_duration,
            'guitar_segments': len(segments),
            'total_guitar_duration': total_duration,
            'guitar_percentage': guitar_percentage,
            'max_confidence': timeline['max_confidence'],
            'average_confidence': timeline['average_confidence'],
            'duration_stats': {
                'min': min(durations),
                'max': max(durations),
                'mean': np.mean(durations),
                'median': np.median(durations)
            }
        }