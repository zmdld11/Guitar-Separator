# src/sep/inference.py
import os
import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from src.sep.config import GuitarSeparationConfig
from src.sep.model import GuitarSeparationModel
from src.rec.timeline_analyzer import GuitarTimelineAnalyzer  # 使用二分模型

class GuitarSeparator:
    """吉他分离器（结合二分模型）"""
    
    def __init__(self, sep_model_path=None, rec_model_path=None, device=None):
        self.config = GuitarSeparationConfig()
        
        # 设置设备
        if device is None:
            self.device = self.config.get_device()
        else:
            self.device = device
        
        # 加载分离模型
        self.sep_model = GuitarSeparationModel(self.config).to(self.device)
        if sep_model_path is None:
            sep_model_path = os.path.join(self.config.MODEL_DIR, "best_model.pth")
        self.load_separation_model(sep_model_path)
        self.sep_model.eval()
        
        # 加载二分模型
        if rec_model_path is None:
            rec_model_path = os.path.join(self.config.REC_MODEL_DIR, "best_model.pth")
        
        if os.path.exists(rec_model_path):
            self.rec_analyzer = GuitarTimelineAnalyzer(model_path=rec_model_path, device=self.device)
            print("✅ 二分模型加载成功")
        else:
            self.rec_analyzer = None
            print("⚠️  二分模型未找到，将处理整个音频")
        
        print(f"🎸 吉他分离器初始化完成，使用设备: {self.device}")
    
    def load_separation_model(self, model_path):
        """加载分离模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"分离模型文件不存在: {model_path}")
        
        # 使用weights_only=False加载
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.sep_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 分离模型已从 {model_path} 加载")
    
    def separate_audio(self, audio_path, output_path=None, use_rec_model=True, 
                      threshold=0.5, extend_time=1.0):
        """
        分离吉他音轨
        
        Args:
            audio_path: 输入音频路径
            output_path: 输出音频路径
            use_rec_model: 是否使用二分模型
            threshold: 二分模型阈值
            extend_time: 扩展时间（秒）
        """
        print(f"🎸 分离吉他音轨: {os.path.basename(audio_path)}")
        
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=self.config.SAMPLE_RATE, mono=False)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])  # 转换为立体声
        
        original_length = audio.shape[1]
        print(f"⏱️  音频时长: {original_length / sr:.2f}秒")
        
        # 如果没有二分模型或不使用，直接处理整个音频
        if not use_rec_model or self.rec_analyzer is None:
            print("⚠️  未使用二分模型，处理整个音频...")
            guitar_audio = self._process_segment(audio, sr)
        else:
            # 使用二分模型检测吉他时间段
            print("🔍 使用二分模型检测吉他时间段...")
            
            # 保存临时音频文件用于二分模型分析
            temp_path = "temp_for_detection.wav"
            sf.write(temp_path, audio.T, sr)
            
            # 检测吉他时间段
            timeline, _, _ = self.rec_analyzer.analyze_audio_timeline(
                temp_path,
                window_size=3.0,
                hop_size=1.0,
                threshold=threshold,
                extend_before=extend_time,
                extend_after=extend_time
            )
            
            # 删除临时文件
            os.remove(temp_path)
            
            segments = timeline['segments']
            print(f"📊 检测到 {len(segments)} 个吉他时间段")
            
            # 创建输出数组（初始为静音）
            guitar_audio = np.zeros_like(audio)
            
            if segments:
                # 处理每个吉他时间段
                for i, segment in enumerate(segments):
                    start_time = segment['start']
                    end_time = segment['end']
                    
                    print(f"🔄 处理时间段 {i+1}: {start_time:.1f}s - {end_time:.1f}s")
                    
                    # 提取时间段音频
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    segment_audio = audio[:, start_sample:end_sample]
                    
                    # 分离吉他
                    separated = self._process_segment(segment_audio, sr)
                    
                    # 放回原位置
                    guitar_audio[:, start_sample:end_sample] = separated
            else:
                print("⚠️  未检测到吉他时间段，返回静音")
        
        # 保存输出
        if output_path is None:
            audio_name = Path(audio_path).stem
            output_path = os.path.join(self.config.OUTPUT_DIR, f"{audio_name}_guitar.wav")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存音频
        sf.write(output_path, guitar_audio.T, sr)
        print(f"✅ 吉他音轨已保存到: {output_path}")
        
        return guitar_audio
    
    def _process_segment(self, audio, sr):
        """处理音频段"""
        # 确保长度合适
        target_length = int(self.config.DURATION * sr)
        audio_len = audio.shape[1]
        
        if audio_len < target_length:
            # 如果太短，填充
            padding = target_length - audio_len
            audio = np.pad(audio, ((0, 0), (0, padding)), mode='constant')
        elif audio_len > target_length:
            # 如果太长，分块处理
            return self._process_long_audio(audio, sr)
        
        # 转换为张量
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)  # (1, 2, length)
        
        # 分离
        with torch.no_grad():
            separated = self.sep_model(audio_tensor)
            separated = separated.squeeze(0).cpu().numpy()  # (2, length)
        
        # 裁剪回原始长度
        if audio_len < target_length:
            separated = separated[:, :audio_len]
        
        return separated
    
    def _process_long_audio(self, audio, sr):
        """处理长音频（分块处理）"""
        print(f"🔄 分块处理长音频: {audio.shape[1] / sr:.1f}秒")
        
        target_length = int(self.config.DURATION * sr)
        hop_length = target_length // 2  # 50%重叠
        
        output = np.zeros_like(audio)
        weight_sum = np.zeros(audio.shape[1])
        
        # 分块处理
        for start in range(0, audio.shape[1], hop_length):
            end = start + target_length
            if end > audio.shape[1]:
                break
            
            # 提取块
            chunk = audio[:, start:end]
            
            # 转换为张量并处理
            chunk_tensor = torch.FloatTensor(chunk).unsqueeze(0).to(self.device)
            with torch.no_grad():
                separated_chunk = self.sep_model(chunk_tensor)
                separated_chunk = separated_chunk.squeeze(0).cpu().numpy()
            
            # 使用汉宁窗重叠相加
            window = np.hanning(target_length)
            separated_chunk = separated_chunk * window
            
            # 累加到输出
            output[:, start:end] += separated_chunk
            weight_sum[start:end] += window
        
        # 避免除零
        weight_sum[weight_sum == 0] = 1
        
        # 归一化
        for i in range(output.shape[0]):
            output[i] /= weight_sum
        
        return output

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='吉他音轨分离')
    parser.add_argument('audio_path', type=str, help='输入音频路径')
    parser.add_argument('--output-path', type=str, default=None, help='输出音频路径')
    parser.add_argument('--sep-model', type=str, default=None, help='分离模型路径')
    parser.add_argument('--rec-model', type=str, default=None, help='二分模型路径')
    parser.add_argument('--no-rec', action='store_true', help='不使用二分模型')
    parser.add_argument('--threshold', type=float, default=0.5, help='二分模型阈值')
    parser.add_argument('--extend', type=float, default=1.0, help='时间段扩展秒数')
    
    args = parser.parse_args()
    
    try:
        separator = GuitarSeparator(
            sep_model_path=args.sep_model,
            rec_model_path=args.rec_model
        )
        
        guitar_audio = separator.separate_audio(
            args.audio_path,
            output_path=args.output_path,
            use_rec_model=not args.no_rec,
            threshold=args.threshold,
            extend_time=args.extend
        )
        
        print("🎸 吉他分离完成!")
        
    except Exception as e:
        print(f"❌ 分离失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()