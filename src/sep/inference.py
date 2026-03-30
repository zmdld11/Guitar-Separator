import torch
import os
import librosa
import numpy as np
import soundfile as sf
from src.sep.model import MiniBSRoFormer
from src.sep.config import Config
from src.rec.timeline_analyzer import GuitarTimelineAnalyzer

class GuitarSeparator:
    def __init__(self, sep_model_path=None, rec_model_path=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.config = Config()
        self.sep_model = self._load_sep_model(sep_model_path)
        
        # 加载二分模型（用来做前置过滤，如果需要的话可以提高分离效率）
        self.rec_analyzer = GuitarTimelineAnalyzer(model_path=rec_model_path, device=self.device)

    def _load_sep_model(self, model_path):
        if model_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_path = os.path.join(project_root, "sep_model", "checkpoints", "best_model.pth")
            if not os.path.exists(model_path):
                model_path = os.path.join(project_root, "sep_model", "checkpoints", "last_checkpoint.pth")
                
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Separation model not found at {model_path}")
            
        print(f"🎸 正在加载吉他分离模型 (V4.0自动适配架构): {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 实例化当前的 MiniBSRoFormer
        model = MiniBSRoFormer(self.config).to(self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # load_state_dict加载权重
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def separate_audio(self, audio_path, output_path=None, use_rec_model=True, threshold=0.5, extend_time=1.0):
        print(f"\n🎵 开始分离处理: {os.path.basename(audio_path)} ...")
        
        if use_rec_model:
            print("  --- 1. 吉他存活时间段检测 ---")
            timeline, _, _ = self.rec_analyzer.analyze_audio_timeline(
                audio_path, threshold=threshold, window_size=3.0, hop_size=1.0
            )
            # timeline 返回的直接是包含吉他的片段字典 [{'start': ..., 'end': ...}, ...]，包裹在 'segments' 键中
            active_segments = timeline.get('segments', [])
            if not active_segments:
                print("  => 此音频未检测到强烈吉他片段，输出静音轨。")
                audio_info = sf.info(audio_path)
                return np.zeros((audio_info.frames,))
        else:
            # 不用识别模型，全曲通过分离模型
            audio_duration = librosa.get_duration(path=audio_path)
            active_segments = [{'start': 0.0, 'end': audio_duration}]

        # 载入音频
        print("  --- 2. 开始分离吉他音轨 ---")
        audio, sr = librosa.load(audio_path, sr=self.config.sample_rate, mono=True)
        final_guitar_audio = np.zeros_like(audio)
        
        with torch.no_grad():
            for i, seg in enumerate(active_segments):
                start_time = max(0, seg['start'] - extend_time)
                end_time = seg['end'] + extend_time
                
                start_idx = int(start_time * sr)
                end_idx = int(end_time * sr)
                
                if start_idx >= len(audio): continue
                end_idx = min(len(audio), end_idx)
                
                segment_audio = audio[start_idx:end_idx]
                
                if len(segment_audio) == 0: continue
               
                # 转换到张量 [Batch, Channels, Time] -> [1, 1, Time]
                seg_tensor = torch.from_numpy(segment_audio).float().unsqueeze(0).unsqueeze(0).to(self.device)
                
                # 为了防止爆显存，超过15秒的长音频切块处理
                max_chunk_length = sr * 15 
                out_segment_audio_list = []
                
                for chunk_start in range(0, seg_tensor.size(2), max_chunk_length):
                    chunk_end = min(chunk_start + max_chunk_length, seg_tensor.size(2))
                    chunk = seg_tensor[:, :, chunk_start:chunk_end]
                    
                    # 自动混合精度加速推理
                    with torch.amp.autocast('cuda', enabled=self.config.use_amp):
                        out_chunk = self.sep_model(chunk)
                    
                    # 转回 numpy array
                    out_chunk_np = out_chunk.squeeze(0).squeeze(0).cpu().numpy()
                    out_segment_audio_list.append(out_chunk_np)
                
                if out_segment_audio_list:
                    out_segment_audio = np.concatenate(out_segment_audio_list, axis=-1)
                    actual_len = min(len(out_segment_audio), end_idx - start_idx)
                    
                    existing_slice = final_guitar_audio[start_idx:start_idx+actual_len]
                    mask_zero = existing_slice == 0
                    
                    # 拼接赋值，防止重叠部分爆音
                    final_guitar_audio[start_idx:start_idx+actual_len] = np.where(
                        mask_zero, 
                        out_segment_audio[:actual_len], 
                        (existing_slice + out_segment_audio[:actual_len]) / 2.0
                    )
                    
                print(f"    - 处理进度: 片段 {i+1}/{len(active_segments)} ({start_time:.1f}s - {end_time:.1f}s) 处理完成。")
                
        if output_path is not None:
            sf.write(output_path, final_guitar_audio, sr)
            print(f"✅ 分离完成，吉他伴奏已保存至: {output_path}")
            
        return final_guitar_audio
