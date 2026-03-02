#!/usr/bin/env python
# test/test_separation.py
"""
吉他分离模型测试脚本
"""

import os
import sys
import argparse
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.sep.inference import GuitarSeparator
from src.rec.timeline_analyzer import GuitarTimelineAnalyzer

def find_test_audio():
    """查找测试音频文件"""
    possible_dirs = [
        os.path.join(project_root, "music"),
        os.path.join(project_root, "test_music"),
        os.path.join(project_root, "data", "extract", "moisesdb", "guitar_segments"),
        os.path.join(project_root, "data", "extract", "medleydb", "guitar_segments"),
    ]
    
    audio_files = []
    for test_dir in possible_dirs:
        if os.path.exists(test_dir):
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file.lower().endswith(('.wav', '.mp3', '.flac')):
                        audio_files.append(os.path.join(root, file))
    
    return audio_files[:5]  # 返回前5个文件

def test_separation(args):
    """测试分离模型"""
    print("🎸 吉他分离模型测试")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎮 使用设备: {device}")
    
    # 加载分离器
    try:
        separator = GuitarSeparator(
            sep_model_path=args.sep_model,
            rec_model_path=args.rec_model,
            device=device
        )
    except Exception as e:
        print(f"❌ 加载分离器失败: {e}")
        return
    
    # 如果指定了音频文件，测试单个文件
    if args.audio_path:
        audio_files = [args.audio_path]
    else:
        # 自动查找测试文件
        audio_files = find_test_audio()
        if not audio_files:
            print("❌ 未找到测试音频文件")
            print("请将测试音频放在以下目录之一:")
            for dir_name in ["music", "test_music"]:
                print(f"  - {os.path.join(project_root, dir_name)}")
            return
    
    print(f"📁 找到 {len(audio_files)} 个测试文件")
    
    for audio_path in audio_files:
        if not os.path.exists(audio_path):
            print(f"❌ 文件不存在: {audio_path}")
            continue
        
        print(f"\n🎵 测试文件: {os.path.basename(audio_path)}")
        
        try:
            # 分离吉他音轨
            output_dir = os.path.join(project_root, "output", "separation_test")
            os.makedirs(output_dir, exist_ok=True)
            
            audio_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.join(output_dir, f"{audio_name}_separated.wav")
            
            guitar_audio = separator.separate_audio(
                audio_path,
                output_path=output_path,
                use_rec_model=not args.no_rec,
                threshold=args.threshold,
                extend_time=args.extend
            )
            
            # 可视化结果
            if args.visualize and guitar_audio is not None:
                visualize_results(audio_path, output_path, guitar_audio)
            
            print(f"✅ 测试完成: {audio_path}")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()

def visualize_results(original_path, separated_path, guitar_audio):
    """可视化分离结果"""
    import librosa
    
    # 加载原始音频
    orig_audio, sr = librosa.load(original_path, sr=None, mono=True)
    
    # 计算频谱
    orig_spec = librosa.stft(orig_audio, n_fft=2048, hop_length=512)
    orig_db = librosa.amplitude_to_db(np.abs(orig_spec), ref=np.max)
    
    # 吉他音频频谱（取第一个通道）
    if guitar_audio.ndim > 1:
        guitar_mono = guitar_audio[0] if guitar_audio.shape[0] == 2 else guitar_audio
    else:
        guitar_mono = guitar_audio
    
    # 确保长度一致
    min_len = min(len(orig_audio), len(guitar_mono))
    orig_audio = orig_audio[:min_len]
    guitar_mono = guitar_mono[:min_len]
    
    guitar_spec = librosa.stft(guitar_mono, n_fft=2048, hop_length=512)
    guitar_db = librosa.amplitude_to_db(np.abs(guitar_spec), ref=np.max)
    
    # 绘制频谱对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 原始音频波形
    axes[0, 0].plot(orig_audio[:sr*5])  # 前5秒
    axes[0, 0].set_title('Original Audio (Waveform)')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 吉他音频波形
    axes[0, 1].plot(guitar_mono[:sr*5])
    axes[0, 1].set_title('Separated Guitar (Waveform)')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 原始音频频谱
    librosa.display.specshow(orig_db, sr=sr, hop_length=512, x_axis='time', y_axis='log', ax=axes[1, 0])
    axes[1, 0].set_title('Original Audio (Spectrogram)')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Frequency')
    
    # 吉他音频频谱
    librosa.display.specshow(guitar_db, sr=sr, hop_length=512, x_axis='time', y_axis='log', ax=axes[1, 1])
    axes[1, 1].set_title('Separated Guitar (Spectrogram)')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    
    # 保存图像
    output_dir = os.path.dirname(separated_path)
    img_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(original_path))[0]}_comparison.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"📊 可视化结果已保存: {img_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='吉他分离模型测试')
    parser.add_argument('--audio-path', type=str, default=None, help='测试音频路径')
    parser.add_argument('--sep-model', type=str, default=None, help='分离模型路径')
    parser.add_argument('--rec-model', type=str, default=None, help='二分模型路径')
    parser.add_argument('--no-rec', action='store_true', help='不使用二分模型')
    parser.add_argument('--threshold', type=float, default=0.5, help='二分模型阈值')
    parser.add_argument('--extend', type=float, default=1.0, help='时间段扩展秒数')
    parser.add_argument('--visualize', action='store_true', help='可视化结果')
    parser.add_argument('--batch', action='store_true', help='批量测试')
    
    args = parser.parse_args()
    
    test_separation(args)

if __name__ == "__main__":
    main()