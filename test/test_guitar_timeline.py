#!/usr/bin/env python
"""
吉他时间线分析测试脚本
使用当前项目的配置结构
"""

import os
import sys
import torch
import librosa
import numpy as np
import argparse
import soundfile as sf

# 修改项目根目录路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 导入吉他二分模型相关模块
try:
    # 先尝试导入吉他二分模型配置
    from src.rec.config import GuitarRecConfig
    
    # 导入吉他时间线分析器
    from src.rec.timeline_analyzer import GuitarTimelineAnalyzer
    
    print("✅ 成功导入吉他二分模型模块")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保吉他二分模型文件已正确创建:")
    print("  - src/rec/config.py")
    print("  - src/rec/timeline_analyzer.py")
    sys.exit(1)

def load_guitar_model(model_path=None, device=None):
    """加载吉他模型"""
    try:
        if model_path is None:
            # 使用默认模型路径
            model_path = os.path.join(GuitarRecConfig.MODEL_DIR, "best_model.pth")
            if not os.path.exists(model_path):
                print(f"⚠️  模型文件不存在: {model_path}")
                print("请先训练吉他二分模型 (运行: python src/rec/train.py)")
                return None
        
        analyzer = GuitarTimelineAnalyzer(model_path=model_path, device=device)
        return analyzer
    except Exception as e:
        print(f"❌ 加载吉他模型失败: {e}")
        return None

def find_music_files():
    """查找音乐文件"""
    # 检查多个可能的音乐目录
    possible_dirs = [
        os.path.join(project_root, "music"),
        os.path.join(project_root, "audio"),
        os.path.join(project_root, "test_music"),
        os.path.join(GuitarRecConfig.DATA_DIR, "music"),
    ]
    
    music_files = []
    
    for music_dir in possible_dirs:
        if os.path.exists(music_dir):
            print(f"📁 检查音乐目录: {music_dir}")
            for file in os.listdir(music_dir):
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
                    music_files.append(os.path.join(music_dir, file))
    
    if not music_files:
        # 检查根目录是否有音乐文件
        for file in os.listdir(project_root):
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
                music_files.append(os.path.join(project_root, file))
    
    return music_files

def test_guitar_timeline_with_model(analyzer, audio_path, threshold=0.5, 
                                    window_size=3.0, hop_size=1.0, 
                                    extend_before=1.0, extend_after=1.0):
    """使用吉他分析器测试时间线分析"""
    print(f"\n{'='*60}")
    print(f"🎸 吉他时间线分析: {os.path.basename(audio_path)}")
    print(f"{'='*60}")
    
    try:
        # 分析音频时间线
        print(f"\n📈 开始完整时间线分析(窗口={window_size}s, 跳跃={hop_size}s)...")
        timeline, guitar_probs, timestamps = analyzer.analyze_audio_timeline(
            audio_path, 
            window_size=window_size,
            hop_size=hop_size,
            threshold=threshold,
            extend_before=extend_before,
            extend_after=extend_after
        )
        
        # 获取音频时长
        y, sr = librosa.load(audio_path, sr=None)
        audio_duration = len(y) / sr
        
        # 生成报告
        report = analyzer.generate_report(timeline, audio_duration, audio_path)
        
        # 可视化时间线
        output_dir = os.path.join(GuitarRecConfig.OUTPUT_DIR, "guitar_timeline")
        os.makedirs(output_dir, exist_ok=True)
        
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        # 保存时间线图
        timeline_img_path = os.path.join(output_dir, f"guitar_timeline_{audio_name}.png")
        analyzer.visualize_timeline(timeline, guitar_probs, timestamps, audio_duration, timeline_img_path)
        
        # 打印一些概率信息 - 修复数组判断问题
        if guitar_probs is not None and len(guitar_probs) > 0:
            guitar_probs_array = np.array(guitar_probs)
            print(f"\n📊 概率统计:")
            print(f"  平均概率: {np.mean(guitar_probs_array):.4f}")
            print(f"  最大概率: {np.max(guitar_probs_array):.4f}")
            print(f"  最小概率: {np.min(guitar_probs_array):.4f}")
            print(f"  超过阈值({threshold})的窗口比例: {np.sum(guitar_probs_array > threshold)/len(guitar_probs_array):.2%}")
            
            # 显示概率分布
            prob_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            for i in range(len(prob_bins)-1):
                count = np.sum((guitar_probs_array >= prob_bins[i]) & (guitar_probs_array < prob_bins[i+1]))
                if count > 0:
                    print(f"  概率在[{prob_bins[i]:.1f}-{prob_bins[i+1]:.1f})的窗口: {count}个 ({count/len(guitar_probs_array):.1%})")
        
        # 提取吉他音频（如果检测到吉他）
        if timeline['segments']:
            # 1. 静音覆盖版本
            silence_output_path = os.path.join(output_dir, f"{audio_name}_guitar_with_silence.wav")
            analyzer.extract_guitar_audio_with_silence(audio_path, timeline, silence_output_path, preserve_original_sr=True)
            
            # 2. 裁剪版本（用于对比）
            crop_output_path = os.path.join(output_dir, f"{audio_name}_guitar_only.wav")
            analyzer.extract_guitar_only_audio(audio_path, timeline, crop_output_path, preserve_original_sr=True)
            
            # 3. 保存原始音频副本（用于对比）
            original_copy_path = os.path.join(output_dir, f"{audio_name}_original.wav")
            sf.write(original_copy_path, y, sr)
            
            print(f"\n📁 输出文件保存在: {output_dir}")
            print(f"  - 时间线图: {os.path.basename(timeline_img_path)}")
            print(f"  - 静音覆盖版: {os.path.basename(silence_output_path)} (保持原始长度)")
            print(f"  - 裁剪版: {os.path.basename(crop_output_path)} (只保留吉他)")
            print(f"  - 原始副本: {os.path.basename(original_copy_path)}")
            print("\n🎧 建议对比:")
            print(f"  1. 原始音频 vs 静音覆盖版: 可以听到吉他部分保留，非吉他部分静音")
            print(f"  2. 静音覆盖版 vs 裁剪版: 前者保持原始长度，后者只保留吉他片段")
        else:
            print("\n⚠️  未检测到吉他时间段")
            print("💡 尝试降低阈值: --threshold 0.4")
            print("💡 尝试调整窗口大小: --window-size 2.0 --hop-size 0.5")
            print("💡 尝试增加扩展时间: --extend 1.0")
            print("💡 尝试减小平滑参数: 修改timeline_analyzer.py中的smooth_sigma=0.5")
        
        return report
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主测试函数"""
    parser = argparse.ArgumentParser(description='吉他时间线分析')
    parser.add_argument('--model-path', type=str, default=None,
                       help='吉他模型路径（默认使用rec_model/best_model.pth）')
    parser.add_argument('--audio-path', type=str, 
                       default=None,
                       help='测试音频路径')
    parser.add_argument('--batch', action='store_true',
                       help='批量测试所有找到的音乐文件')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='吉他检测阈值（建议0.4-0.6）')
    parser.add_argument('--window-size', type=float, default=3.0,
                       help='分析窗口大小（秒）')
    parser.add_argument('--hop-size', type=float, default=1.0,
                       help='窗口跳跃大小（秒）')
    parser.add_argument('--extend', type=float, default=1.0,
                       help='时间段前后扩展秒数')
    parser.add_argument('--music-dir', type=str, default=None,
                       help='音乐文件目录')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式，打印更多信息')
    
    args = parser.parse_args()
    
    print("=== 🎸 吉他时间线分析测试 ===")
    
    # 1. 初始化配置
    GuitarRecConfig.create_directories()
    
    # 2. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎮 使用设备: {device}")
    
    if args.batch or args.audio_path is None:
        # 批量测试模式
        if args.music_dir:
            music_dir = args.music_dir
            if os.path.exists(music_dir):
                music_files = [os.path.join(music_dir, f) for f in os.listdir(music_dir) 
                             if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg'))]
            else:
                print(f"❌ 目录不存在: {music_dir}")
                return
        else:
            # 自动查找音乐文件
            music_files = find_music_files()
        
        if not music_files:
            print("❌ 没有找到可测试的音乐文件")
            print("请将音乐文件放在以下目录之一:")
            print("  - 项目根目录/music/")
            print("  - 项目根目录/audio/")
            print("  - 项目根目录/test_music/")
            print("  - 项目根目录/data/music/")
            print("或者使用 --music-dir 参数指定目录")
            return
        
        print(f"📁 批量吉他时间线分析 {len(music_files)} 个音乐文件...")
        print(f"🎯 参数: 窗口={args.window_size}s, 跳跃={args.hop_size}s, 阈值={args.threshold}, 扩展={args.extend}s")
        
        all_reports = []
        
        for music_path in music_files[:3]:  # 限制测试前3个文件
            print(f"\n🎵 分析文件: {os.path.basename(music_path)}")
            
            # 加载吉他分析器
            analyzer = load_guitar_model(args.model_path, device)
            if analyzer is None:
                print("⚠️  跳过此文件...")
                continue
                
            # 分析时间线
            report = test_guitar_timeline_with_model(
                analyzer, music_path, args.threshold, 
                args.window_size, args.hop_size, args.extend, args.extend
            )
            if report:
                all_reports.append(report)
        
        # 输出批量测试总结
        if all_reports:
            print(f"\n{'='*70}")
            print("📊 批量测试总结")
            print(f"{'='*70}")
            
            total_files = len(all_reports)
            files_with_guitar = sum(1 for r in all_reports if r and r['guitar_segments'] > 0)
            
            print(f"📁 分析文件总数: {total_files}")
            print(f"🎸 包含吉他的文件数: {files_with_guitar}")
            
            if files_with_guitar > 0:
                guitar_reports = [r for r in all_reports if r and r['guitar_segments'] > 0]
                avg_guitar_percentage = np.mean([r['guitar_percentage'] for r in guitar_reports])
                print(f"📈 平均吉他占比: {avg_guitar_percentage:.1f}%")
            
            # 按吉他占比排序
            print(f"\n🏆 吉他占比最高的文件:")
            sorted_reports = sorted([r for r in all_reports if r], key=lambda x: x['guitar_percentage'], reverse=True)
            for i, report in enumerate(sorted_reports[:5]):
                if report['guitar_segments'] > 0:
                    print(f"  {i+1}. {report['audio_file']}: {report['guitar_percentage']:.1f}% "
                          f"({report['guitar_segments']}个片段)")
    else:
        # 单个文件测试模式
        if not os.path.exists(args.audio_path):
            print(f"❌ 错误: 测试音频文件不存在 - {args.audio_path}")
            
            # 尝试在可能的目录中查找
            print("🔍 尝试在常见目录中查找...")
            music_files = find_music_files()
            if music_files:
                print(f"📁 找到 {len(music_files)} 个音乐文件:")
                for i, file in enumerate(music_files[:5]):
                    print(f"  {i+1}. {os.path.basename(file)}")
                print("请使用 --batch 参数批量测试，或指定正确的文件路径")
            return
        
        # 加载吉他分析器
        analyzer = load_guitar_model(args.model_path, device)
        if analyzer is None:
            print("❌ 无法加载吉他模型")
            print("请先训练吉他二分模型 (运行: python src/rec/train.py)")
            return
        
        # 分析时间线
        test_guitar_timeline_with_model(
            analyzer, args.audio_path, args.threshold,
            args.window_size, args.hop_size, args.extend, args.extend
        )

if __name__ == "__main__":
    main()