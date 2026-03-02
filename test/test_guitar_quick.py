#!/usr/bin/env python
# test/test_guitar_quick.py
"""
吉他快速检测脚本
用于快速测试单段音频是否包含吉他
"""

import os
import sys
import argparse

# 修改项目根目录路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.rec.timeline_analyzer import GuitarTimelineAnalyzer

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='吉他快速检测')
    parser.add_argument('audio_path', type=str,
                       help='测试音频路径')
    parser.add_argument('--model-path', type=str, default=None,
                       help='吉他模型路径')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='检测阈值')
    
    args = parser.parse_args()
    
    print("🎸 吉他快速检测")
    print("=" * 50)
    
    # 检查音频文件是否存在
    if not os.path.exists(args.audio_path):
        print(f"❌ 音频文件不存在: {args.audio_path}")
        return
    
    try:
        # 创建吉他分析器
        analyzer = GuitarTimelineAnalyzer(model_path=args.model_path)
        
        print(f"🎵 测试音频: {os.path.basename(args.audio_path)}")
        print(f"🎯 检测阈值: {args.threshold}")
        
        # 检测吉他概率
        prob_guitar = analyzer.detect_guitar_probability(args.audio_path)
        
        print(f"\n📊 检测结果:")
        print(f"  吉他概率: {prob_guitar:.3f}")
        print(f"  检测结果: {'有吉他🎸' if prob_guitar > args.threshold else '无吉他❌'}")
        
        if prob_guitar > args.threshold:
            print(f"  ✅ 置信度: {prob_guitar*100:.1f}%")
        else:
            print(f"  ❌ 置信度: {prob_guitar*100:.1f}%")
        
    except Exception as e:
        print(f"❌ 检测过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()