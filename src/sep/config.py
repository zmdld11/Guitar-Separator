# src/sep/config.py
import os
import json
import torch

class GuitarSeparationConfig:
    """吉他分离模型配置"""
    
    # 路径配置
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "sep_model")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    REC_MODEL_DIR = os.path.join(BASE_DIR, "rec_model")
    
    # 提取后的数据集路径
    EXTRACT_DIR = os.path.join(DATA_DIR, "extract")
    DATASET_JSON = os.path.join(EXTRACT_DIR, "separation", "dataset.json")
    DATASET_STATS = os.path.join(EXTRACT_DIR, "dataset_stats.json")
    
    # 音频处理配置 - 使用更简单的参数确保形状匹配
    SAMPLE_RATE = 22050  # 保持一致的采样率
    DURATION = 2.0  # 减少到2秒，确保能被2整除多次
    HOP_LENGTH = 512
    N_FFT = 1024
    N_MELS = 64
    N_CHANNELS = 2  # 立体声
    
    # 训练配置
    BATCH_SIZE = 1  # 使用batch size 1以避免形状问题
    EPOCHS = 30  # 减少训练轮数
    LEARNING_RATE = 1e-4
    GRADIENT_ACCUMULATION_STEPS = 1  # 不使用梯度累积
    VALIDATION_SPLIT = 0.1
    TEST_SPLIT = 0.1
    
    # 模型配置
    HIDDEN_SIZE = 32
    NUM_ENCODERS = 3
    NUM_TRANSFORMER_LAYERS = 0  # 不使用Transformer
    NUM_HEADS = 4
    DROPOUT_RATE = 0.1
    
    # 训练策略
    USE_AMP = False  # 暂时禁用混合精度训练以简化问题
    USE_GRADIENT_CLIPPING = True
    GRADIENT_CLIP_VALUE = 1.0
    WEIGHT_DECAY = 1e-5
    WARMUP_STEPS = 500
    
    # 数据增强
    USE_AUGMENTATION = False  # 暂时禁用数据增强
    AUGMENT_PROB = 0.0
    TIME_MASK_PROB = 0.0
    FREQ_MASK_PROB = 0.0
    GAIN_RANGE = (-3, 3)
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        directories = [cls.MODEL_DIR, cls.OUTPUT_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 目录已创建: {directory}")
        
        # 在模型目录下创建子目录
        os.makedirs(os.path.join(cls.MODEL_DIR, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(cls.MODEL_DIR, "logs"), exist_ok=True)
        os.makedirs(os.path.join(cls.MODEL_DIR, "tensorboard"), exist_ok=True)
    
    @classmethod
    def get_device(cls):
        """获取训练设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 使用GPU: {torch.cuda.get_device_name(0)}")
            print(f"🎮 GPU内存: {gpu_memory:.2f} GB")
            
            # 设置CUDA内存优化
            torch.cuda.empty_cache()
            
        else:
            device = torch.device('cpu')
            print("🖥️  使用CPU")
            
        return device