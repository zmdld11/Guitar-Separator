# src/rec/config.py
import os

class GuitarRecConfig:
    """吉他识别二分模型配置"""
    
    # 路径配置 - 修正根目录路径
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "rec_model")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    
    # 数据集路径
    POSITIVE_SAMPLES_DIR = os.path.join(DATA_DIR, "extract", "moisesdb", "guitar_segments")
    NEGATIVE_SAMPLES_DIR = os.path.join(DATA_DIR, "extract", "negative_samples")
    
    # 音频处理配置
    TARGET_SAMPLE_RATE = 22050
    AUDIO_DURATION = 3  # 秒
    N_MELS = 128
    
    # 训练配置
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    
    # 模型配置
    INPUT_SHAPE = (1, 128, 130)  # (channels, height, width)
    
    # 训练策略
    USE_DATA_AUGMENTATION = True
    EARLY_STOPPING_PATIENCE = 10
    DROPOUT_RATE = 0.3
    WEIGHT_DECAY = 0.0001
    LABEL_SMOOTHING = 0.1
    
    # 类别权重（解决不平衡问题）
    USE_CLASS_WEIGHTS = True
    
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