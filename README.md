# 吉他音轨分离模型训练

该项目受Demucs模型启发，希望借鉴Demucs模型开发经历训练一个吉他音轨分离模型。

## 训练环境

1. 编程语言：python3.10
2. 编程工具：vscode
3. 环境控制：anaconda
4. 硬件情况：
   1. CPU：core I9-14900HX
   2. GPU：nvidia 4060 laptop 8GB
   3. 内存：32G
   4. 硬盘空间：~~2T~~ 4T
5. 数据集：
   1. MedleyDB
   2. moisesdb_v0.1

## 模型结构

学习Demucs模型，采用双U-Net模型结合transformer的形式。将音频文件分别生成波形图和频谱图并行输入时域分支和频域分支处理，并在U-Net模型最底层进行sel-attention和cross-attention。

## 其他要求

1. 尽可能压榨计算机每一份性能。
2. 在论文中，他们进行了数据集扩充。首先训练一个分类器，以在小时间帧上检测每个声源的缺失或存在，使用我们知道每个声源贡献的监督训练集。当我们检测到一个音频片段，其中声源至少有5秒的静音时，我们将其添加到一个新集合 中。然后，我们可以将示例与从监督训练集中获取的单个声源 混合，以形成新的训练数据。

## 已知问题

1. 电脑硬件差，不能按照Demucs模型复现。
2. 训练集很大，moisesdb现在有120g，medley有70g
3. 训练集的分轨音频中有许多空白音频，实测对模型训练影响很大。
4. test_guitar_timeline.py的窗口大小不能小于3s，不然会导致识别不到吉他。

## 有待尝试的解决办法

1. 把数据集中所有的空白音频段剔除，只留下有音频的片段。
2. 利用1中提取出的数据集训练一个吉他识别的二分模型，用于检测一个音频文件在哪些时间段出现了吉他。
3. 用二分模型检测哪些地方没有出现吉他，并直接将那些部分设置为静音。对于那些可能存在吉他的地方，使用训练的分离模型进行音轨分离。由于二分模型检测时滑动窗口导致的识别误差，我觉得需要对二分模型识别出来可能的时间段向前向后延长1s，再进行音轨分离，来保证输出听感的完整性。
4. 由于我们有二分模型，所以我们的音轨分离模型只需要集中处理那些存在吉他的混合音轨数据集，这样我们需要处理的数据量应该可以大大减少，而且不会受空白音频段的影响。

## 文件路径（新）

```txt
instrument_separator/                 # 项目根目录
├── data/                             # 数据集与缓存目录
│   ├── moisesdb_v0.1/                # MoisesDB 原始数据集
│   │   ├── 0a589d65-50a3-4999-8f16-b5b6199bceee/
│   │   │   └── guitar/               # 吉他音轨文件夹
│   │   │       ├── 4b5dc0f9-d4c6-4648-b567-2f9418b856e0.wav
│   │   │       └── ...
│   │   └── ...
│   ├── MedleyDB/                     # MedleyDB 原始数据集
│   │   └── Allegria_MendelssohnMovement1/
│   │       ├── Allegria_MendelssohnMovement1_RAW/
│   │       ├── Allegria_MendelssohnMovement1_STEMS/
│   │       └── Allegria_MendelssohnMovement1_MIX.wav
│   │   └── ...
│   ├── Metadata/                     # MedleyDB 元数据文件夹
│   │   └── xxx.yaml
│   └── extract/                      # 提取处理后的训练数据
│       ├── moisesdb/                 # MoisesDB处理结果
│       │   ├── mixed_guitars/        # 混合后的吉他音轨
│       │   │   └── trackname_mixed_guitar.wav
│       │   ├── guitar_segments/      # 吉他音频片段（目标输出）
│       │   │   └── trackname/
│       │   │       ├── guitar_trackname_seg_0000.wav
│       │   │       └── ...
│       │   ├── mix_segments/         # 混合音频片段（模型输入）
│       │   │   └── trackname/
│       │   │       ├── mix_trackname_seg_0000.wav
│       │   │       └── ...
│       │   └── extraction_record.json
│       ├── medleydb/                 # MedleyDB处理结果
│       │   ├── mixed_guitars/
│       │   ├── guitar_segments/
│       │   ├── mix_segments/
│       │   └── extraction_record.json
│       ├── negative_samples/         # 非吉他片段（二分模型负样本）
│       │   ├── moisesdb/
│       │   │   └── trackname/
│       │   │       ├── drums/
│       │   │       ├── bass/
│       │   │       ├── piano/
│       │   │       ├── vocals/
│       │   │       └── other/
│       │   └── negative_samples_record.json
│       ├── binary_classification/    # 二分模型数据集
│       │   └── dataset.json
│       ├── separation/               # 分离模型数据集
│       │   └── dataset.json
│       ├── dataset_stats.json		  # 数据集总时长、曲目数等信息
│       └── dataset_statistics.csv    # 数据集统计信息
├── extract_active_segments.py        # 音频提取主脚本
├── rec_model/                        # 二分模型存储目录
│   ├── checkpoints/                  # 训练检查点
│   ├── logs/                         # 训练日志
│   └── best_model.pth                # 最佳模型
├── sep_model/                        # 分离模型存储目录
│   ├── checkpoints/
│   ├── logs/
│   └── best_model.pth
├── output/                           # 输出目录
│   ├── logs/                         # 通用日志
│   └── results/                      # 测试结果
├── music/                            # 测试音频文件夹
│   ├── test_input.mp3
│   └── test_output/
├── src/                              # 源代码目录
│   ├── rec/                          # 二分模型相关代码
│   │   ├── config.py                 # 配置文件
│   │   ├── dataset.py                # 数据集加载
│   │   ├── model.py                  # 模型定义
│   │   ├── train.py                  # 训练脚本
│   │   ├── inference.py              # 推理脚本
│   │   └── utils.py                  # 工具函数
│   ├── sep/                          # 分离模型相关代码
│   │   ├── config.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── utils.py
│   └── common/                       # 共用代码
│       ├── audio_utils.py            # 音频处理工具
│       ├── data_utils.py             # 数据预处理工具
│       └── visualization.py          # 可视化工具
└── test/                             # 测试目录
    └── xxx.py                        # 人工测试代码
```

## 项目进度

- [x] 创建extract文件夹，完成有效音频的提取和分类。（共457个吉他样本，总时长约15小时，最小样本时长3s，最长595.6s，moisesdb有366个样本，medleydb有91个样本，加上负训练集共24.5g）
- [x] 使用extract数据集完成二分模型训练。（目前训练到版本1.0，在很多情况下还会出现漏检多检的情况，后续跟进）
- [ ] 使用extract数据集完成分离模型训练。分离模型使用模仿demucs的u-net模型，然后只分离存在吉他的部分，其余由二分模型判断为不存在吉他的部分直接改为静音。
