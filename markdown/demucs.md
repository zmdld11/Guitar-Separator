# Demucs 音乐源分离

[![Support Ukraine](https://img.shields.io/badge/Support-Ukraine-FFD500?style=flat&labelColor=005BBB)](https://opensource.fb.com/support-ukraine)
![tests badge](https://github.com/facebookresearch/demucs/workflows/tests/badge.svg)
![linter badge](https://github.com/facebookresearch/demucs/workflows/linter/badge.svg)


**重要提示：** 由于我不再在 Meta 工作，**此存储库不再维护**。
我在 [github.com/adefossez/demucs](https://github.com/adefossez/demucs) 创建了一个分支。请注意，此项目不再积极维护，
只有重要的错误修复会在新存储库中处理。请不要为功能请求或 Demucs 不能完美满足您的用例而提出问题 :)

这是 Demucs 的第 4 版发布（v4），具有基于混合 Transformer 的源分离功能。
**对于经典的 Hybrid Demucs (v3)：** [转到此提交][demucs_v3]。
如果您遇到问题并想要旧版 Demucs，请提出问题，然后您可以使用 `git checkout v3` 回到 Demucs v3。您也可以转到 [Demucs v2][demucs_v2]。


Demucs 是一个最先进的音乐源分离模型，目前能够从其余伴奏中分离出
鼓、贝斯和人声。
Demucs 基于受 [Wave-U-Net][waveunet] 启发的 U-Net 卷积架构。
v4 版本具有 [Hybrid Transformer Demucs][htdemucs]，这是一个使用 Transformer 的混合频谱图/波形分离模型。
它基于 [Hybrid Demucs][hybrid_paper]（此存储库中也提供），并将最内层
替换为跨域 Transformer 编码器。该 Transformer 在每个域内使用自注意力，
并在跨域时使用交叉注意力。
该模型在 MUSDB HQ 测试集上实现了 9.00 dB 的 SDR。此外，当使用稀疏注意力
内核扩展其感受野并进行每个声源的微调时，我们实现了最先进的 9.20 dB SDR。

样本可在 [我们的样本页面](https://ai.honu.io/papers/htdemucs/index.html) 上找到。
查看 [我们的论文][htdemucs] 以获取更多信息。
它已在 [MUSDB HQ][musdb] 数据集 + 800 首歌曲的额外训练数据集上训练。
该模型可为任何歌曲分离鼓、贝斯、人声和其他音轨。


由于 Hybrid Transformer Demucs 是全新的，默认情况下未激活，您可以在后述的
常规命令中使用 `-n htdemucs_ft` 激活它。
单一、非微调的模型以 `-n htdemucs` 提供，重新训练的基线
以 `-n hdemucs_mmi` 提供。我们论文中描述的稀疏混合 Transformer 模型未提供，因为其
需要尚未准备好发布的自定义 CUDA 代码。
我们还在发布一个实验性的 6 声源模型，该模型添加了 `吉他` 和 `钢琴` 声源。
快速测试似乎显示 `吉他` 的质量尚可，但 `钢琴` 声源存在大量串音和伪影。


<p align="center">
<img src="./demucs.png" alt="表示 Hybrid Transformer Demucs 结构的示意图，
    具有双重 U-Net 结构，一个分支用于时域，
    一个分支用于频域。编码器和解码器之间有一个跨域 Transformer。"
width="800px"></p>



## 如果您已经在使用 Demucs，重要新闻

有关更多详细信息，请参阅 [发布说明](./docs/release.md)。

- 2023年2月22日：添加了对 [SDX 2023 挑战赛](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023) 的支持，
    请参阅专用 [文档页面](./docs/sdx23.md)
- 2022年12月7日：Demucs v4 现已在 PyPI 上发布。**htdemucs** 模型现在默认使用。同时发布
    一个 6 声源模型（添加 `吉他` 和 `钢琴`，尽管后者目前效果不佳）。
- 2022年11月16日：添加了新的 **Hybrid Transformer Demucs v4** 模型。
	添加对 [torchaudio 实现的 HDemucs](https://pytorch.org/audio/stable/tutorials/hybrid_demucs_tutorial.html) 的支持。
- 2022年8月30日：添加了可重现性和消融网格，以及论文的更新版本。
- 2022年8月17日：发布 v3.0.5：设置分割段长度以减少内存。与 pyTorch 1.12 兼容。
- 2022年2月24日：发布 v3.0.4：分成两个音轨（即卡拉 OK 模式）。
    导出为 float32 或 int24。
- 2021年12月17日：发布 v3.0.3：错误修复（感谢 @keunwoochoi），GPU 内存大幅
    减少（感谢 @famzah）以及 CPU 上的新多核评估（`-j` 标志）。
- 2021年11月12日：发布 **Demucs v3**，具有混合域分离功能。在所有声源上都有显著改进
	。这是赢得索尼 MDX 挑战赛的模型。
- 2021年5月11日：添加对 MusDB-HQ 和任意 wav 集的支持，用于 MDX 挑战赛。有关更多信息
关于使用 Demucs 参加挑战赛，请参阅 [Demucs MDX 说明](docs/mdx.md)


## 与其他模型的比较

我们在此提供论文中呈现的不同指标的摘要。
您还可以在 [我的 Soundcloud 播放列表][soundcloud] 上比较 Hybrid Demucs (v3)、[KUIELAB-MDX-Net][kuielab]、[Spleeter][spleeter]、Open-Unmix、Demucs (v1) 和 Conv-Tasnet 在我最喜欢的歌曲之一上的表现。

### 准确性比较

`总体 SDR` 是 4 个声源中每个声源 SDR 的平均值，`MOS 质量` 是由人类听众给出的
自然度和无伪影的 1 到 5 的评分（5 = 无伪影），`MOS 污染`
是 1 到 5 的评分，5 表示零其他声源污染。我们请读者参考我们的 [论文][hybrid_paper]，
以获取更多详细信息。

| 模型                         | 域     | 额外数据？      | 总体 SDR | MOS 质量 | MOS 污染 |
| ---------------------------- | ------ | --------------- | -------- | -------- | -------- |
| [Wave-U-Net][waveunet]       | 波形   | 否              | 3.2      | -        | -        |
| [Open-Unmix][openunmix]      | 频谱图 | 否              | 5.3      | -        | -        |
| [D3Net][d3net]               | 频谱图 | 否              | 6.0      | -        | -        |
| [Conv-Tasnet][demucs_v2]     | 波形   | 否              | 5.7      | -        |          |
| [Demucs (v2)][demucs_v2]     | 波形   | 否              | 6.3      | 2.37     | 2.36     |
| [ResUNetDecouple+][decouple] | 频谱图 | 否              | 6.7      | -        | -        |
| [KUIELAB-MDX-Net][kuielab]   | 混合   | 否              | 7.5      | **2.86** | 2.55     |
| [Band-Split RNN][bandsplit]  | 频谱图 | 否              | **8.2**  | -        | -        |
| **Hybrid Demucs (v3)**       | 混合   | 否              | 7.7      | **2.83** | **3.04** |
| [MMDenseLSTM][mmdenselstm]   | 频谱图 | 804 首歌曲      | 6.0      | -        | -        |
| [D3Net][d3net]               | 频谱图 | 1.5k 首歌曲     | 6.7      | -        | -        |
| [Spleeter][spleeter]         | 频谱图 | 25k 首歌曲      | 5.9      | -        | -        |
| [Band-Split RNN][bandsplit]  | 频谱图 | 1.7k (仅混合物) | **9.0**  | -        | -        |
| **HT Demucs f.t. (v4)**      | 混合   | 800 首歌曲      | **9.0**  | -        | -        |


