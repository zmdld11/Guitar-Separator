# 音频分离模型 (HTDemucs) 架构说明

## 1. 整体架构
当前模型采用 **HT Demucs v4 风格**，是一种双流（时域 + 频域）混合结构的 U-Net 模型。其核心理念是分别在时域（处理瞬态特征）和频域（处理谐波特征）对音频进行编码，然后在一个共享的跨域 Transformer 瓶颈层进行特征融合，最后再通过平行的波形与频谱解码器分别重构信号。

- **总参数量**: 约 52.72 M（对于音轨分离任务属于大型模型结构）。
- **输入**: 单声道（Mono）混合音频波形 `(batch, 1, T)`。
- **输出**: 时域波形输出 与 频域掩膜（Mask）输出（未显示融合步骤的代码）。

---

## 2. 详细模块解析

### 2.1 时域分支 (Time-Domain Branch)
直接对原始音频段（Waveform）进行一维卷积特征提取。
- **结构**: `EncoderLayer` 堆叠 5 层 (`time_depth = 5`)。
- **配置**:
  - `time_channels = 64` (初始通道)。
  - `kernel_size = 8`，`stride = 4`。
  - 最后两层引入了基于 LSTM 和 MultiheadAttention 的残差块 (`use_lstm=True, use_attn=True`)。
- **解码**: `DecoderLayer` 使用 `ConvTranspose1d` 进行转置卷积上采样。跳跃连接方式为直接相加（`x + skip`）配合 `skip_proj` 通道投影对齐。

### 2.2 频域分支 (Frequency-Domain Branch)
对波形进行 STFT 后获得的复数频谱（实部和虚部作为 2 个通道）进行二维特征提取。
- **STFT 设定**: `n_fft = 2048`, `hop_length = 512`, `window = hann`。
- **结构**: `FreqEncoderLayer` 堆叠 5 层 (`freq_depth = 5`)。仅在频率维度上进行下采样，保留时间维度的分辨率。
- **配置**:
  - `freq_channels = 64` (初始通道)。
  - `kernel_size = (8, 4)`，`stride = (4, 4)` (部分注释可能不一致但基本逻辑是时频卷积)。
  - 同样在深层引入 LSTM 和注意力残差块。
  - 配备了 `FreqPositionalEmbedding` 处理频域位置信息。
- **解码**: `FreqDecoderLayer` 进行频率维度的反卷积，输出复数掩膜 `(batch, 2, t, f)`。

### 2.3 跨分支注入融合 (Injection Projection)
在编码器的每个对应层（层级对应），模型通过 `self.inject_projs` (一维卷积或 `Identity`) 将时域特征库中的通道投影至频域通道规格，为跨域特征融合做准备（前向传播中应会有特征合并）。

### 2.4 共享编码/解码与 Transformer 瓶颈 (Shared & Bottleneck)
时频域最深层特征合并后进入共享层：
- **Shared Encoder**: `shared_depth = 1`，通道数扩展至 128 `shared_channels = 128`。
- **CrossTransformer**: 
  - `dim = 512`（或更深），`heads = 8`，`layers = 4`。
  - 采用标准自注意力层与前馈网络 (`FFN`) 的交替结构。捕捉超长感受野内的音符和语义依赖关系。
- **Shared Decoder**: 将融合后的全局特征逐步上采样，为分别送入两个独立解码器准备。

### 2.5 基础算子与技巧
- **GELU 激活函数**: 用于主体非线性变换。
- **LayerScale**: 以微小初始值 (`1e-3`) 给残差分支增加可学习标量缩放，利于深层网络稳定。
- **GLU (Gated Linear Unit)**: 用于特征门控过滤。
- **GroupNorm & LayerNorm**: 分别用于卷积特征和序列特征的规范化。

---

## 3. 为什么参数量达到 52.72M？
1. **庞大的基底前置通道**: 设定起始通道 `time_channels=64`、`freq_channels=64`，并依层倍增。随着层数 `depth=5` 的加深，底层的通道数逼近512甚至更高，导致深层的 `Conv1d / Conv2d` 参数成倍增加。
2. **频域的二维卷积**: `Conv2d` 和 `ConvTranspose2d` 本身参数量就大于 1D 的卷积核。
3. **Transformer 瓶颈**: 4 层 512 维的自注意力加上 FFN (扩大4倍至 2048) 是个参数消耗大户。
4. **大量辅助模块**: 几乎在中间层及底层广泛铺设了双向 LSTM (`bidirectional=True`) 与 MultiheadAttention，这些操作堆叠使得整体参数极速膨胀。
