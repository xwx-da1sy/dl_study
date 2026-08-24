# Swin Transformer 基础知识

本目录单独记录 Swin Transformer 的概念学习，不与 `vision_transformer_practice` 混放。

当前笔记：

- `01_SwinTransformer概述与四阶段结构.ipynb`：理解 Window Attention、Shifted Window、Patch Merging 和四阶段层级结构。
- `02_WindowAttention窗口划分与QKV形状.ipynb`：跟踪窗口划分、QKV、多头注意力和窗口还原的完整 shape。
- `03_ShiftedWindow移动窗口与AttentionMask.ipynb`：理解窗口移动、循环移位、Attention Mask 和跨窗口信息传播。
- `04_相对位置偏置_RelativePositionBias.ipynb`：理解相对位移、偏置表、索引矩阵及其与 Attention logits 的结合。

后续按概念顺序继续学习 Patch Merging 和完整 Swin Block；只有明确进入实践阶段后才加入必要代码。
