# Swin Transformer 基础知识

本目录单独记录 Swin Transformer 的概念学习，不与 `vision_transformer_practice` 混放。

`images/` 保存与各课对应的 SVG 教学示意图，包括窗口划分、移动窗口、mask、相对位置偏置、Patch Merging、Swin Block 和四阶段结构。

当前笔记：

- `01_SwinTransformer概述与四阶段结构.ipynb`：理解 Window Attention、Shifted Window、Patch Merging 和四阶段层级结构。
- `02_WindowAttention窗口划分与QKV形状.ipynb`：跟踪窗口划分、QKV、多头注意力和窗口还原的完整 shape。
- `03_ShiftedWindow移动窗口与AttentionMask.ipynb`：理解窗口移动、循环移位、Attention Mask 和跨窗口信息传播。
- `04_相对位置偏置_RelativePositionBias.ipynb`：理解相对位移、偏置表、索引矩阵及其与 Attention logits 的结合。
- `05_PatchMerging层级下采样.ipynb`：理解 2 × 2 token 合并、通道压缩和层级下采样。
- `06_SwinBlock完整结构.ipynb`：理解 LayerNorm、残差连接、MLP 及 W-MSA/SW-MSA Block 的完整组成。
- `07_SwinTransformer完整结构与复盘.ipynb`：串联四个 Stage、分类头、CNN/ViT 对比和理论完成标准。

原始 Swin Transformer 核心理论已经完成。后续只有明确进入预训练推理、shape 跟踪或微调实践时才加入必要代码。
