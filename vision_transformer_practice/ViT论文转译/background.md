# ViT 背景调研

> 论文：An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
> 作者：Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zhai, Unterthiner, Dehghani, Minderer, Heigold, Gelly, Uszkoreit, Houlsby（Google Research）
> arXiv:2010.11929 ｜ 2020-10-22（v1）｜ ICLR 2021 camera-ready（v2, 2021-06-03）
> 代码：https://github.com/google-research/vision_transformer

## 一、论文定位（一句话）
首次证明：把图像切成 patch 序列、直接喂给标准 Transformer 编码器，在大规模预训练下可以匹敌甚至超越当时最强的 CNN，同时预训练算力开销显著更低。

## 二、研究情境（2020 年 CV 领域状态）

**NLP 已被 Transformer 革命，CV 仍被 CNN 主导** [论文本身 §1]
- Transformer（Vaswani et al. 2017）已成 NLP 事实标准；BERT/GPT 路线（大规模预训练 + 微调）使模型规模突破千亿参数。
- 但在视觉，卷积架构（LeCun 1989; AlexNet 2012; ResNet 2016）仍是 SOTA，大尺度图像识别（Mahajan 2018; Xie 2020 Noisy Student; Kolesnikov 2020 BiT）都基于 ResNet。

**此前的"attention 入 CV"尝试与瓶颈** [论文本身 §2]
- 像素级全局自注意力是 O(像素数²)，无法扩展到真实分辨率。
- 已有近似方案：局部自注意力（Parmar 2018; Ramachandran 2019）、稀疏注意力（Child 2019 Sparse Transformer）、分块/分轴注意力（Weissenborn 2019; Wang 2020a）——这些在视觉任务上有 promising 结果，但"需要复杂工程才能在硬件加速器上高效实现"，因而未能在大规模图像识别上取代 ResNet。
- ViT 的判断：与其继续设计特殊注意力模式，不如**用最朴素的方式**（标准 Transformer + patch 序列）并依赖大规模数据来弥补。

## 三、核心突破与定位

1. **去掉 CNN 归纳偏置**：不引入平移等变性 / 局部性 / 二维邻域结构（除切 patch 与位置插值两处），让模型从数据中自己学空间关系。
2. **"大规模训练胜过归纳偏置"**（large scale training trumps inductive bias）——这是全文最核心的论断。
3. **工程朴素性**：尽量贴近原版 Transformer，使 NLP 的高效实现可"开箱即用"。

## 四、关键前身工作（论文明确点出）

- **Cordonnier et al. 2020**：最接近 ViT，对 2×2 patch 做全自注意力；ViT 与之的差异是——(a) 论证大规模预训练使 vanilla Transformer 与 SOTA CNN 竞争；(b) patch 不限于 2×2，可处理中等分辨率图像。
- **iGPT（Chen et al. 2020a）**：把 Transformer 用在（降分辨率/降色后的）像素序列上做生成式自监督，线性探测在 ImageNet 达 72%。ViT 走的是 patch + 监督预训练路线。
- **BERT（Devlin 2019）**：ViT 借鉴其 `[class]` token 作为分类表示；模型配置（Base/Large）也沿用 BERT 命名。
- **BiT（Kolesnikov 2020）与 Noisy Student（Xie 2020）**：作为对照的 SOTA CNN 基线。

## 五、影响力

- **被引 21,938 次**（截至 OpenAlex 当前数据）[外部信源: https://api.openalex.org/works?search=An%20Image%20is%20Worth%2016x16%20Words]——属 CV 领域被引最高的论文之一。
- 开启"视觉 Transformer"研究浪潮，直接催生一系列后续工作：
  - **DeiT（Touvron 2021）**：用知识蒸馏 + 强增广，让 ViT 在 ImageNet 单数据集也能训好，回应了 ViT"小数据欠佳"的局限 [Agent推断]。
  - **Swin Transformer（Liu 2021）**：引入层级结构与移位窗口注意力，解决 ViT 处理高分辨率/检测分割时序列过长的问题 [Agent推断]。
  - **MAE（He 2022）**：掩码自编码器，兑现了 ViT 结尾提到的"自监督预训练有潜力"，自监督 ViT 性能大幅逼近甚至持平监督预训练 [Agent推断]。
  - CLIP、DINO、DINOv2 等视觉自监督/多模态工作普遍以 ViT 为骨干 [Agent推断]。

## 六、可质疑 / 需留意的情境点

- 论文的"胜利"高度依赖两个私有/超大私有数据集（JFT-300M 18k 类 303M 图、ImageNet-21k 14M 图）与 TPUv3 算力，普通研究者难以复现"击败 SOTA"那一档结果；公平小数据对比下 ViT 反而不如 ResNet（见 §4.3）。这一点论文坦承，但容易被"ViT 击败 CNN"的标题式传播淡化。
- 预训练效率比较（TPUv3-core-days）受训练计划/优化器/权重衰减等多因素影响，论文在 §4.4 做了受控缩放研究以缓解，但仍非完全同条件 [论文本身 §4.2]。

## 七、术语速查
见 translation.md 末尾术语表。
