# ViT 引用脉络（关键参考文献）

> 按对理解本文的重要性排序；每条给概览与摘要要点，帮助判断是否链式深读。
> 来源标注：[论文本身引用] / [外部信源:arXiv]。arXiv ID 基于公开认知，[Agent推断] 处表示未在本轮逐条核实。

- 《Attention Is All You Need》(2017) · Vaswani et al. · 来源：arXiv:1706.03762 [外部信源:arXiv] · 概览：Transformer 原始论文，提出自注意力与编码器-解码器栈，ViT 直接复用其编码器。 · 摘要要点：用自注意力替代循环；多头注意力；位置编码；机器翻译 SOTA。→ **必读**（ViT 的架构基石）。

- 《BERT: Pre-training of Deep Bidirectional Transformers》(2019) · Devlin et al. · 来源：arXiv:1810.04805 [外部信源:arXiv] · 概览：NLP 预训练范式代表，ViT 借鉴其 `[class]` token 与 Base/Large 配置。 · 摘要要点：掩码语言建模 + 下一句预测，微调刷新多项 NLP 基准。→ **必读**（理解 ViT 设计哲学）。

- 《Language Models are Few-Shot Learners》(GPT-3, 2020) · Brown et al. · 来源：arXiv:2005.14165 [外部信源:arXiv] · 概览：论证 Transformer 随规模不饱和，ViT 引其为"大规模训练"可行性的依据。 · 摘要要点：1750 亿参数，少样本学习，性能随规模持续提升。→ **建议读**（理解 ViT"大模型不饱和"信念来源）。

- 《Deep Residual Learning for Image Recognition》(ResNet, 2016) · He et al. · 来源：arXiv:1512.03385 [外部信源:arXiv] · 概览：ViT 的主要 CNN 对照基线（BiT 基于 ResNet）；残差连接思想也用于 ViT 编码器。 · 摘要要点：残差学习解决深层退化，152 层 ImageNet 夺冠。→ **必读**（对照背景）。

- 《On the Relationship between Self-Attention and Convolutional Layers》(2020) · Cordonnier et al. · 来源：arXiv:1911.03584 [外部信源:arXiv] [Agent推断:ID] · 概览：ViT 最相关工作，对 2×2 patch 做全自注意力并证明与卷积的关系；ViT 与之的差异是规模预训练 + 中等分辨率。 · 摘要要点：多头自注意力可表达卷积；2×2 patch 全注意力在小图有效。→ **建议读**（理解 ViT 的新颖性边界）。

- 《Generative Pretraining from Pixels》(iGPT, 2020) · Chen et al. · 来源：arXiv:2002.09773 [外部信源:arXiv] [Agent推断:ID] · 概览：另一条"Transformer 用于视觉"路线，像素级生成式自监督；ViT 对照其 72% ImageNet。 · 摘要要点：降分辨率/降色后做自回归/掩码预训练，线性探测 ImageNet 72%。→ **建议读**（对比 patch 监督 vs 像素自监督）。

- 《Big Transfer (BiT)》(2020) · Kolesnikov et al. · 来源：arXiv:1912.11370 [外部信源:arXiv] [Agent推断:ID] · 概览：ViT 主要 CNN 基线，大 ResNet 监督迁移；ViT-L/16 在同数据上超越它且省算力。 · 摘要要点：大规模监督预训练 + GroupNorm/标准化卷积，迁移 SOTA。→ **建议读**（理解对照设置）。

- 《Self-Training with Noisy Student Improves ImageNet Classification》(2020) · Xie et al. · 来源：arXiv:1911.04252 [外部信源:arXiv] [Agent推断:ID] · 概览：当时 ImageNet SOTA，ViT-H/14 仅以微小优势超越它。 · 摘要要点：教师-学生 + 噪声增广半监督，EfficientNet-L2 达 ImageNet 88.4%。→ **可选读**（理解 ImageNet SOTA 语境）。

- 《End-to-End Object Detection with Transformers》(DETR, 2020) · Carion et al. · 来源：arXiv:2005.12872 [外部信源:arXiv] · 概览：ViT 在结论中引其为"Transformer 用于其他视觉任务有潜力"的佐证。 · 摘要要点：用 Transformer 做检测，去除锚框/NMS，端到端集合预测。→ **建议读**（视觉 Transformer 任务拓展）。

- 《Generating Long Sequences with Sparse Transformers》(2019) · Child et al. · 来源：arXiv:1904.10509 [外部信源:arXiv] [Agent推断:ID] · 概览：稀疏注意力近似全局注意力，是 ViT §2 提到的"硬件不友好"路线对照。 · 摘要要点：因子化稀疏注意力，O(N√N) 处理长序列。→ **可选读**（理解为何 ViT 选择朴素路线）。

- 《ImageNet: A Large-Scale Hierarchical Image Database》(2009) · Deng et al. · 来源：CVPR 2009 · 概览：ViT 预训练/评估数据集基础。 · 摘要要点：1.2M 图像、1000 类层级结构。→ **背景知识**。

## 派生后续工作（非原文引用，便于链式追踪）
- 《Training data-efficient image transformers & distillation through attention》(DeiT, 2021) · Touvron et al. · arXiv:2012.12877 [Agent推断] · 概览：用知识蒸馏+强增广让 ViT 在 ImageNet 单数据集也能训好，回应"小数据欠佳"。
- 《Swin Transformer: Hierarchical Vision Transformer using Shifted Windows》(2021) · Liu et al. · arXiv:2103.14030 [Agent推断] · 概览：层级+移位窗口注意力，解决高分辨率序列过长，扩展到检测/分割。
- 《Masked Autoencoders Are Scalable Vision Learners》(MAE, 2022) · He et al. · arXiv:2111.06377 [Agent推断] · 概览：兑现 ViT §4.6/§5 提到的自监督潜力，掩码重建自监督大幅逼近监督预训练。
