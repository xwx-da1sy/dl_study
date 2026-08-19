# ViT 总体掌握（Overview）

> An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
> Dosovitskiy et al., 2020 (arXiv:2010.11929) ｜ ICLR 2021 ｜ Google Research

## 一句话定位
首个证明「纯 Transformer 直接吃图像 patch 序列 + 大规模预训练」可在图像分类上匹敌甚至超越 SOTA CNN、且预训练更省算力的工作。

## 解决什么问题
2020 年视觉被 CNN 主导，把注意力引入视觉的尝试（局部/稀疏/分轴注意力）都需特殊工程、难在硬件上扩展。问题：能否像 NLP 那样，用**最朴素的标准 Transformer**直接处理图像？

## 核心贡献
1. **方法**：图像 → 切 P×P patch → 线性投影为序列 → 加可学习 1D 位置嵌入 → 标准 Transformer 编码器 → 用 BERT 式 `[class]` token 做分类。
2. **核心论断**：**大规模训练胜过归纳偏置（large scale training trumps inductive bias）**——ViT 几乎不带 CNN 的平移等变性/局部性，小数据上不如 ResNet，但大数据预训练后全面反超。
3. **结果**：ViT-H/14 在 ImageNet 88.55%、ReaL 90.72%、CIFAR-100 94.55%、VTAB 77.63%，且预训练仅需 2.5k TPUv3-core-days（BiT 需 9.9k、Noisy Student 12.3k）；ImageNet-21k 预训练版仅需 0.23k，8 核 TPUv3 ~30 天可复现。

## 方法主线
切 patch → 线性嵌入 → +位置嵌入 → [class] token 前置 → Transformer 编码器（Pre-LN + 残差 + MSA/MLP 交替，MLP 用 GELU）→ z_L^0 经 LN 作分类表示 → 大数据监督预训练 → 下游微调（换线性头 + 高分辨率 + 位置嵌入 2D 插值）。另有 CNN+ViT 混合变体。

## 章节结构地图
| 节 | 内容 | 要点 |
|----|------|------|
| §1 引言 | 背景动机 | NLP 成功、CNN 主导、ViT 思路、核心论断 |
| §2 相关工作 | 前人尝试 | 局部/稀疏/分轴注意力的局限；Cordonnier/iGPT/BERT |
| §3 方法 | 模型设计 | 3.1 ViT（patch/嵌入/编码器/归纳偏置/混合）3.2 微调与高分辨率 |
| §4 实验 | 验证 | 4.1 设置 4.2 vs SOTA 4.3 数据需求 4.4 缩放 4.5 内部表示分析 4.6 自监督初探 |
| §5 结论 | 总结与展望 | 简单可扩展、效果出众；未来=其他任务/自监督/继续扩展 |

## 模型配置
| 模型 | 层数 | 隐藏维 D | MLP 维 | 头数 | 参数 |
|------|------|----------|--------|------|------|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

## 影响力
被引 21,938（OpenAlex）；开启视觉 Transformer 浪潮，催生 DeiT/Swin/MAE/CLIP/DINO 等一系列后续工作，ViT 至今是视觉骨干默认选项之一。
