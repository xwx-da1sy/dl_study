# DETR 基础知识

本目录承接已经完成的 CNN、Attention、Transformer、ViT 与 Swin Transformer 学习，开始进入目标检测与 DETR。

这一阶段先理解目标检测任务和 DETR 的核心思想，再逐步拆解 Backbone、位置编码、Encoder、Decoder、Object Queries、集合预测损失与推理流程。不会一开始就堆训练代码。

## 学习主线

```text
图像分类与目标检测的区别
-> 传统检测器为什么需要 Anchor 和 NMS
-> DETR 的端到端集合预测
-> CNN Backbone 与二维位置编码
-> Transformer Encoder
-> Object Queries 与 Transformer Decoder
-> 二分图匹配（匈牙利匹配）
-> 分类、边界框与 GIoU 损失
-> DETR 推理流程
-> 原始 DETR 的局限与后续改进方向
```

## 当前笔记

- `01_DETR概述与集合预测.ipynb`：从图像分类过渡到目标检测，理解 DETR 的整体结构、Object Queries、一对一集合预测和匈牙利匹配。
- `02_DETR输入表示_Backbone与二维位置编码.ipynb`：跟踪图片、padding mask、CNN 特征图、1 x 1 通道投影、图像 tokens 和二维位置编码的完整 shape，明确 Encoder 的实际输入。

## 第一阶段目标

完成第一课后，需要能说清楚：

1. 图像分类与目标检测的输出有什么不同。
2. 为什么检测结果不能只用一个类别向量表示。
3. 传统检测器中的 Anchor、候选框和 NMS 分别解决什么问题。
4. DETR 为什么被称为端到端目标检测器。
5. CNN Backbone、Encoder 和 Decoder 各自负责什么。
6. Object Query 为什么更像一个可学习的“预测槽位”，而不是固定物体或固定位置。
7. 为什么 DETR 把一张图片的目标看成无序集合。
8. 为什么训练时必须先进行一对一匹配。
9. `no object` 类别有什么作用。
10. DETR 的类别输出和边界框输出分别是什么形状。
