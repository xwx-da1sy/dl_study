# Attention 注意力机制学习导航

这个目录承接已经完成的 PyTorch、MLP、CNN 与 CNN-MNIST 学习。

现在已经具备学习 Attention 所需的主要基础：

- Tensor 形状与矩阵乘法
- Softmax
- 前向传播与反向传播
- 分类任务的训练流程
- CNN 的局部特征提取

这一阶段先学习 Attention 机制本身，再逐步进入 Self-Attention、Multi-Head Attention 和 Transformer。不会一开始就堆 Transformer 代码。

## 学习主线

```text
Attention 为什么出现
-> 注意力分数与权重
-> Query、Key、Value
-> Self-Attention
-> 矩阵形式与形状变化
-> Multi-Head Attention
-> 位置编码
-> Transformer Encoder
```

## 当前学习

- `01_Attention机制概述.ipynb`：第一课，从 CNN-MNIST 后的新问题出发，理解 Attention 为什么出现、如何从加权汇总一步步推出“相关性分数 -> Softmax -> 注意力权重 -> 加权求和”，以及它和 MLP、CNN 的区别。

## 第一阶段目标

学完第一课后，需要先能说清楚：

1. 为什么处理一条输入时，不同信息的重要程度可能不同。
2. Attention Mechanism 为什么可以理解成一种“动态加权汇总机制”。
3. 相关性分数、注意力权重和加权结果分别是什么。
4. Softmax 为什么适合把相关性分数变成注意力权重。
5. Attention 与 Linear 层里的固定参数权重有什么区别。
6. Attention 与 CNN 的局部特征提取有什么不同。
7. 为什么 Attention 会带来 `N x N` 级别的关系计算。
