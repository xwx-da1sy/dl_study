# Attention 注意力机制学习导航

这个目录承接已经完成的 PyTorch、MLP、CNN 与 CNN-MNIST 学习。

现在已经具备学习 Attention 所需的主要基础：

- Tensor 形状与矩阵乘法
- Softmax
- 前向传播与反向传播
- 分类任务的训练流程
- CNN 的局部特征提取

这一阶段先补齐序列建模所需的最小背景，再学习 Attention 机制本身，之后逐步进入 Self-Attention、Multi-Head Attention 和 Transformer。不会一开始就堆 Transformer 代码。

## 学习主线

```text
序列建模前置背景
-> token、位置与向量表示
-> 为什么一个位置要参考其他位置
-> RNN 的历史思路与长距离信息问题
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

- `00_序列建模与Attention前置背景.ipynb`：预备课，补齐理解 QKV 所需的序列建模背景：token、位置、向量表示、上下文、RNN 的顺序传递思路，以及 Attention 为什么直接建模位置关系。
- `01_Attention机制概述.ipynb`：第一课，从 CNN-MNIST 后的新问题出发，理解 Attention 为什么出现、如何从加权汇总一步步推出“相关性分数 -> Softmax -> 注意力权重 -> 加权求和”，以及它和 MLP、CNN 的区别。
- `02_Query_Key_Value是什么.ipynb`：第二课，理解 Query、Key、Value 的分工：Q 和 K 用来计算相关性分数，Softmax 把分数变成权重，V 才是真正被加权汇总的内容。

## 第一阶段目标

学完预备课和前两课后，需要先能说清楚：

1. 什么是序列、token、位置和向量表示。
2. 为什么一个位置的含义经常需要参考其他位置。
3. RNN 的经典思路为什么是按顺序传递信息。
4. 为什么 Attention 改成直接计算位置之间的关系。
5. 为什么处理一条输入时，不同信息的重要程度可能不同。
6. Attention Mechanism 为什么可以理解成一种“动态加权汇总机制”。
7. 相关性分数、注意力权重和加权结果分别是什么。
8. Softmax 为什么适合把相关性分数变成注意力权重。
9. Attention 与 Linear 层里的固定参数权重有什么区别。
10. Attention 与 CNN 的局部特征提取有什么不同。
11. 为什么 Attention 会带来 `N x N` 级别的关系计算。
12. Query、Key、Value 分别承担什么职责。
13. 为什么 Q、K、V 通常来自同一份输入，但通过不同可学习变换得到。
14. 为什么 Q 和 K 用来匹配，V 用来汇总。
