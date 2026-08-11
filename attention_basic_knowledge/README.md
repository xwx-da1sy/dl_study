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
-> 3Blue1Brown 视角复盘 Attention 与 Transformer
-> Multi-Head Attention
-> 位置编码
-> Transformer Encoder
```

## 当前学习

- `00_序列建模与Attention前置背景.ipynb`：预备课，补齐理解 QKV 所需的序列建模背景：token、位置、向量表示、上下文、RNN 的顺序传递思路，以及 Attention 为什么直接建模位置关系。
- `01_Attention机制概述.ipynb`：第一课，从 CNN-MNIST 后的新问题出发，理解 Attention 为什么出现、如何从加权汇总一步步推出“相关性分数 -> Softmax -> 注意力权重 -> 加权求和”，以及它和 MLP、CNN 的区别。
- `02_Query_Key_Value是什么.ipynb`：第二课，理解 Query、Key、Value 的分工：Q 和 K 用来计算相关性分数，Softmax 把分数变成权重，V 才是真正被加权汇总的内容。
- `03_Self-Attention自注意力机制.ipynb`：第三课，理解 Self-Attention 为什么是同一份输入内部“自己看自己”，以及每个位置如何通过 Q、K、V 参考整句话并更新自己的表示。
- `04_Self-Attention矩阵形式与形状变化.ipynb`：第四课，把 Self-Attention 写成矩阵形式，重点理解 `QK^T`、按行 Softmax、乘以 V 这几步的形状变化。
- `05_用3Blue1Brown视角重讲Attention与Transformer.ipynb`：桥接复盘课，参考 3Blue1Brown 的讲课顺序，从 token、embedding、上下文改写、形容词更新名词的例子，重新理解 QKV、注意力图、Value 更新和 Transformer 层层改写向量的整体图景。

## 第一阶段目标

学完预备课、前四课和 3Blue1Brown 复盘课后，需要先能说清楚：

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
15. Self-Attention 中的 self 指什么。
16. 为什么每个位置都可以直接看同一句话里的所有位置。
17. 为什么 Self-Attention 会得到 `N x N` 的注意力表。
18. Self-Attention 输出为什么仍然保留每个位置，但每个位置已经融合上下文。
19. `QK^T` 为什么会得到 `N x N` 的分数表。
20. Softmax 为什么通常对注意力分数表逐行计算。
21. 为什么注意力权重乘以 V 后输出形状是 `N x d_v`。
22. 加上 batch 后，Self-Attention 的主要形状如何从 `B x N x D` 变化到 `B x N x d_v`。
23. 为什么初始 embedding 还没有充分吸收上下文。
24. Attention Block 为什么可以理解成让 token 向量互相传递信息。
25. 如何用“形容词更新名词”的例子解释 Query、Key、Value。
26. 为什么 Multi-Head 可以理解成多个关系角度并行工作。
27. GPT 中 mask 的基本作用是什么。
