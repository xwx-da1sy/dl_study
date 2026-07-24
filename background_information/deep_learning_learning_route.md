# 深度学习学习背景与路线

## 一、当前背景信息

### 已掌握内容

- 已学习 Python 基础。
- 已学习一部分 NumPy、Pandas 等数据处理工具。
- 已完成机器学习阶段的学习。
- 当前目标是进入深度学习阶段，并参考黑马程序员深度学习课程进行系统学习。

### 学习定位

你不需要再从 Python、数据分析、传统机器学习重新开始，而是应该直接进入深度学习核心内容：

- PyTorch 深度学习框架
- 神经网络基础
- 反向传播与模型优化
- CNN 图像分类
- RNN 文本生成
- NLP 基础
- Attention 与 Transformer
- 预训练模型与 Hugging Face

### 推荐主线

以黑马程序员《神经网络与深度学习》课程为视频主线，配合以下资料补充：

- PyTorch 官方教程
- 《动手学深度学习》
- Hugging Face 官方文档

学习重点不只是看完课程，而是每个阶段都产出一个可以运行、可以复盘、可以写进简历或作品集的小项目。

## 二、总体学习周期

建议学习周期：8-10 周。

建议每日学习时间：2-3 小时。

每日时间分配：

- 40%：观看黑马程序员深度学习课程
- 40%：代码跟敲、复现、修改实验
- 20%：阅读文档、整理笔记、复盘原理

每周至少完成一个小项目或一个完整实验。

## 三、阶段一：PyTorch 与神经网络基础

建议时间：1-2 周。

### 学习内容

- Tensor 创建、索引、切片、广播机制
- Tensor 与 NumPy 的转换
- GPU / CUDA 基础使用
- 自动求导 `autograd`
- `nn.Module` 的使用
- `Dataset` 与 `DataLoader`
- 损失函数
- 优化器
- 标准训练循环

### 必须掌握

- 能独立写出 PyTorch 训练模板。
- 能解释 `forward()`、`loss.backward()`、`optimizer.step()` 的作用。
- 能理解训练集、验证集、测试集在深度学习中的使用方式。

### 练习项目

- 使用 PyTorch 重写线性回归。
- 使用 PyTorch 实现二分类 MLP。
- 使用 MNIST 完成手写数字识别。

## 四、阶段二：反向传播与深层网络训练

建议时间：1 周。

### 学习内容

- 神经元与感知机
- 多层感知机 MLP
- 激活函数：ReLU、Sigmoid、Tanh、Softmax
- 交叉熵损失
- 梯度下降
- 反向传播
- 学习率、Batch Size、Epoch
- 梯度消失与梯度爆炸
- Dropout
- Batch Normalization
- 权重初始化

### 必须掌握

- 能画出 MLP 的前向传播流程。
- 能理解损失如何通过反向传播更新参数。
- 能根据 loss 和 accuracy 曲线判断训练是否正常。

### 练习项目

- Fashion-MNIST 图像分类。
- 对比不同学习率对训练效果的影响。
- 对比 SGD、Adam 等优化器。
- 记录并可视化 loss / accuracy 曲线。

## 五、阶段三：CNN 图像分类

建议时间：2 周。

### 学习内容

- 卷积操作
- 卷积核、步长、填充、通道
- 池化层
- CNN 网络结构
- LeNet
- AlexNet
- VGG
- ResNet 基本思想
- 图像增强
- 迁移学习

### 必须掌握

- 能理解 `Conv2d(in_channels, out_channels, kernel_size, stride, padding)`。
- 能独立搭建一个简单 CNN。
- 能解释为什么 CNN 适合处理图像数据。
- 能使用预训练模型完成迁移学习。

### 练习项目

- CIFAR-10 图像分类。
- 猫狗图片分类。
- 使用预训练 ResNet 完成自定义图片分类。
- 编写单张图片推理脚本。

## 六、阶段四：RNN 与序列模型

建议时间：1-2 周。

### 学习内容

- 序列数据特点
- RNN 基本结构
- 隐藏状态 hidden state
- LSTM
- GRU
- 文本向量化
- 字符级文本生成
- 词级文本分类

### 必须掌握

- 能解释 RNN 的输入、输出和隐藏状态。
- 能理解 RNN 为什么适合序列数据。
- 能理解 RNN 的局限性，以及 LSTM / GRU 的改进思路。

### 练习项目

- 使用 RNN 实现文本生成。
- 使用 LSTM 完成情感分类。
- 中文文本分词后进行分类实验。

## 七、阶段五：NLP 基础到 Attention

建议时间：1-2 周。

### 学习内容

- one-hot 编码
- Word Embedding
- word2vec 基本思想
- Seq2Seq
- Attention 机制
- 简单机器翻译任务

### 必须掌握

- 能理解词向量相比 one-hot 的优势。
- 能解释 Encoder-Decoder 结构。
- 能理解 Attention 为什么能缓解长序列信息丢失问题。

### 练习项目

- 简单英译中或英译法 Seq2Seq 案例。
- 给 Seq2Seq 增加 Attention。
- 中文新闻标题分类。

## 八、阶段六：Transformer 与预训练模型

建议时间：2 周。

### 学习内容

- Self-Attention
- Q、K、V
- Multi-Head Attention
- Positional Encoding
- Transformer Encoder
- Transformer Decoder
- BERT 与 GPT 的区别
- Hugging Face Transformers 基础使用

### 必须掌握

- 能解释 Self-Attention 的计算流程。
- 能解释 Q、K、V 的含义。
- 能理解 Transformer 为什么适合并行训练。
- 能使用 Hugging Face 完成一个简单文本分类任务。

### 练习项目

- 手写简化版 Self-Attention。
- 使用 Hugging Face 微调中文 BERT 文本分类模型。
- 使用预训练模型完成情感分类或新闻分类。

## 九、阶段七：综合项目

建议时间：1-2 周。

从下面项目中选择 2 个深入完成。

### 项目一：CIFAR-10 图像分类完整项目

要求：

- 数据加载
- 数据增强
- CNN 或 ResNet 模型
- 训练与验证
- 训练曲线可视化
- 模型保存
- 单张图片推理
- README 总结

### 项目二：中文文本分类项目

要求：

- 数据清洗
- 分词或 tokenizer
- Embedding / LSTM / BERT 至少选择一种模型
- 训练与评估
- 混淆矩阵
- 错误样本分析
- README 总结

### 项目三：RNN 文本生成项目

要求：

- 构建字符表
- 构造序列数据
- 训练 RNN 或 LSTM
- 实现文本生成
- 尝试 temperature 采样
- README 总结

### 项目四：简易 Transformer 文本分类

要求：

- 实现 Embedding
- 实现 Self-Attention
- 实现 Encoder Block
- 添加分类头
- 完成训练和评估
- README 总结

## 十、推荐学习顺序

```text
PyTorch 基础
-> MLP 与反向传播
-> MNIST / Fashion-MNIST
-> CNN
-> CIFAR-10 / 迁移学习
-> RNN / LSTM
-> 文本生成 / 文本分类
-> Attention / Seq2Seq
-> Transformer
-> BERT / Hugging Face
-> 综合项目
```

## 十一、每个模块的自检问题

每学完一个模块，至少回答下面 4 个问题：

1. 这个模型主要解决什么问题？
2. 输入和输出的张量形状是什么？
3. 损失函数是怎么定义的？
4. 如果模型效果不好，可以从哪些方面调整？

## 十二、阶段性目标

### 2 周后

- 能独立写 PyTorch 训练流程。
- 能完成 MNIST 或 Fashion-MNIST 分类。

### 4 周后

- 能搭建 CNN。
- 能完成 CIFAR-10 或猫狗分类。

### 6 周后

- 能理解 RNN / LSTM。
- 能完成文本生成或文本分类。

### 8 周后

- 能理解 Attention 和 Transformer。
- 能使用 Hugging Face 微调一个小型文本分类模型。

### 10 周后

- 至少完成 2 个完整项目。
- 每个项目有代码、训练结果、推理脚本和 README。

## 十三、学习建议

- 不要只看视频，一定要写代码。
- 不要只跑通代码，要主动修改超参数观察效果。
- 每个模型都要关注输入输出形状。
- 每个项目都要保留训练记录。
- 学完一个阶段后，用自己的话写一页总结。
- 项目优先级高于刷课程进度。

