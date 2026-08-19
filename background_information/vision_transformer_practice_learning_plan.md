# Vision Transformer 实践学习计划

更新日期：2026-08-14

## 一、当前学习位置

目前已经完成或正在完成的主线包括：

- PyTorch Tensor、自动求导、Dataset、DataLoader 和标准训练循环
- MLP、激活函数、损失函数、优化器、正则化和归一化
- CNN、LeNet 与 CNN-MNIST 完整训练实践
- Attention、Query、Key、Value
- Self-Attention 与矩阵形状变化
- Multi-Head Attention
- Positional Encoding
- Position-wise FFN
- 残差连接与 LayerNorm
- Transformer Encoder 整体结构

当前的优势是理论零件已经比较完整，能够解释 Attention 和 Transformer Encoder 中主要模块的作用。

当前最需要补充的是：

~~~text
从“能解释结构”
进入“能运行项目、跟踪形状、阅读源码、修改实验、比较结果”
~~~

因此，下一阶段不再继续无限拆解单个公式，而是以 Vision Transformer 为入口，开展实践型学习。

## 二、为什么现在适合学习 ViT

Vision Transformer 可以把已经学过的两条主线连接起来：

~~~text
CNN 阶段积累的图像分类经验
                +
Attention 阶段积累的 Transformer Encoder
                ↓
        Vision Transformer
~~~

ViT 的核心流程是：

~~~text
图像
→ 切分为 patches
→ 将每个 patch 转换成 token 向量
→ 加入 CLS token 和位置编码
→ Transformer Encoder
→ 分类头
~~~

这能帮助我们把 token、Embedding、Multi-Head Attention、FFN、残差连接和 LayerNorm 真正放进一个可训练模型中。

## 三、本地实践环境

当前本地环境：

~~~text
Python：3.13.14
PyTorch：2.10.0
torchvision：0.25.0
CUDA：可用
GPU：NVIDIA GeForce RTX 5060 Laptop GPU
显存：约 8GB
timm：尚未安装
~~~

适合完成：

- CIFAR-10 上的 Tiny ViT 从零训练
- torchvision 预训练 ViT 和 Swin-T 推理
- 小数据集上的预训练模型微调
- 混合精度训练
- 中间张量形状跟踪
- Attention Map 和特征可视化

暂时不把“从零训练 ImageNet 规模模型”作为目标。

## 四、调整后的学习主线

~~~text
Transformer Encoder 整体结构
→ torchvision ViT 推理与源码跟踪
→ 从零实现 Tiny ViT
→ CIFAR-10 训练与 CNN 对比
→ 阅读精简 ViT 开源实现
→ torchvision Swin-T 推理与微调
→ Window Attention、Shifted Window、Patch Merging
→ 阅读微软官方 Swin 工程
→ timm 工程化训练与微调
→ 返回 Transformer Decoder、BERT 和 GPT
~~~

## 五、阶段一：完成 Transformer Encoder 组装

### 学习内容

先完成：

- attention_basic_knowledge/10_Transformer_Encoder整体结构.ipynb

需要能独立画出：

~~~text
Token Embedding + Position Encoding
                ↓
Multi-Head Self-Attention
                ↓
残差连接 + LayerNorm
                ↓
Position-wise FFN
                ↓
残差连接 + LayerNorm
                ↓
一个 Transformer Encoder Layer
~~~

需要能追踪：

~~~text
输入：B x N x D
MHA：B x N x D
Add & Norm：B x N x D
FFN：B x N x d_ff → B x N x D
Add & Norm：B x N x D
输出：B x N x D
~~~

### 完成标准

- 能解释 MHA 和 FFN 的分工
- 能解释为什么 Encoder Layer 输入输出形状通常不变
- 能解释残差连接为什么要求形状一致
- 能解释 LayerNorm 沿哪个维度计算
- 能说明多层 Encoder 如何连续改写 token 表示

## 六、阶段二：torchvision ViT 预训练推理

### 实践目标

使用 torchvision 的预训练 vit_b_16 完成：

1. 加载预训练模型和权重
2. 使用官方预处理方式处理图片
3. 对单张图片进行分类
4. 查看模型结构
5. 记录关键中间张量的形状

重点追踪：

~~~text
B x 3 x 224 x 224
→ Patch Embedding
→ B x 196 x D
→ 加入 CLS token
→ B x 197 x D
→ Transformer Encoder
→ 取 CLS token
→ Linear 分类头
~~~

### 重点阅读源码

PyTorch 官方 ViT 源码：

- [torchvision Vision Transformer](https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py)

第一遍只追踪：

~~~text
_process_input
class_token
EncoderBlock
Encoder
forward
~~~

不要尝试从第一行读到最后一行。

### 完成标准

- 能说明图像如何变成 patch token 序列
- 能计算 patch 数量
- 能解释为什么需要 CLS token
- 能把 ViT 中的 EncoderBlock 对应到已经学过的 Transformer Encoder
- 能记录从输入图片到分类输出的完整 shape

## 七、阶段三：从零实现 Tiny ViT

这是本阶段最重要的实践项目。

### 建议模型配置

~~~text
数据集：CIFAR-10
图像大小：32 x 32
patch_size：4
patch 数量：8 x 8 = 64
模型维度 D：192
Encoder 深度：4～6
注意力头数：3 或 6
FFN 隐藏维度 d_ff：768
类别数：10
~~~

### 需要自己实现的模块

~~~text
PatchEmbedding
Class Token
Position Embedding
Multi-Head Self-Attention
Position-wise FFN
Residual Connection
LayerNorm
Transformer Encoder Block
ViT 分类头
~~~

第一版先使用假数据验证：

~~~text
输入图片
→ 前向传播
→ 输出形状
→ 计算一个假损失
→ backward
→ 检查梯度
~~~

确认模型结构正确以后，再接入 CIFAR-10。

### 训练产出

- 训练集损失曲线
- 验证集损失和准确率曲线
- 测试集准确率
- 最佳模型保存和加载
- 单张图片预测
- 参数量统计
- 与之前 CNN 模型的对比

### 第一组对照实验

每次只改变一个变量：

~~~text
patch_size：4 与 8
depth：4 与 6
heads：3 与 6
是否使用数据增强
是否使用 Dropout
不同学习率
~~~

需要记录的不只是最终准确率，还包括：

- 参数量
- 单个 epoch 时间
- 显存占用
- 训练是否稳定
- 是否出现过拟合

## 八、阶段四：阅读精简 ViT 开源实现

推荐仓库：

- [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)

该仓库适合用来对照自己的实现，但包含很多 ViT 变体，第一遍只阅读基础 ViT。

对照检查：

~~~text
自己的 PatchEmbedding 对应源码中的哪里
自己的 Attention 对应源码中的哪里
自己的 PreNorm 对应源码中的哪里
Encoder Blocks 如何堆叠
CLS token 和位置编码如何加入
最终分类输出如何取得
~~~

### 完成标准

- 能把开源源码中的类和自己的模块一一对应
- 能解释开源实现中至少三处与自己实现不同的设计
- 能修改一个小功能并重新跑通
- 不只是调用仓库 API 得到一个预测结果

## 九、阶段五：Swin Transformer 实践

Swin Transformer 应该放在普通 ViT 之后学习。

### 先理解四个核心变化

~~~text
ViT：全局 Self-Attention
Swin：Window Attention

ViT：平坦的 token 序列
Swin：具有多个 Stage 的层级结构

ViT：通常保持 patch 分辨率
Swin：Patch Merging 逐层降低分辨率、增加通道

普通 Window Attention：窗口之间缺少交流
Swin：Shifted Window 让相邻窗口逐层交换信息
~~~

### 第一轮实践

先使用 torchvision 的 swin_t：

1. 加载预训练权重
2. 完成图片分类推理
3. 替换分类头
4. 在小数据集上微调
5. 记录每个 Stage 的 shape

重点跟踪：

~~~text
224 x 224
→ 56 x 56 x 96
→ 28 x 28 x 192
→ 14 x 14 x 384
→ 7 x 7 x 768
→ 分类头
~~~

PyTorch 官方 Swin 源码：

- [torchvision Swin Transformer](https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py)

### 第二轮实践

阅读微软官方工程：

- [Microsoft Swin Transformer](https://github.com/microsoft/Swin-Transformer)

重点阅读：

~~~text
models
configs
get_started.md
分类任务的训练入口
预训练权重加载方式
~~~

第一阶段不要求从零训练官方 ImageNet 配置，也不要求立刻进入目标检测和语义分割。

### 完成标准

- 能解释 Window Attention 为什么降低计算量
- 能解释 Shifted Window 为什么能让窗口之间交流
- 能说明 Patch Merging 和 CNN 下采样的相似之处
- 能追踪 Swin 四个 Stage 的分辨率和通道变化
- 能对 ViT、Swin、CNN 的结构特点做比较

## 十、阶段六：学习 timm 工程化实践

推荐仓库：

- [pytorch-image-models / timm](https://github.com/huggingface/pytorch-image-models)

timm 包含大量图像模型、预训练权重以及训练、验证、推理脚本。

建议在完成手写 Tiny ViT 和 torchvision Swin 实践后再安装和使用。

学习内容：

~~~text
列出可用模型
创建预训练模型
替换分类头
获取官方数据预处理配置
冻结主干网络
完整微调
使用训练和验证脚本
保存与恢复 checkpoint
~~~

目标不是记住所有 API，而是理解成熟开源工程如何统一管理：

- 模型注册
- 配置
- 预训练权重
- 数据增强
- 优化器
- 学习率调度
- 混合精度
- 训练日志
- checkpoint

## 十一、8GB 显存实践原则

### Tiny ViT

- CIFAR-10 从较小模型开始
- 先尝试合适的小 batch，显存不足时再降低
- 不要一开始堆很深的 Encoder

### 预训练 ViT/Swin 微调

- 输入为 224 x 224 时，从较小 batch 开始
- 优先使用自动混合精度
- 显存不足时先减小 batch，再考虑梯度累积
- 第一轮可以冻结主干，只训练分类头
- 第二轮再解冻部分层或全部层

### 暂不作为目标

- 从零训练 ImageNet
- 一开始就复现论文全部指标
- 同时启动多个大型模型
- 为追求准确率不断堆大模型

## 十二、阅读 GitHub 工程的固定流程

每个开源项目统一按照下面的顺序学习：

~~~text
1. 阅读 README，确认任务、输入、输出和运行方式
2. 先运行最小官方示例
3. 打印模型结构，找到 forward 入口
4. 跟踪关键张量 shape
5. 把源码模块对应到自己的理论笔记
6. 只修改一个变量，重新运行
7. 比较修改前后的结果
8. 记录遇到的问题、解决方法和结论
~~~

不要采用：

~~~text
克隆仓库
→ 安装依赖
→ 复制命令
→ 成功运行
→ 结束
~~~

能够运行只是开始。真正的学习产出应该是：

- 能说清数据如何流动
- 能定位关键源码
- 能解释主要 shape
- 能修改实验
- 能比较结果
- 能复现一次训练或微调

## 十三、建议的实践目录

当前实践目录已经整理为：

~~~text
vision_transformer_practice/
├── 01_torchvision_ViT预训练推理.ipynb
├── 02_ViT输入与PatchEmbedding.ipynb
├── 03_ViT输入组装_CLS与位置编码.ipynb
├── 04_TinyViT_EncoderBlock结构拆解.ipynb
├── 05_训练曲线读图指南.ipynb
├── 06_正则化技术_EarlyStopping与LabelSmoothing.ipynb
├── 07_数据增强原理与实践.ipynb
├── 08_vit-pytorch基础ViT源码对照.ipynb
├── 09_CNN与ViT结构对比.ipynb
├── 10_TinyViT_AttentionMap可视化.ipynb
├── tiny_vit.py
├── train_tiny_vit.py
├── evaluate_tiny_vit.py
└── infer.py
~~~

具体训练代码可以在真正开始对应项目时再创建，不提前堆放空脚本。

## 十四、阶段完成后的下一步

完成 ViT 和 Swin 后，再返回语言 Transformer 主线：

~~~text
Padding Mask
→ Causal Mask
→ Transformer Decoder
→ Cross-Attention
→ 完整 Encoder-Decoder Transformer
→ BERT 与 GPT
→ Hugging Face Transformers
~~~

此时已经通过 ViT/Swin 完整实现和训练过 Transformer Encoder，再学习 BERT、GPT 时，对 Block、Attention、FFN、残差连接和 LayerNorm 会更熟悉。

## 十五、近期执行清单

- [x] 完成 Transformer Encoder 整体结构复盘
- [x] 运行 torchvision vit_b_16 预训练推理
- [x] 跟踪 ViT 从图片到 patch token 的 shape
- [x] 阅读 torchvision ViT 的 forward
- [x] 从零实现 Tiny ViT
- [x] 在 CIFAR-10 上完成训练
- [x] 与已有 CNN-MNIST/CNN 经验做结构对比
- [x] 阅读基础版 vit-pytorch
- [x] 提取并可视化 TinyViT 的 CLS Attention Map
- [ ] 运行 torchvision swin_t
- [ ] 跟踪 Swin 各 Stage 的 shape
- [ ] 完成一次 Swin 小数据集微调
- [ ] 阅读微软官方 Swin 工程
- [ ] 安装并学习 timm
- [ ] 整理 ViT 与 Swin 阶段复盘

## 十六、最终目标

完成这一实践阶段后，应当能够：

1. 独立解释 ViT 如何把图像转换为 token 序列。
2. 独立实现一个能够训练的 Tiny ViT。
3. 使用 PyTorch 完成 ViT/Swin 的预训练推理和微调。
4. 跟踪 ViT 和 Swin 的关键张量形状。
5. 阅读并定位开源工程中的模型入口和前向传播。
6. 对比 CNN、ViT、Swin 的结构特点和适用场景。
7. 修改至少一个模型或训练配置，并通过实验分析影响。
8. 为后续学习 Transformer Decoder、BERT 和 GPT 建立实践基础。
