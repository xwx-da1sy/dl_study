# dl_study

该仓库是我的深度学习学习仓库。当前主线是参考黑马程序员《神经网络与深度学习》课程，配合 PyTorch 官方教程和个人练习 notebook，系统进入深度学习阶段。

## 学习目录

- `background_information/`：学习背景与路线规划。
- `pytorch_basic_knowledge/`：PyTorch 基础、Tensor、DataLoader、线性回归等内容。
- `neural_network_basic_knowledge/`：神经网络基础概念（激活函数、MLP、优化器、归一化、MNIST 实战等）。
- `cnn_basic_knowledge/`：CNN 卷积神经网络（图像基础、卷积层、池化层、网络结构、LeNet 等）。
- `attention_basic_knowledge/`：Attention 注意力机制（动态加权、QKV、Self-Attention、Multi-Head Attention、Transformer 等）。
- `nlp_embedding_basic_knowledge/`：NLP Embedding 基础（Tokenization、词表与 Token ID、One-hot 与 Embedding 矩阵），补足文本进入 Transformer 前的表示环节。
- `vision_transformer_practice/`：Vision Transformer 实践（预训练推理、Patch Embedding、Tiny ViT、ViT 论文转译、可视化、CNN 公平对照与 224×224 放大训练）。
- `swin_transformer_basic_knowledge/`：Swin Transformer 基础（Window Attention、Shifted Window、Patch Merging 与层级结构）。
- `swin_transformer_practice/`：Swin Transformer 实践（预训练推理与 shape 对齐、CIFAR-100 自定义 Swin、超参数搜索与训练评估）。
- `swin_transformer_imagenet100_practice/`：在 ImageNet-100 上训练四阶段自定义 Swin（含超参数搜索、训练与评估）。
- `object_detection_basic_knowledge/`：目标检测共同基础（边界框、IoU、目标分配、NMS、mAP 与 YOLO 基本范式）。
- `detr_basic_knowledge/`：DETR 基础（目标检测、Object Queries、集合预测、二分图匹配与检测损失）。

## 学习日志

### 2026-07-24

- 创建 `pytorch_basic_knowledge` 学习文件夹。
- 阅读 `background_information/deep_learning_learning_route.md`，明确当前阶段直接进入深度学习核心内容。
- 开始 PyTorch 基础学习。
- 学习 Tensor 的基本概念和创建方式。
- 编写张量创建练习脚本：
  - `pytorch_basic_knowledge/class01_tensor_creat.py`
- 初步整理 PyTorch 笔记，开始使用 Jupyter Notebook 记录，不使用 Markdown 笔记文件。
- 学习 `Dataset` 和 `DataLoader` 的基础概念：
  - `Dataset` 负责组织样本。
  - `DataLoader` 负责按 batch 读取数据。
- 完成笔记：
  - `pytorch_basic_knowledge/02_数据读取_Dataset与DataLoader.ipynb`

### 2026-07-27

- 继续完善 Tensor 基础内容。
- 学习 Tensor 常用运算：
  - 加减乘除
  - 逐元素运算
  - 矩阵乘法
  - 向量点积
- 学习 Tensor 索引和切片。
- 学习 Tensor 形状变换：
  - `reshape`
  - `view`
  - `flatten`
  - `squeeze`
  - `unsqueeze`
- 重点补充 `unsqueeze(dim)` 中 `dim` 的含义：它表示在 shape 中插入新维度的位置。
- 学习维度交换：
  - `transpose`
  - `t`
  - `permute`
- 学习张量拼接：
  - `torch.cat`
  - `torch.stack`
- 学习广播机制。
- 将原来过长的 PyTorch 第一节 notebook 拆分成多个主题 notebook，便于复习：
  - `pytorch_basic_knowledge/01_PyTorch基础学习导航.ipynb`
  - `pytorch_basic_knowledge/01-01_Tensor基础_创建_dtype与NumPy互转.ipynb`
  - `pytorch_basic_knowledge/01-02_索引_形状变换_维度交换_拼接与广播.ipynb`
  - `pytorch_basic_knowledge/01-03_数学运算_dim_keepdim_max与点积.ipynb`
  - `pytorch_basic_knowledge/01-04_autograd自动求导与最小线性回归.ipynb`
- 编写对应练习脚本：
  - `pytorch_basic_knowledge/class02_tensor_calculate.py`
  - `pytorch_basic_knowledge/class03_tensor_index.py`
  - `pytorch_basic_knowledge/class06_tensor_reshape.py`

### 2026-07-28

- 学习 PyTorch 自动求导 `autograd`。
- 理解：
  - `requires_grad`
  - `loss.backward()`
  - `.grad`
  - 梯度累加
  - `optimizer.zero_grad()`
  - `detach()`
  - `torch.no_grad()`
  - `item()`
- 编写自动求导相关练习脚本：
  - `pytorch_basic_knowledge/class04_tensor_autograd.py`
  - `pytorch_basic_knowledge/class05_tensor_detach.py`
- 系统模拟多维线性回归：
  - 构造多维特征数据。
  - 使用真实参数生成标签。
  - 使用 `TensorDataset` 和 `DataLoader`。
  - 使用 `nn.Linear` 建立多维线性回归模型。
  - 使用 `MSELoss` 作为损失函数。
  - 使用 `SGD` 优化器训练模型。
  - 对比真实参数和模型学习到的参数。
- 完成笔记：
  - `pytorch_basic_knowledge/03_多维线性回归模拟.ipynb`
- 补充优化器学习内容：
  - 优化器负责更新模型参数。
  - `loss.backward()` 负责计算梯度。
  - `optimizer.step()` 负责真正修改参数。
  - SGD 基本更新公式：`新参数 = 旧参数 - 学习率 × 梯度`。
  - 学习 `params`、`lr`、`momentum`、`weight_decay` 的含义。
- 开始神经网络基础学习：
  - `neural_network_basic_knowledge/01_什么是神经网络.ipynb` — 神经元、层、权重、偏置、激活函数等基本结构。
- 学习常见激活函数的公式推导与作用：
  - `neural_network_basic_knowledge/01-01_Sigmoid函数公式推导.ipynb` — 二分类概率映射、梯度消失。
  - `neural_network_basic_knowledge/01-02_Tanh函数公式推导.ipynb` — 零中心化、梯度消失。
  - `neural_network_basic_knowledge/01-03_ReLU函数公式推导.ipynb` — 解决梯度消失、稀疏激活、Dead ReLU。
  - `neural_network_basic_knowledge/01-04_Softmax函数公式推导.ipynb` — 多分类概率输出、交叉熵关系。
- 学习感知机与多层感知机：
  - `neural_network_basic_knowledge/02_感知机与多层感知机.ipynb` — 分类问题、决策边界、线性可分与非线性可分、MLP 结构。

### 2026-07-29

- 深入学习激活函数的优缺点分析与应用场景选择：
  - 补充 Tanh 与 ReLU 的对比分析。
  - 补充 Softmax 与交叉熵损失的关系推导。
- 学习参数初始化：
  - `neural_network_basic_knowledge/01-05_参数初始化.ipynb` — 全零初始化的问题、Xavier 初始化、He（Kaiming）初始化。
  - `neural_network_basic_knowledge/parameter_initialization.py` — Xavier 与 He 的公式推导、fan_in/fan_out、选择规则速记。
  - 理解 Xavier 适合 Tanh/Sigmoid，He 适合 ReLU 家族的原因。
  - 掌握偏置通常初始化为 0 的惯例。
- 深化感知机与多层感知机：
  - 详细讲解分类问题、决策边界、AND 问题与 XOR 问题。
  - 理解单个感知机只能解决线性可分问题，MLP 通过组合多条边界解决非线性问题。
- 学习损失函数与梯度下降：
  - `neural_network_basic_knowledge/03_损失函数与梯度下降.ipynb`
  - 损失函数衡量预测与真实之间的差距。
  - MSE 用于回归，交叉熵用于分类。
  - 梯度下降的更新原理：沿负梯度方向调整参数。
  - 学习率对训练的影响。
- 学习反向传播与链式法则：
  - `neural_network_basic_knowledge/04_反向传播与链式法则.ipynb`
  - 链式法则的核心：复合函数求导。
  - 反向传播的高效性：从输出层向输入层逐层计算梯度。
  - 理解计算图与前向/反向传播的关系。
- 学习神经网络完整训练流程：
  - `neural_network_basic_knowledge/05_神经网络训练流程.ipynb`
  - 完整训练循环：前向传播 → 计算损失 → 反向传播 → 更新参数。
  - 过拟合与欠拟合的概念与表现。
  - 训练集/验证集/测试集的划分意义。
- 学习优化器：
  - `neural_network_basic_knowledge/06_优化器_SGD_Momentum_Adam.ipynb`
  - SGD：基础随机梯度下降。
  - Momentum：引入动量加速收敛、减少震荡。
  - Adam：自适应学习率，结合 Momentum 与 RMSProp 的优点。

### 2026-07-31
- 学习过拟合与正则化：
  - `neural_network_basic_knowledge/07_过拟合与正则化.ipynb`
  - 过拟合：模型在训练集表现好，但在验证集表现差。
  - 欠拟合：模型在训练集和验证集都表现差。
  - 正则化方法：
    - L1/L2 正则化
    - Dropout
    - 数据增强
    - 提前停止（Early Stopping）
- 学习 Batch Normalization 批量归一化：
  - `neural_network_basic_knowledge/08_BatchNormalization批量归一化.ipynb`
  - BN 的作用：加速训练、缓解梯度消失
- 学习指数加权移动平均（EMA）：
  - `neural_network_basic_knowledge/05-01_指数加权移动平均.ipynb`
  - 理解 EMA 的原理：通过对历史值加权平均来平滑数据。
  - 理解 EMA 在优化器（Momentum、Adam）中的作用。
- 学习 MLP 结构设计：
  - `neural_network_basic_knowledge/09_MLP结构设计_输入层隐藏层输出层.ipynb`
  - 理解输入层、隐藏层、输出层的设计与神经元数量选择。
  - 优化 MLP 模型结构：增加隐藏层和神经元数量以提升模型表达能力。
- 整理文件结构，按学习阶段重新组织目录。
- 增强手写数字预测交互体验。

### 2026-08-06

- 继续完善优化器相关内容：
  - 深入理解指数加权移动平均（EMA）与优化器的关系。
- 学习 AdaGrad 优化器：
  - `neural_network_basic_knowledge/06-00_AdaGrad从SGD到自适应学习率.ipynb`
  - 理解 AdaGrad 的自适应学习率原理：根据历史梯度自动调整每个参数的学习率。
  - 理解 AdaGrad 的优缺点：适合稀疏数据，但学习率单调递减可能导致过早停止。
- 学习参数初始化在实战中的应用：
  - 复习 Xavier 初始化（适合 Tanh/Sigmoid）与 He/Kaiming 初始化（适合 ReLU 家族）。
  - 定义更深的 MLP 模型结构，实践参数初始化策略。
  - 实现模型训练和预测脚本，完成 MLP 手写数字识别全流程。

### 2026-08-07

- 进入 CNN（卷积神经网络）学习阶段：
  - 创建 `cnn_basic_knowledge/` 目录，开始系统学习 CNN。
- 学习图像基础知识：
  - `cnn_basic_knowledge/01_图像基础知识.ipynb`
  - 理解图像在计算机中表示为数字矩阵。
  - 理解像素、灰度图与 RGB 彩色图的区别。
  - 理解图像张量形状：灰度图 `1×H×W`，彩色图 `3×H×W`。
  - 实现图像显示函数。
- 学习 CNN 基本概念：
  - `cnn_basic_knowledge/02_CNN概述介绍.ipynb`
  - 理解 CNN 为什么适合处理图像（局部感知、权重共享、层次化特征提取）。
  - 理解 CNN 整体结构：卷积层 → 激活函数 → 池化层 → 全连接层。
- 学习卷积层运算规则：
  - `cnn_basic_knowledge/03_卷积层运算规则.ipynb`
  - 理解卷积核如何滑动计算（逐元素乘加）。
  - 理解 stride（步长）、padding（填充）对输出尺寸的影响。
  - 掌握输出尺寸公式：`(W - K + 2P)/S + 1`。
- 学习多通道卷积计算：
  - `cnn_basic_knowledge/05_多通道卷积计算.ipynb`
  - 理解输入通道与输出通道的关系。
  - 理解多通道卷积的参数量计算。
- 学习多卷积核计算：
  - `cnn_basic_knowledge/06_多卷积核计算.ipynb`
  - 理解多个卷积核如何并列提取不同特征。
  - 理解卷积核个数与输出通道数的对应关系。
- 学习特征图（Feature Map）：
  - `cnn_basic_knowledge/07_特征图.ipynb`
  - 理解特征图的定义、来源及其在 CNN 中的重要性。
  - 理解浅层特征图（边缘、纹理）与深层特征图（语义信息）的区别。

### 2026-08-10

- 学习池化层（Pooling Layer）原理：
  - `cnn_basic_knowledge/08_池化层原理与运算规则.ipynb`
  - 理解池化层的作用：压缩特征图尺寸、减少参数量、防止过拟合。
  - 掌握最大池化（Max Pooling）与平均池化（Average Pooling）的运算规则。
  - 理解池化不会改变通道数。
- 学习 CNN 整体网络结构：
  - `cnn_basic_knowledge/09_CNN整体网络结构.ipynb`
  - 把卷积层、激活函数、池化层、展平和全连接层串联成完整 CNN 结构。
  - 理解数据在 CNN 中的流动过程。
- 学习 Flatten 与全连接分类头：
  - `cnn_basic_knowledge/10_Flatten与全连接分类头.ipynb`
  - 理解特征图如何展平（Flatten）成一维向量。
  - 理解全连接分类头如何将特征向量映射到类别分数。
- 学习 LeNet 经典网络结构：
  - `cnn_basic_knowledge/11_LeNet经典网络结构.ipynb`
  - 理解 LeNet 如何用卷积、池化、展平和全连接完成手写数字分类。
  - 掌握 LeNet 的层次结构和参数配置。
- 完成 CNN-MNIST 实战：
  - 完整实现数据加载、模型、损失函数、Adam、训练、评估、模型保存与单张图片预测。
  - 正式训练 5 个 epoch，测试准确率达到 98% 以上。
- 进入 Attention 学习阶段：
  - 创建 `attention_basic_knowledge/` 学习目录。
  - 完成 `attention_basic_knowledge/01_Attention机制概述.ipynb`。
  - 初步理解注意力分数、Softmax、注意力权重和动态加权汇总。

### 2026-08-11

- 补充序列建模与 Attention 的前置背景：
  - `attention_basic_knowledge/00_序列建模与Attention前置背景.ipynb`
  - 理解固定长度表示的局限，以及序列中不同 token 需要交换信息的原因。
- 系统学习 Query、Key、Value：
  - `attention_basic_knowledge/02_Query_Key_Value是什么.ipynb`
  - 理解 Q 用于提出当前 token 的查询，K 用于描述可被匹配的特征，V 用于携带最终被汇总的信息。
  - 掌握 `QK^T` 生成注意力分数、Softmax 生成权重、权重与 V 相乘得到上下文化表示的完整流程。
- 学习 Self-Attention 的标量形式与矩阵形式：
  - `attention_basic_knowledge/03_Self-Attention自注意力机制.ipynb`
  - `attention_basic_knowledge/04_Self-Attention矩阵形式与形状变化.ipynb`
  - 理解 Q、K、V 的形状以及注意力权重矩阵 `N×N` 的含义。
  - 理解缩放因子 `sqrt(d_k)` 用于控制点积数值范围，避免 Softmax 过早饱和。
- 从可视化视角重新梳理 Attention 与 Transformer：
  - `attention_basic_knowledge/05_用3Blue1Brown视角重讲Attention与Transformer.ipynb`

### 2026-08-12

- 学习 Multi-Head Attention：
  - `attention_basic_knowledge/06_Multi-Head_Attention多头注意力机制.ipynb`
  - 理解多头不是无限扩大维度，而是把总表示维度分到多个子空间，并行学习不同关系后再拼接。
- 学习位置编码：
  - `attention_basic_knowledge/07_位置编码_Positional_Encoding.ipynb`
  - 理解 Self-Attention 本身不包含顺序信息，因此需要额外注入 token 的位置。
  - 学习固定正余弦位置编码的构造思路。
- 学习 Transformer 中的 FFN、残差连接与 LayerNorm：
  - `attention_basic_knowledge/08_FeedForward_Network前馈神经网络.ipynb`
  - `attention_basic_knowledge/09_残差连接与LayerNorm.ipynb`
  - 理解 FFN 本质上是对每个 token 独立使用、参数共享的 MLP。
  - 理解残差连接提供信息与梯度的直接通路，LayerNorm 在单个 token 的特征维度上进行归一化。
- 完成 Transformer Encoder 整体结构学习：
  - `attention_basic_knowledge/10_Transformer_Encoder整体结构.ipynb`
  - 串联 Multi-Head Attention、残差连接、LayerNorm 与 FFN，掌握 Encoder 的输入输出形状与信息流。

### 2026-08-14

- 制定 ViT 实践学习路径，创建 `vision_transformer_practice/` 目录。
- 运行 torchvision 预训练 ViT，建立对完整图像分类流程的整体认识：
  - `vision_transformer_practice/01_torchvision_ViT预训练推理.ipynb`
- 学习 ViT 的 Patch Embedding：
  - `vision_transformer_practice/02_ViT输入与PatchEmbedding.ipynb`
  - 理解 RGB 图像如何通过 `kernel_size=stride=patch_size` 的 `Conv2d` 被切分并投影为 patch tokens。
  - 掌握卷积权重形状 `D×C×P×P`、输出 token 形状以及参数量的计算方法。
  - 验证 Patch Embedding 的卷积参数能够接收梯度并参与反向传播。
- 学习 ViT 输入组装：
  - `vision_transformer_practice/03_ViT输入组装_CLS与位置编码.ipynb`
  - 理解每个样本拥有一个 CLS token，并由分类头读取第 0 个位置的上下文化表示。
  - 区分固定正余弦位置编码与 ViT 常用的可学习位置编码。
  - 理解输入分辨率变化时对位置编码网格进行插值的基本流程。
- 学习 Attention 的反向传播：
  - `attention_basic_knowledge/11_Attention反向传播概述.ipynb`
  - `attention_basic_knowledge/12_Attention反向传播逐步推导.ipynb`
  - `attention_basic_knowledge/13_用PyTorch观察Attention梯度.ipynb`
  - 从输出 `O=AV` 开始，理解梯度如何依次流向 V、注意力权重 A、Softmax、Q、K 以及投影矩阵。
- 拆解 Tiny ViT EncoderBlock：
  - `vision_transformer_practice/04_TinyViT_EncoderBlock结构拆解.ipynb`
  - 理解 ViT EncoderBlock 与 Transformer Encoder 的对应关系。
  - 复习多头自注意力、FFN、残差连接、LayerNorm 的结构与参数计算。
- 完成 Tiny ViT 的 CIFAR-10 训练、评估与训练策略实验：
  - 最佳验证准确率约为 `89.74%`，测试准确率约为 `89.57%`。
  - 整理训练曲线、Early Stopping、Label Smoothing、数据增强与 Mixup。
- 对照阅读 `vit-pytorch` 基础 ViT 源码：
  - `vision_transformer_practice/08_vit-pytorch基础ViT源码对照.ipynb`
  - 建立自研模块与开源模块的对应关系，跟踪关键 shape，并完成 CLS/mean pooling 小实验。
- 对比 CNN 与 ViT 的图像建模方式：
  - `vision_transformer_practice/09_CNN与ViT结构对比.ipynb`
  - 从特征表示、局部感受野、全局交互、归纳偏置和计算复杂度理解两种架构。
  - 通过局部扰动实验观察 CNN 第一层、Patch Embedding 和 Encoder Block 的信息传播范围。
- 可视化训练后 TinyViT 的注意力：
  - `vision_transformer_practice/10_TinyViT_AttentionMap可视化.ipynb`
  - 提取 `B x L x H x N x N` 注意力权重，还原 CLS 对 64 个 patches 的空间关注分布。
  - 对比首层、末层和不同注意力头，并使用注意力熵观察分布的集中程度。
- 组合多层注意力并进行遮挡验证：
  - `vision_transformer_practice/11_TinyViT_AttentionRollout与遮挡验证.ipynb`
  - 将残差路径加入注意力矩阵，按传播顺序累计 6 层 Attention Rollout。
  - 对比 Raw Attention 与 Rollout，并通过遮挡高低得分 patches 检查模型响应。
- 准备 CNN 与 TinyViT 的公平对照实验：
  - `vision_transformer_practice/12_CNN与TinyViT公平对照实验准备.ipynb`
  - CNN 参数量为 `4,692,426`，与 TinyViT 的 `4,771,082` 相差约 `1.649%`。
  - 两者复用相同 CIFAR-10 划分、增强、AdamW、交叉熵、Mixup、Warmup、余弦退火与 Early Stopping。
  - CNN 使用独立训练与评估入口，避免覆盖 TinyViT 的 checkpoint 和结果。

### 2026-08-16

- 在 `vision_transformer_practice/04_TinyViT_EncoderBlock结构拆解.ipynb` 中补充纸上推导练习参考答案：不运行代码，从 shape 出发手推 EncoderBlock 中 MHA 的 Q/K/V、注意力矩阵、残差连接与 FFN 的信息流与参数量，加深对模型内部机制的理解。

### 2026-08-17

- 优化器进阶补充 AdamW：
  - `neural_network_basic_knowledge/02_mlp_training_theory/03_optimizers_and_lr/06-01_优化器进阶_RMSProp与Adam再理解.ipynb`
  - 理解 AdamW 与 Adam 的核心区别：把权重衰减从自适应梯度中解耦（先做 Adam 更新、再独立衰减权重）。
  - 理解 bias 与 LayerNorm 的缩放/偏移参数通常不参与权重衰减，以及 Transformer / ViT 常用 AdamW 的原因。
- 工程化 Tiny ViT 的 CIFAR-10 训练代码：
  - 新增模型与组件 `vision_transformer_practice/vit/`：`config.py`、`data.py`、`embedding.py`、`encoder.py`、`model.py`、`training.py`、`evaluation.py` 等。
  - 新增入口脚本 `vit/train_tiny_vit.py`、`vit/evaluate_tiny_vit.py`、`vit/infer.py`，支持完整训练、可视化评估与单图预测。
  - 将训练轮数从 100 提升至 500，配合 AdamW 与余弦退火等策略提高训练充分性。
  - 新增训练曲线读图指南：`vision_transformer_practice/05_训练曲线读图指南.ipynb`，配套分析 ViT 训练效果并总结。
  - 新增 `.gitignore`，排除大文件、训练 checkpoints、结果与虚拟环境等。

### 2026-08-18

- 将 Mixup 数据增强应用到 TinyViT 训练中，防止过拟合；同步更新 `05_训练曲线读图指南.ipynb` 分析对应训练效果。
- 新增柱状图脚本 `vision_transformer_practice/vit/plot_performance_evaluation.py`：展示错误 top10、每个标签正确率与常见错误。

### 2026-08-19

- 完成正则化与数据增强主题 notebook：
  - `vision_transformer_practice/06_正则化技术_EarlyStopping与LabelSmoothing.ipynb` — Early Stopping 与 Label Smoothing 的原理、实现及配合使用方法。
  - `vision_transformer_practice/07_数据增强原理与实践.ipynb` — 四大类增强方法及其在项目中的应用，并补充实践观察与小结。
- 新增 224×224 ViT 放大训练项目：
  - `vision_transformer_practice/vit_224/`：`config.py`、`data.py`、`embedding.py`、`encoder.py`、`model.py`、`training.py`、`evaluation.py` 与 `train_vit224.py` / `evaluate_vit224.py`。
  - 在 CIFAR-10 上进行放大分辨率训练，结果单独存放于 `results_224/` 与 `checkpoints_224/`。
- 阅读并转译 ViT 论文，整理背景调研：
  - `vision_transformer_practice/ViT论文转译/`：`overview.md`、`background.md`、`translation.md`、`notes.md` 与 `citations.md`。
  - 分析 ViT 的核心贡献与创新点，整理相关引用与术语速查。

### 2026-08-21

- 实现 CNN 公平对照模型：
  - `vision_transformer_practice/cnn_baseline/`：`config.py`、`model.py`。
  - `vision_transformer_practice/compare_cnn/`：`train_cnn_baseline.py`、`evaluate_cnn_baseline.py`。
  - 复用与 TinyViT 相同的训练流程（优化器、数据增强、调度等），checkpoint 与结果独立存放，确保模型间公平对比。
- 开始 Swin Transformer 基础学习，创建 `swin_transformer_basic_knowledge/`：
  - `01_SwinTransformer概述与四阶段结构.ipynb` — 理解 Window Attention、Shifted Window、Patch Merging 与四阶段层级结构。
  - `02_WindowAttention窗口划分与QKV形状.ipynb` — 跟踪窗口划分、QKV、多头注意力与窗口还原的完整 shape。

### 2026-08-24

- 完善 Swin Transformer 基础笔记，补完核心理论：
  - `03_ShiftedWindow移动窗口与AttentionMask.ipynb` — W-MSA 与 SW-MSA、循环移位、Attention Mask 与跨窗口信息传播。
  - `04_相对位置偏置_RelativePositionBias.ipynb` — 相对位移、偏置表、索引矩阵及其与 Attention logits 的结合。
  - `05_PatchMerging层级下采样.ipynb` — 2×2 token 合并、通道压缩与层级下采样。
  - `06_SwinBlock完整结构.ipynb` — LayerNorm、残差连接、MLP 与 W-MSA/SW-MSA Block 的完整组成。
  - `07_SwinTransformer完整结构与复盘.ipynb` — 串联四个 Stage 与分类头，并与 CNN/ViT 对比。
  - 新增 `images/` 示意图，重构各阶段结构与 Patch Merging 过程的讲解。
- 新增 NLP Embedding 基础，创建 `nlp_embedding_basic_knowledge/`：
  - `01_Tokenization文本如何切成Token.ipynb`
  - `02_词表与Token_ID.ipynb`
  - `03_One-hot与Embedding矩阵.ipynb`
  - 理解文本如何切分成 token、token 如何映射为整数 ID、整数 ID 如何通过 Embedding 矩阵变成向量。
- 整理 `vision_transformer_practice` 代码结构：重命名脚本并更新导入路径，将训练/评估相关代码包化进 `vit/`、`vit_224/` 等模块。

### 2026-09-01

- 创建 `swin_transformer_practice/`，进入 Swin Transformer 的 CIFAR-100 实践：
  - `01_pretrained_swin_t_inference_and_shape_trace.py` — 运行 torchvision 预训练 Swin-T，并把真实输出与理论 shape 对齐。
  - 重构 CIFAR-100 数据加载与模型配置：新增 `swin/` 包（`config.py`、`data.py`、`embedding.py`、`encoder.py`、`model.py`、`optimization.py`、`training.py`、`evaluation.py`），集中管理路径与参数。
  - 在 `swin/data.py` 中实现 Mixup 与 CutMix 混合数据增强，更新配置以支持新参数。
  - 新增超参数搜索 `search_hyperparameters.py`：搜索学习率与 Weight Decay，调整训练轮数与标签平滑度，新增随机擦除概率；官方测试集不参与选择。
  - 新增 `train_custom_swin.py` 与 `evaluate_custom_swin.py`：使用最佳参数完整训练后在官方测试集上最终评估一次。

### 2026-09-02

- 将 Swin 实践扩展到 ImageNet：
  - 先以 ImageNet-1K 为目标改写为四阶段网络结构并调整架构以提升训练效果。
  - 因数据规模较大，随后重构为 ImageNet-100 支持，迁移为 `swin_transformer_imagenet100_practice/`。
  - 数据按 ImageFolder 组织（train/val 各 100 类），从 train 每类固定划出 50 张作为内部验证集，val 仅供最终评估。
- 进入 DETR 基础学习，创建 `detr_basic_knowledge/`：
  - `01_DETR概述与集合预测.ipynb` — 目标检测基本概念、DETR 整体结构、Object Queries、一对一集合预测与匈牙利匹配。
  - `02_DETR输入表示_Backbone与二维位置编码.ipynb` — 图片、padding mask、CNN 特征图、1×1 通道投影、图像 tokens 与二维位置编码的完整 shape。

### 2026-09-03

- 创建 `object_detection_basic_knowledge/`，先补齐目标检测共同基础：
  - `01_目标检测是什么_从分类到定位.ipynb`
  - `02_IoU边界框重叠度.ipynb`
  - `03_分类与边界框回归.ipynb`
  - 建立"图像分类→定位→目标检测、边界框与坐标表示、IoU、分类头与边界框回归头"的检测主线。
- 调整学习顺序：`detr_basic_knowledge/` 的 DETR 笔记保留为后续内容，先完成目标检测共同基础再回到 DETR。

## 当前阶段目标

- 能熟练理解 Tensor 的 shape、dtype、device。
- 能独立解释 `dim`、`keepdim`、`unsqueeze`、`transpose`、`permute`、`cat`、`stack`。
- 能理解 PyTorch 自动求导和标准训练循环。
- 能用 PyTorch 独立完成线性回归训练流程。
- 能推导常见激活函数公式并选择合适的激活函数。
- 能理解并应用 Xavier/He 参数初始化方法。
- 能解释感知机与 MLP 的区别，以及为什么 MLP 能解决 XOR 问题。
- 能阐述损失函数、梯度下降、反向传播之间的逻辑关系。
- 能描述完整的神经网络训练流程，识别过拟合与欠拟合。
- 能对比 SGD、Momentum、Adam、AdaGrad 的特点与使用场景。
- 能解释 EMA（指数加权移动平均）在优化器中的作用。
- 能用 MLP 独立完成 MNIST 手写数字识别全流程。
- 能解释 CNN 为什么适合图像任务（局部感知、权重共享）。
- 能理解并计算卷积层输出尺寸、多通道卷积的参数量。
- 能区分最大池化与平均池化的作用和使用场景。
- 能画出 CNN 完整结构（卷积→激活→池化→展平→全连接）。
- 能复现 LeNet 网络结构。
- 能用 PyTorch 完成 CNN-MNIST 的训练、评估、保存和预测流程。
- 能解释 Q、K、V、缩放点积注意力和注意力权重矩阵的含义与形状变化。
- 能说明 Self-Attention 与 Multi-Head Attention 的区别，并计算每个 head 的维度。
- 能画出 Transformer Encoder 的完整结构，解释 FFN、残差连接与 LayerNorm 的作用。
- 能说明 Attention 反向传播时梯度经过 V、Softmax、Q、K 和投影矩阵的基本路径。
- 能解释 ViT 如何使用 Conv2d 将 RGB 图像转换为 patch tokens，并计算 Patch Embedding 参数量。
- 能解释 CLS token、可学习位置编码和位置编码插值在 ViT 中的作用。
- 能区分图像分类、单目标定位与目标检测，理解边界框坐标格式与归一化。
- 能从交集与并集出发计算两个 `xyxy` 框的 IoU，理解其取值范围、阈值与局限。
- 能解释一条检测结果为什么同时包含离散类别与连续坐标，以及分类头与边界框回归头的损失如何共同反向传播。
- 当前阶段：已完成目标检测共同基础的前三课（从分类到定位、IoU、分类与边界框回归），下一步进入正/负样本与目标分配、NMS、mAP 等检测流程环节，之后回到 DETR 的集合预测。
