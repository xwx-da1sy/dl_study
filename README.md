# dl_study

该仓库是我的深度学习学习仓库。当前主线是参考黑马程序员《神经网络与深度学习》课程，配合 PyTorch 官方教程和个人练习 notebook，系统进入深度学习阶段。

## 学习目录

- `background_information/`：学习背景与路线规划。
- `pytorch_basic_knowledge/`：PyTorch 基础、Tensor、DataLoader、线性回归等内容。
- `neural_network_basic_knowledge/`：神经网络基础概念（激活函数、MLP、优化器、归一化、MNIST 实战等）。
- `cnn_basic_knowledge/`：CNN 卷积神经网络（图像基础、卷积层、池化层、网络结构、LeNet 等）。

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
- 下一步：使用 PyTorch 实现 CNN 图像分类实战。