# dl_study

该仓库是我的深度学习学习仓库。当前主线是参考黑马程序员《神经网络与深度学习》课程，配合 PyTorch 官方教程和个人练习 notebook，系统进入深度学习阶段。

## 学习目录

- `background_information/`：学习背景与路线规划。
- `pytorch_basic_knowledge/`：PyTorch 基础、Tensor、DataLoader、线性回归等内容。
- `neural_network_basic_knowledge/`：神经网络基础概念。

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
- 能对比 SGD、Momentum、Adam 的特点与使用场景。
- 下一步进入 `nn.Module` 实战：搭建完整 MLP 解决分类/回归任务。
