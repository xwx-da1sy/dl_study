# 神经网络学习导航

这个目录按学习阶段和主题重新整理。主线是：

```text
基础概念 -> MLP 与训练理论 -> MNIST 实战 -> CNN 预留
```

## 01 基础概念

目录：[01_foundation_concepts](01_foundation_concepts)

### 01 神经网络导入

1. [什么是神经网络](01_foundation_concepts/01_neural_network_intro/01_什么是神经网络.ipynb)

### 02 激活函数

1. [Sigmoid 函数公式推导](01_foundation_concepts/02_activation_functions/01-01_Sigmoid函数公式推导.ipynb)
2. [Tanh 函数公式推导](01_foundation_concepts/02_activation_functions/01-02_Tanh函数公式推导.ipynb)
3. [ReLU 函数公式推导](01_foundation_concepts/02_activation_functions/01-03_ReLU函数公式推导.ipynb)
4. [Softmax 函数公式推导](01_foundation_concepts/02_activation_functions/01-04_Softmax函数公式推导.ipynb)

### 03 参数初始化

1. [参数初始化](01_foundation_concepts/03_parameter_initialization/01-05_参数初始化.ipynb)
2. [参数初始化辅助脚本](01_foundation_concepts/03_parameter_initialization/parameter_initialization.py)

## 02 MLP 与训练理论

目录：[02_mlp_training_theory](02_mlp_training_theory)

### 01 MLP 核心概念

1. [感知机与多层感知机](02_mlp_training_theory/01_mlp_core/02_感知机与多层感知机.ipynb)
2. [MLP 结构设计](02_mlp_training_theory/01_mlp_core/09_MLP结构设计_输入层隐藏层输出层.ipynb)

### 02 训练过程

1. [损失函数与梯度下降](02_mlp_training_theory/02_training_process/03_损失函数与梯度下降.ipynb)
2. [常用损失函数](02_mlp_training_theory/02_training_process/03-01_常用损失函数.ipynb)
3. [反向传播与链式法则](02_mlp_training_theory/02_training_process/04_反向传播与链式法则.ipynb)
4. [神经网络训练流程](02_mlp_training_theory/02_training_process/05_神经网络训练流程.ipynb)

### 03 优化器与学习率

1. [指数加权移动平均](02_mlp_training_theory/03_optimizers_and_lr/05-01_指数加权移动平均.ipynb)
2. [优化器：SGD、Momentum、Adam](02_mlp_training_theory/03_optimizers_and_lr/06_优化器_SGD_Momentum_Adam.ipynb)
3. [AdaGrad：从 SGD 到自适应学习率](02_mlp_training_theory/03_optimizers_and_lr/06-00_AdaGrad从SGD到自适应学习率.ipynb)
4. [优化器进阶：RMSProp 与 Adam 再理解](02_mlp_training_theory/03_optimizers_and_lr/06-01_优化器进阶_RMSProp与Adam再理解.ipynb)
5. [学习率调整策略](02_mlp_training_theory/03_optimizers_and_lr/06-02_学习率调整策略.ipynb)

### 04 泛化与正则化

1. [泛化能力与训练曲线诊断](02_mlp_training_theory/04_generalization_regularization/06-03_泛化能力与训练曲线诊断.ipynb)
2. [过拟合与正则化](02_mlp_training_theory/04_generalization_regularization/07_过拟合与正则化.ipynb)
3. [正则化方法深入理解](02_mlp_training_theory/04_generalization_regularization/07-01_正则化方法深入理解.ipynb)

### 05 归一化与结构补充

1. [Batch Normalization 批量归一化](02_mlp_training_theory/05_normalization_and_architecture/08_BatchNormalization批量归一化.ipynb)
2. [BatchNorm 深入理解](02_mlp_training_theory/05_normalization_and_architecture/08-01_BatchNorm深入理解.ipynb)

## 03 MNIST 实战

目录：[03_mnist_practice](03_mnist_practice)

从概念到代码逐步进入：

1. [MNIST 手写数字分类概念导入](03_mnist_practice/00_MNIST手写数字分类_概念导入.ipynb)
2. [MLP 手写数字识别 API 预备知识](03_mnist_practice/01_MLP手写数字识别_API预备知识.ipynb)

相关脚本和资源也保留在这个目录中：

- [数据加载脚本](03_mnist_practice/01_load_mnist_data.py)
- [MLP 模型定义脚本](03_mnist_practice/02_define_mlp_model.py)
- [训练脚本](03_mnist_practice/03_train_mnist.py)
- [预测脚本](03_mnist_practice/04_predict_mnist.py)
- [手写数字绘制预测脚本](03_mnist_practice/05_draw_digit_predict.py)
- `data/`：MNIST 数据缓存
- `models/`：训练得到的模型权重

## 04 CNN 卷积神经网络

目录：[04_cnn_convolutional_neural_network](04_cnn_convolutional_neural_network)

这个目录先作为后续 CNN 学习的预留位置，目前不展开内容。
