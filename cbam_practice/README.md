# CBAM 对照实验：CIFAR-10

实验比较同一个小型 CNN 加入 CBAM 前后的分类表现，用于学习 CBAM 在实际训练中的作用，不是原论文 ImageNet 实验的完整复现。

## 实验设计

- 数据：使用 `vision_transformer_practice/data` 中已有的 CIFAR-10。训练集按类别均衡划分为 45,000 张训练、5,000 张验证；官方 10,000 张测试图片只用于最终评估。
- 对照组：三层卷积 CNN；实验组：相同 CNN 的每个卷积块后加入 CBAM。两组相同随机种子、基础网络初始化、数据划分、每轮顺序与数据增强，以及训练配置。
- 模型按验证集准确率选择最佳轮次，再各自在测试集评估一次；记录损失、准确率、参数量、时间和 CBAM 首个 batch 的梯度范数。
- 默认训练 5 轮，Adam 学习率 0.001，batch size 128。单次短训练的结果受随机性和训练轮数影响，不足以证明普遍优势。CBAM 增加了参数和计算量，这也是结果解释的一部分。

## 运行

在项目根目录执行：

```powershell
python cbam_practice/compare_cbam.py
```

可用 `--epochs`、`--batch-size`、`--seed` 修改配置。结果存放于 `results/results.json`；两组最佳模型分别保存在 `results/baseline_best.pt` 和 `results/cbam_best.pt`。重复运行会覆盖同名结果，请先另存需要保留的实验。

实现遵循通道注意力后空间注意力的核心结构；空间分支未加入作者官方代码中的 BatchNorm，因此这是教学用的简化实现，非逐项复刻官方实现。
